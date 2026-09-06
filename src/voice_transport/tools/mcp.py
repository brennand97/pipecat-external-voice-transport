"""Async MCP tool provider backed by the official MCP Python SDK."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Literal

from .base import ToolDefinition, ToolResult

MCPTransport = Literal["stdio", "sse", "streamable_http"]


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    """One explicitly trusted MCP server endpoint or child process."""

    name: str
    transport: MCPTransport
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    # Resolved from an administrator-provided environment variable; never
    # accepted as a literal value in trusted tool configuration.
    bearer_token: str | None = None
    request_timeout_seconds: float = 15.0
    max_concurrent_calls: int = 2
    allowed_tools: frozenset[str] = frozenset()


@dataclass(slots=True)
class MCPToolProvider:
    """Lazy, async MCP client with SDK-managed protocol negotiation.

    The official SDK negotiates the highest mutually supported MCP protocol
    version during initialize. This provider supports every protocol version
    supported by that SDK, without blocking unrelated model sessions.
    """

    config: MCPServerConfig
    _stack: AsyncExitStack | None = None
    _session: Any = None
    _connect_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _calls: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self._calls = asyncio.Semaphore(self.config.max_concurrent_calls)

    async def list_tools(self) -> list[ToolDefinition]:
        session = await self._session_or_connect()
        result = await session.list_tools()
        definitions = []
        for tool in result.tools:
            if not self._is_allowed(tool.name):
                continue
            schema = getattr(tool, "input_schema", None)
            if not isinstance(schema, dict):
                raise ValueError(f"MCP tool {tool.name!r} has an invalid input schema")
            definitions.append(
                ToolDefinition(tool.name, tool.description or "", schema)
            )
        return definitions

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if not self._is_allowed(name):
            return ToolResult(
                content=[{"type": "text", "text": f"Tool is not allowed: {name}"}],
                is_error=True,
            )
        session = await self._session_or_connect()
        async with self._calls:
            # Keep streamable HTTP operations in the task that owns its
            # context; ``wait_for`` would create a child task.
            async with asyncio.timeout(self.config.request_timeout_seconds):
                result = await session.call_tool(name, arguments)
        return ToolResult(
            content=[item.model_dump(mode="json") for item in result.content],
            # MCP SDK v2 uses Pydantic snake_case attributes. Keep a legacy
            # fallback for compatible older SDK result objects.
            is_error=bool(
                getattr(result, "is_error", getattr(result, "isError", False))
            ),
        )

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None

    async def _session_or_connect(self):
        if self._session is not None:
            return self._session
        async with self._connect_lock:
            if self._session is not None:
                return self._session
            from mcp import ClientSession
            from mcp.client.sse import sse_client
            from mcp.client.stdio import StdioServerParameters, stdio_client
            from mcp.client.streamable_http import streamable_http_client

            stack = AsyncExitStack()
            if self.config.transport == "stdio":
                if not self.config.command:
                    raise ValueError("stdio MCP servers require command")
                streams = await stack.enter_async_context(
                    stdio_client(
                        StdioServerParameters(
                            command=self.config.command,
                            args=list(self.config.args),
                            env=self.config.env,
                        )
                    )
                )
            elif self.config.transport == "sse":
                streams = await stack.enter_async_context(
                    sse_client(self._require_url())
                )
            else:
                http_client = None
                if self.config.bearer_token:
                    from httpx2 import AsyncClient

                    http_client = await stack.enter_async_context(
                        AsyncClient(
                            headers={
                                "Authorization": f"Bearer {self.config.bearer_token}"
                            }
                        )
                    )
                streams = await stack.enter_async_context(
                    streamable_http_client(self._require_url(), http_client=http_client)
                )
            read_stream, write_stream = streams
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            self._stack = stack
            self._session = session
            return session

    def _is_allowed(self, name: str) -> bool:
        return any(
            name.startswith(pattern[:-1]) if pattern.endswith("*") else name == pattern
            for pattern in self.config.allowed_tools
        )

    def _require_url(self) -> str:
        if not self.config.url:
            raise ValueError(f"{self.config.transport} MCP servers require url")
        return self.config.url
