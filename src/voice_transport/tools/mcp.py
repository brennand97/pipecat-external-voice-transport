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
    request_timeout_seconds: float = 15.0
    max_concurrent_calls: int = 2


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
        return [
            ToolDefinition(tool.name, tool.description or "", tool.inputSchema)
            for tool in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        session = await self._session_or_connect()
        async with self._calls:
            result = await asyncio.wait_for(
                session.call_tool(name, arguments),
                timeout=self.config.request_timeout_seconds,
            )
        return ToolResult(
            content=[item.model_dump(mode="json") for item in result.content],
            is_error=bool(result.isError),
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
                streams = await stack.enter_async_context(
                    streamable_http_client(self._require_url())
                )
            read_stream, write_stream = streams
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            self._stack = stack
            self._session = session
            return session

    def _require_url(self) -> str:
        if not self.config.url:
            raise ValueError(f"{self.config.transport} MCP servers require url")
        return self.config.url
