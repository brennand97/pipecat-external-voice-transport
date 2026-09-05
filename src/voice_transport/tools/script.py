"""Explicitly configured asynchronous JSON-line script tools."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from .base import ToolDefinition, ToolResult


@dataclass(frozen=True, slots=True)
class ScriptToolConfig:
    """Trusted script configuration supplied by deployment, never by the model."""

    name: str
    command: tuple[str, ...]
    timeout_seconds: float = 15.0
    max_concurrent_calls: int = 2


@dataclass(slots=True)
class ScriptToolProvider:
    """Lazy script adapter using a small JSON request/response contract.

    The script receives one JSON object on stdin and writes one JSON object to
    stdout. Discovery request: ``{"method":"tools.list"}``. Invocation request:
    ``{"method":"tools.call","name":"...","arguments":{...}}``.
    """

    config: ScriptToolConfig
    _calls: asyncio.Semaphore = field(init=False)
    _tools: list[ToolDefinition] | None = None
    _discovery_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        if not self.config.command:
            raise ValueError("script tool command must not be empty")
        if self.config.timeout_seconds <= 0 or self.config.max_concurrent_calls < 1:
            raise ValueError("script tool limits must be positive")
        self._calls = asyncio.Semaphore(self.config.max_concurrent_calls)

    async def list_tools(self) -> list[ToolDefinition]:
        if self._tools is not None:
            return list(self._tools)
        async with self._discovery_lock:
            if self._tools is None:
                response = await self._run({"method": "tools.list"})
                tools = response.get("tools")
                if not isinstance(tools, list):
                    raise ValueError("script tools.list response requires tools array")
                self._tools = [_parse_tool(item) for item in tools]
            return list(self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        async with self._calls:
            response = await self._run(
                {"method": "tools.call", "name": name, "arguments": arguments}
            )
        content = response.get("content")
        if not isinstance(content, list) or not all(
            isinstance(item, dict) for item in content
        ):
            raise ValueError("script tools.call response requires content object array")
        return ToolResult(content, is_error=bool(response.get("is_error", False)))

    async def close(self) -> None:
        """One-shot scripts have no retained process to close."""

    async def _run(self, request: dict[str, Any]) -> dict[str, Any]:
        process = await asyncio.create_subprocess_exec(
            *self.config.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _stderr = await asyncio.wait_for(
                process.communicate(json.dumps(request).encode()),
                timeout=self.config.timeout_seconds,
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(f"script tool {self.config.name} timed out") from None
        if process.returncode != 0:
            raise RuntimeError(f"script tool {self.config.name} exited unsuccessfully")
        try:
            response = json.loads(stdout)
        except (TypeError, json.JSONDecodeError) as err:
            raise ValueError("script tool returned invalid JSON") from err
        if not isinstance(response, dict):
            raise ValueError("script tool response must be an object")
        return response


def _parse_tool(value: object) -> ToolDefinition:
    if not isinstance(value, dict):
        raise ValueError("script tool definition must be an object")
    name = value.get("name")
    description = value.get("description")
    input_schema = value.get("input_schema")
    if (
        not isinstance(name, str)
        or not isinstance(description, str)
        or not isinstance(input_schema, dict)
    ):
        raise ValueError("script tool definition is invalid")
    return ToolDefinition(name, description, input_schema)
