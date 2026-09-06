"""Provider-neutral asynchronous tool contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    content: list[dict[str, Any]]
    is_error: bool = False


class AsyncToolProvider(Protocol):
    """A lazily connected source of async tools."""

    async def list_tools(self) -> list[ToolDefinition]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...

    async def close(self) -> None: ...
