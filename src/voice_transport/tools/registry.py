"""Concurrent discovery and dispatch across trusted async tool providers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .base import AsyncToolProvider, ToolDefinition, ToolResult


@dataclass(slots=True)
class ToolRegistry:
    """A session-scoped registry; duplicate model-facing names fail closed."""

    providers: tuple[AsyncToolProvider, ...]
    _tools: dict[str, AsyncToolProvider] = field(default_factory=dict)
    _definitions: list[ToolDefinition] = field(default_factory=list)
    _ready: bool = False

    async def discover(self) -> list[ToolDefinition]:
        if self._ready:
            return list(self._definitions)
        discovered = await asyncio.gather(
            *(provider.list_tools() for provider in self.providers)
        )
        for provider, tools in zip(self.providers, discovered, strict=True):
            for tool in tools:
                if tool.name in self._tools:
                    raise ValueError(f"duplicate tool name: {tool.name}")
                self._tools[tool.name] = provider
                self._definitions.append(tool)
        self._ready = True
        return list(self._definitions)

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if not self._ready:
            await self.discover()
        provider = self._tools.get(name)
        if provider is None:
            return ToolResult(
                content=[{"type": "text", "text": f"Unknown tool: {name}"}],
                is_error=True,
            )
        return await provider.call_tool(name, arguments)

    async def close(self) -> None:
        await asyncio.gather(*(provider.close() for provider in self.providers))
