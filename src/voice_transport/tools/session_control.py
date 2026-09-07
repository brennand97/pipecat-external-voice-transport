"""Server-owned tools for controlling the current transport session."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .base import ToolDefinition, ToolResult

END_SESSION_TOOL = "transport__EndSession"


@dataclass(slots=True)
class SessionControlToolProvider:
    """Expose a session-local end control without involving Home Assistant."""

    end_requested: asyncio.Event

    async def list_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                END_SESSION_TOOL,
                "End the current voice conversation only when the user explicitly "
                "asks to stop or end this conversation. Do not use this for "
                "commands to stop music, timers, lights, or other devices.",
                {"type": "object", "properties": {}, "additionalProperties": False},
            )
        ]

    async def call_tool(self, name: str, arguments: dict[str, object]) -> ToolResult:
        if name != END_SESSION_TOOL:
            return ToolResult(
                [{"type": "text", "text": f"Unknown tool: {name}"}], is_error=True
            )
        self.end_requested.set()
        return ToolResult([{"type": "text", "text": "Ending the conversation."}])

    async def close(self) -> None:
        """The event is owned by the transport handler."""
