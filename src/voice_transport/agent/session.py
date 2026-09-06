"""Isolated contract for a single provider conversation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """A provider-independent event returned to transport orchestration."""

    type: str
    text: str | None = None
    audio: bytes | None = None
    sample_rate: int | None = None
    channels: int | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_result: list[dict[str, Any]] | None = None
    tool_arguments_truncated: bool = False
    tool_result_truncated: bool = False
    is_error: bool | None = None


class AgentSession(Protocol):
    """Lifecycle contract between orchestration and agent features.

    Implementations own their Pipecat pipeline, runner, provider connection,
    feature-specific queues, and tasks. Session authentication, protocol state,
    audio ingress limits, and WebSocket handling remain outside this boundary.
    """

    async def start(self) -> None: ...

    async def submit_audio(self, turn_id: str, pcm: bytes) -> None: ...

    async def submit_text(self, turn_id: str, text: str) -> None: ...

    async def end_turn(self, turn_id: str) -> None: ...

    async def interrupt(self) -> None: ...

    async def cancel(self) -> None: ...

    def events(self) -> AsyncIterator[AgentEvent]: ...

    async def close(self) -> None: ...
