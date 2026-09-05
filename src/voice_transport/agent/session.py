"""Isolated contract for a single provider conversation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """A provider-independent event returned to transport orchestration."""

    type: str
    text: str | None = None


class AgentSession(Protocol):
    """Lifecycle contract between orchestration and agent features.

    Implementations own their Pipecat pipeline, runner, provider connection,
    feature-specific queues, and tasks. Session authentication, protocol state,
    audio ingress limits, and WebSocket handling remain outside this boundary.
    """

    async def start(self) -> None: ...

    async def push_audio(self, pcm: bytes) -> None: ...

    async def end_input(self) -> None: ...

    async def cancel(self) -> None: ...

    def events(self) -> AsyncIterator[AgentEvent]: ...

    async def close(self) -> None: ...
