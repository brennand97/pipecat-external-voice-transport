"""Small development-style contract for a single provider conversation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """A provider-independent event returned to transport orchestration."""

    type: str
    text: str | None = None


class DevelopmentSession(Protocol):
    """The only lifecycle contract the WebSocket/session layer will use.

    Implementations own their Pipecat pipeline, runner, queues, and provider
    connection. This deliberately resembles a small standalone Pipecat example:
    start it, feed native PCM, finish/cancel it, and consume its events.
    """

    async def start(self) -> None: ...

    async def push_audio(self, pcm: bytes) -> None: ...

    async def end_input(self) -> None: ...

    async def cancel(self) -> None: ...

    def events(self) -> AsyncIterator[AgentEvent]: ...

    async def close(self) -> None: ...
