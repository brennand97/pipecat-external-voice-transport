"""Deterministic provider session used without a provider credential."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from .session import AgentEvent


class FakeAgentSession:
    """Minimal drop-in stand-in for a Pipecat-backed agent session."""

    def __init__(self) -> None:
        self._events: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        self._started = False
        self.input_bytes = 0

    async def start(self) -> None:
        self._started = True

    async def push_audio(self, pcm: bytes) -> None:
        if not self._started:
            raise RuntimeError("fake session has not started")
        self.input_bytes += len(pcm)

    async def end_input(self) -> None:
        await self._events.put(AgentEvent("assistant.response_started"))
        await self._events.put(
            AgentEvent(
                "assistant.text.final",
                "External Transport test session received audio.",
            )
        )
        await self._events.put(AgentEvent("assistant.response_finished"))
        await self._events.put(None)

    async def cancel(self) -> None:
        await self._events.put(None)

    async def close(self) -> None:
        await self._events.put(None)

    async def events(self) -> AsyncIterator[AgentEvent]:
        while event := await self._events.get():
            yield event
