"""Deterministic persistent provider used without provider credentials."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from .session import AgentEvent


class FakeAgentSession:
    """Persistent fake conversation supporting ordered audio and text turns."""

    def __init__(self) -> None:
        self._events: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        self._started = False
        self._closed = False
        self._text: dict[str, str] = {}
        self.input_bytes = 0

    @property
    def effective_profile(self) -> str | None:
        return None

    @property
    def effective_tool_names(self) -> tuple[str, ...]:
        return ()

    async def start(self) -> None:
        self._started = True

    async def submit_audio(self, turn_id: str, pcm: bytes) -> None:
        self._ensure_started()
        self.input_bytes += len(pcm)

    async def submit_text(self, turn_id: str, text: str) -> None:
        self._ensure_started()
        self._text[turn_id] = text

    async def end_turn(self, turn_id: str) -> None:
        self._ensure_started()
        text = self._text.pop(turn_id, None)
        await self._events.put(AgentEvent("assistant.response_started"))
        await self._events.put(
            AgentEvent(
                "assistant.text.final",
                f"External Transport test received: {text}"
                if text is not None
                else "External Transport test session received audio.",
            )
        )
        await self._events.put(AgentEvent("assistant.response_finished"))

    async def interrupt(self) -> None:
        """The actor emits the normalized interruption event."""

    async def cancel(self) -> None:
        await self.close()

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self._events.put(None)

    async def events(self) -> AsyncIterator[AgentEvent]:
        while event := await self._events.get():
            yield event

    def _ensure_started(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("fake session is not active")
