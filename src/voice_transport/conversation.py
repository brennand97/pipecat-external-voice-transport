"""Persistent provider-neutral turn session actor."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol
from uuid import uuid4

from .agent.session import AgentEvent


class TurnInput(StrEnum):
    AUDIO = "audio"
    TEXT = "text"


class ConversationError(ValueError):
    """A command is invalid for the current conversation state."""


class PersistentAgent(Protocol):
    async def start(self) -> None: ...
    async def submit_audio(self, turn_id: str, pcm: bytes) -> None: ...
    async def submit_text(self, turn_id: str, text: str) -> None: ...
    async def end_turn(self, turn_id: str) -> None: ...
    async def interrupt(self) -> None: ...
    async def close(self) -> None: ...
    def events(self) -> AsyncIterator[AgentEvent]: ...


@dataclass(frozen=True, slots=True)
class ConversationEvent:
    type: str
    turn_id: str
    response_id: str | None = None
    text: str | None = None
    audio: bytes | None = None
    sample_rate: int | None = None
    channels: int | None = None
    source: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_result: list[dict[str, Any]] | None = None
    tool_arguments_truncated: bool = False
    tool_result_truncated: bool = False
    is_error: bool | None = None


class ConversationActor:
    """Serialize all turn state and server-owned interruption decisions."""

    def __init__(self, provider: PersistentAgent) -> None:
        self._provider = provider
        self._events: asyncio.Queue[ConversationEvent | None] = asyncio.Queue(128)
        self._lock = asyncio.Lock()
        self._open_turn: tuple[str, TurnInput] | None = None
        self._last_ended_turn: str | None = None
        self._response_turn: str | None = None
        self._response_number = 0
        self._active_response_id: str | None = None
        self._last_response_id: str | None = None
        self._last_response_turn: str | None = None
        self._accept_response_events = False
        self._used_turn_ids: set[str] = set()
        self._text_received = False
        self._closed = False
        self._events_finished = False
        self._event_task: asyncio.Task[None] | None = None

    @property
    def open_turn_id(self) -> str | None:
        return self._open_turn[0] if self._open_turn else None

    @property
    def active_response_id(self) -> str | None:
        return self._active_response_id

    @property
    def effective_profile(self) -> str | None:
        return getattr(self._provider, "effective_profile", None)

    @property
    def effective_tool_names(self) -> tuple[str, ...]:
        return getattr(self._provider, "effective_tool_names", ())

    async def start(self) -> None:
        await self._provider.start()
        self._event_task = asyncio.create_task(self._pump_provider_events())

    async def start_turn(self, turn_id: str, input_type: TurnInput) -> None:
        async with self._lock:
            self._ensure_open()
            if not turn_id or turn_id in self._used_turn_ids:
                raise ConversationError("turn_id must be non-empty and unique")
            if self._open_turn is not None:
                raise ConversationError("another turn is already open")
            if input_type is TurnInput.TEXT:
                await self._interrupt_active_response()
            self._used_turn_ids.add(turn_id)
            self._open_turn = (turn_id, input_type)
            self._text_received = False

    async def submit_audio(self, turn_id: str, pcm: bytes) -> None:
        async with self._lock:
            self._require_turn(turn_id, TurnInput.AUDIO)
            await self._provider.submit_audio(turn_id, pcm)

    async def submit_text(self, turn_id: str, text: str) -> None:
        async with self._lock:
            self._require_turn(turn_id, TurnInput.TEXT)
            if self._text_received:
                raise ConversationError("text turn already contains input")
            if not text or len(text.encode()) > 4_000:
                raise ConversationError("text must be between 1 and 4,000 bytes")
            self._text_received = True
            await self._put(
                ConversationEvent(
                    "user.transcript.final",
                    turn_id,
                    text=text,
                    source="client_text",
                )
            )
            await self._provider.submit_text(turn_id, text)

    async def end_turn(self, turn_id: str) -> None:
        async with self._lock:
            self._require_turn(turn_id, None)
            if self._open_turn[1] is TurnInput.TEXT and not self._text_received:
                raise ConversationError("text turn requires input.text")
            self._open_turn = None
            self._last_ended_turn = turn_id
            self._accept_response_events = True
            await self._provider.end_turn(turn_id)

    async def cancel_response(self, response_id: str) -> None:
        async with self._lock:
            if response_id != self._active_response_id:
                raise ConversationError("response is not active")
            await self._interrupt_active_response()

    async def cancel(self) -> None:
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            # ``asyncio.wait_for`` runs a coroutine in a child task. Streamable
            # HTTP MCP contexts must exit in the task that entered them, so use
            # a current-task timeout while retaining the same bounded close.
            async with asyncio.timeout(3):
                await self._provider.close()
        except TimeoutError:
            pass
        if self._event_task is not None:
            self._event_task.cancel()
            try:
                await asyncio.wait_for(self._event_task, timeout=3)
            except (asyncio.CancelledError, TimeoutError, Exception):
                pass
        await self._finish_events()

    async def events(self) -> AsyncIterator[ConversationEvent]:
        while event := await self._events.get():
            yield event

    async def _interrupt_active_response(self) -> None:
        if self._active_response_id is None or self._response_turn is None:
            return
        turn_id = self._response_turn
        response_id = self._active_response_id
        # Advance actor state before awaiting the provider. Any provider frames
        # racing with cancellation are fenced out by the cleared response.
        self._response_turn = None
        self._active_response_id = None
        self._accept_response_events = False
        await self._provider.interrupt()
        await self._put(
            ConversationEvent("assistant.interrupted", turn_id, response_id=response_id)
        )

    async def _pump_provider_events(self) -> None:
        try:
            async for event in self._provider.events():
                await self._handle_provider_event(event)
        finally:
            await self._finish_events()

    async def _handle_provider_event(self, event: AgentEvent) -> None:
        async with self._lock:
            if event.type == "user.speech_started":
                new_turn = self.open_turn_id
                if new_turn is not None:
                    await self._interrupt_active_response()
                    self._last_ended_turn = new_turn
                    self._accept_response_events = True
                    await self._put(ConversationEvent(event.type, new_turn))
                return
            if event.type.startswith("user.transcript"):
                turn_id = self.open_turn_id or self._last_ended_turn
                if turn_id is not None:
                    await self._put(
                        ConversationEvent(
                            event.type,
                            turn_id,
                            text=event.text,
                            source="provider_audio",
                        )
                    )
                return
            if event.type.startswith("assistant.tool_call_"):
                turn_id = (
                    self._response_turn
                    or self._last_response_turn
                    or self._last_ended_turn
                )
                response_id = self._active_response_id or self._last_response_id
                if turn_id is None or response_id is None:
                    return
                await self._put(
                    ConversationEvent(
                        event.type,
                        turn_id,
                        response_id=response_id,
                        tool_call_id=event.tool_call_id or str(uuid4()),
                        tool_name=event.tool_name,
                        tool_arguments=event.tool_arguments,
                        tool_result=event.tool_result,
                        tool_arguments_truncated=event.tool_arguments_truncated,
                        tool_result_truncated=event.tool_result_truncated,
                        is_error=event.is_error,
                    )
                )
                return
            if event.type == "assistant.response_started":
                if not self._accept_response_events:
                    return
                turn_id = self._last_ended_turn or self.open_turn_id
                if turn_id is None:
                    return
                self._response_number += 1
                self._active_response_id = str(self._response_number)
                self._response_turn = turn_id
            if self._active_response_id is None or self._response_turn is None:
                return
            response_id = self._active_response_id
            turn_id = self._response_turn
            await self._put(
                ConversationEvent(
                    event.type,
                    turn_id,
                    response_id=response_id,
                    text=event.text,
                    audio=event.audio,
                    sample_rate=event.sample_rate,
                    channels=event.channels,
                )
            )
            if event.type == "assistant.response_finished":
                self._last_response_id = self._active_response_id
                self._last_response_turn = self._response_turn
                self._active_response_id = None
                self._response_turn = None
                # A Realtime function call may complete after its spoken
                # preamble and trigger a follow-up response for the same user
                # turn. Keep that turn eligible until a new input interrupts
                # it or the session closes.

    async def _finish_events(self) -> None:
        if self._events_finished:
            return
        self._events_finished = True
        try:
            await asyncio.wait_for(self._events.put(None), timeout=1)
        except TimeoutError:
            pass

    async def _put(self, event: ConversationEvent) -> None:
        try:
            await asyncio.wait_for(self._events.put(event), timeout=1)
        except TimeoutError as err:
            raise RuntimeError("conversation event consumer is not keeping up") from err

    def _require_turn(self, turn_id: str, input_type: TurnInput | None) -> None:
        self._ensure_open()
        if self._open_turn is None or self._open_turn[0] != turn_id:
            raise ConversationError("turn is not active")
        if input_type is not None and self._open_turn[1] is not input_type:
            raise ConversationError("turn input type does not match")

    def _ensure_open(self) -> None:
        if self._closed:
            raise ConversationError("conversation is closed")
