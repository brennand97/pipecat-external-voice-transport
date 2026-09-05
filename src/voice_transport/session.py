"""Explicit, bounded lifecycle management for protocol sessions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum

from .protocol import ProtocolViolation, SessionStart


class SessionState(StrEnum):
    NEW = "new"
    READY = "ready"
    LISTENING = "listening"
    RESPONDING = "responding"
    CANCELLING = "cancelling"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass(slots=True)
class Session:
    start: SessionStart
    max_input_bytes: int
    state: SessionState = SessionState.NEW
    input_bytes: int = 0
    _input_ended: bool = False

    def mark_ready(self) -> None:
        self._transition({SessionState.NEW}, SessionState.READY)

    def add_audio(self, frame: bytes, max_frame_bytes: int) -> None:
        if self.state not in {SessionState.READY, SessionState.LISTENING}:
            raise ProtocolViolation(
                "invalid_state", "Audio is not accepted in the current state."
            )
        if not frame or len(frame) % 2:
            raise ProtocolViolation(
                "invalid_audio_frame",
                "PCM frames must have a non-zero even byte length.",
            )
        if len(frame) > max_frame_bytes:
            raise ProtocolViolation(
                "audio_frame_too_large", "PCM frame exceeds the configured limit."
            )
        if self.input_bytes + len(frame) > self.max_input_bytes:
            raise ProtocolViolation(
                "input_limit_exceeded", "Session input audio limit was reached."
            )
        self.input_bytes += len(frame)
        self.state = SessionState.LISTENING

    def end_input(self) -> None:
        if (
            self.state not in {SessionState.READY, SessionState.LISTENING}
            or self._input_ended
        ):
            raise ProtocolViolation(
                "invalid_state", "Input has already ended or is not active."
            )
        self._input_ended = True
        self.state = SessionState.RESPONDING

    def cancel(self) -> None:
        if self.state in {SessionState.FINISHED, SessionState.FAILED}:
            return
        self.state = SessionState.CANCELLING

    def finish(self) -> None:
        if self.state not in {SessionState.RESPONDING, SessionState.CANCELLING}:
            raise ProtocolViolation(
                "invalid_state", "Session cannot finish in the current state."
            )
        self.state = SessionState.FINISHED

    def fail(self) -> None:
        self.state = SessionState.FAILED

    def _transition(self, allowed: set[SessionState], target: SessionState) -> None:
        if self.state not in allowed:
            raise ProtocolViolation(
                "invalid_state", "Invalid session state transition."
            )
        self.state = target


@dataclass(slots=True)
class SessionRegistry:
    max_sessions: int
    _sessions: dict[str, Session] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def create(self, start: SessionStart, max_input_bytes: int) -> Session:
        async with self._lock:
            if start.session_id in self._sessions:
                raise ProtocolViolation(
                    "duplicate_session", "A session with this ID is already active."
                )
            if len(self._sessions) >= self.max_sessions:
                raise ProtocolViolation(
                    "session_capacity_exceeded", "The server is at session capacity."
                )
            session = Session(start, max_input_bytes)
            self._sessions[start.session_id] = session
            return session

    async def remove(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    @property
    def active_count(self) -> int:
        return len(self._sessions)
