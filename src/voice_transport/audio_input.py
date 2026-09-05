"""Bounded, ordered in-memory native PCM handoff for a session adapter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from .protocol import ProtocolViolation
from .session import Session

_END = object()


class PCMInput:
    """Preserve incoming PCM order while applying backpressure at the WebSocket.

    Frames are held only in this bounded, per-session queue. Consumers receive
    every pre-roll frame before subsequently received live audio; nothing is
    written to disk or retained after it is consumed.
    """

    def __init__(self, session: Session, max_frame_bytes: int, max_frames: int) -> None:
        self._session = session
        self._max_frame_bytes = max_frame_bytes
        self._queue: asyncio.Queue[bytes | object] = asyncio.Queue(maxsize=max_frames)
        self._closed = False

    async def put(self, frame: bytes) -> None:
        if self._closed:
            raise ProtocolViolation("invalid_state", "Input is already closed.")
        self._session.add_audio(frame, self._max_frame_bytes)
        # Waiting here is intentional: it propagates a slow provider/adapter
        # back to the client instead of accumulating unbounded microphone PCM.
        await self._queue.put(frame)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._queue.put(_END)

    async def frames(self) -> AsyncIterator[bytes]:
        while True:
            item = await self._queue.get()
            if item is _END:
                return
            assert isinstance(item, bytes)
            yield item
