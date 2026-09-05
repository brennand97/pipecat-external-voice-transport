"""Ephemeral, signed PCM/WAV streams for assistant output."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import secrets
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


class AudioAccessError(ValueError):
    """Raised when an audio stream URL is invalid, expired, or revoked."""


_END = object()


@dataclass(slots=True)
class AudioStream:
    """One bounded, non-persistent PCM output stream."""

    stream_id: str
    sample_rate: int
    channels: int
    _queue: asyncio.Queue[bytes | object]
    _closed: bool = False
    _revoked: bool = False

    async def write(self, pcm: bytes) -> None:
        if self._closed or self._revoked:
            raise AudioAccessError("audio stream is closed")
        if not pcm or len(pcm) % 2:
            raise ValueError("audio chunks must be non-empty PCM16")
        await self._queue.put(pcm)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Never block session cancellation on an output consumer. If the queue
        # is full, wav_chunks exits after draining its bounded remaining data.
        try:
            self._queue.put_nowait(_END)
        except asyncio.QueueFull:
            pass

    async def revoke(self) -> None:
        self._revoked = True
        await self.close()

    async def wav_chunks(self) -> AsyncIterator[bytes]:
        if self._revoked:
            raise AudioAccessError("audio stream is revoked")
        yield _streaming_wav_header(self.sample_rate, self.channels)
        while True:
            if self._closed and self._queue.empty():
                return
            chunk = await self._queue.get()
            if chunk is _END:
                return
            if self._revoked:
                raise AudioAccessError("audio stream is revoked")
            assert isinstance(chunk, bytes)
            yield chunk


@dataclass(slots=True)
class AudioStreamStore:
    """Own ephemeral output streams and their short-lived signed access tokens."""

    signing_key: bytes
    token_ttl_seconds: int = 60
    max_buffered_chunks: int = 64
    _streams: dict[str, AudioStream] = field(default_factory=dict)

    def create(self, sample_rate: int, channels: int = 1) -> tuple[AudioStream, str]:
        if sample_rate < 8_000 or channels < 1:
            raise ValueError("unsupported output audio format")
        stream_id = secrets.token_urlsafe(18)
        stream = AudioStream(
            stream_id,
            sample_rate,
            channels,
            asyncio.Queue(maxsize=self.max_buffered_chunks),
        )
        self._streams[stream_id] = stream
        expiry = int(time.time()) + self.token_ttl_seconds
        return stream, self._sign(stream_id, expiry)

    def open(self, stream_id: str, token: str) -> AudioStream:
        expiry, signature = _split_token(token)
        if expiry < time.time():
            raise AudioAccessError("audio token has expired")
        expected = self._signature(stream_id, expiry)
        if not hmac.compare_digest(signature, expected):
            raise AudioAccessError("audio token is invalid")
        stream = self._streams.get(stream_id)
        if stream is None or stream._revoked:
            raise AudioAccessError("audio stream is unavailable")
        return stream

    async def revoke(self, stream_id: str) -> None:
        stream = self._streams.pop(stream_id, None)
        if stream is not None:
            await stream.revoke()

    def _sign(self, stream_id: str, expiry: int) -> str:
        return f"{expiry}.{self._signature(stream_id, expiry)}"

    def _signature(self, stream_id: str, expiry: int) -> str:
        value = f"{stream_id}:{expiry}".encode()
        digest = hmac.new(self.signing_key, value, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def _split_token(token: str) -> tuple[int, str]:
    expiry_text, separator, signature = token.partition(".")
    if not separator or not signature:
        raise AudioAccessError("audio token is malformed")
    try:
        return int(expiry_text), signature
    except ValueError as err:
        raise AudioAccessError("audio token is malformed") from err


def _streaming_wav_header(sample_rate: int, channels: int) -> bytes:
    """Return a WAV header with an unknown data length for streaming playback."""
    byte_rate = sample_rate * channels * 2
    block_align = channels * 2
    unknown_length = 0xFFFFFFFF
    return b"".join(
        (
            b"RIFF",
            unknown_length.to_bytes(4, "little"),
            b"WAVEfmt ",
            (16).to_bytes(4, "little"),
            (1).to_bytes(2, "little"),
            channels.to_bytes(2, "little"),
            sample_rate.to_bytes(4, "little"),
            byte_rate.to_bytes(4, "little"),
            block_align.to_bytes(2, "little"),
            (16).to_bytes(2, "little"),
            b"data",
            unknown_length.to_bytes(4, "little"),
        )
    )
