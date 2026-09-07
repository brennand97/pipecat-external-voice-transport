"""Opt-in, redacted, daily JSONL audit logs for external conversations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

SessionAuditMode = Literal["off", "metadata", "debug_content"]
_LOGGER = logging.getLogger(__name__)
_SENSITIVE_KEY = re.compile(
    r"token|secret|password|authorization|api[_-]?key|cookie", re.I
)
_QUERY_STRING = re.compile(r"\?[^\s]*")


class SessionAuditLog:
    """Append redacted session events to daily files owned by deployment policy."""

    def __init__(
        self,
        directory: Path,
        *,
        mode: SessionAuditMode,
        retention_days: int,
        max_audio_bytes_per_session: int = 10_000_000,
    ) -> None:
        self._directory = directory
        self._mode = mode
        self._retention_days = retention_days
        self._max_audio_bytes_per_session = max_audio_bytes_per_session
        self._audio_bytes: dict[tuple[str, str], int] = {}
        self._lock = asyncio.Lock()
        self._available = mode != "off"

    async def initialize(self) -> None:
        if not self._available:
            return
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            await self.prune()
        except OSError as err:
            self._disable_after_io_failure(err)

    async def record(self, session_id: str, event: str, **fields: Any) -> None:
        if not self._available:
            return
        now = datetime.now(UTC)
        entry: dict[str, Any] = {
            "timestamp": now.isoformat(),
            "session_id": session_id,
            "event": event,
        }
        for key, value in fields.items():
            if value is None:
                continue
            # Audio is never audit content, including debug_content. Do not
            # create a future path that can retain PCM, WAV, or signed URLs.
            if key in {"audio", "pcm", "wav", "audio_url", "signed_audio_url"}:
                continue
            if self._mode == "metadata" and key in {
                "transcript",
                "arguments",
                "result",
            }:
                continue
            entry[key] = _redact(_json_safe(value), key)
        encoded = json.dumps(entry, separators=(",", ":"), sort_keys=True)
        path = self._directory / f"sessions-{now.date().isoformat()}.jsonl"
        try:
            async with self._lock:
                await asyncio.to_thread(_append, path, encoded)
        except OSError as err:
            self._disable_after_io_failure(err)

    async def record_debug(self, session_id: str, event: str, **fields: Any) -> None:
        """Record sensitive development diagnostics only in debug-content mode."""
        if self._mode == "debug_content":
            await self.record(session_id, event, **fields)

    async def record_audio(
        self,
        session_id: str,
        direction: Literal["input", "output"],
        pcm: bytes,
        *,
        sample_rate: int,
        channels: int,
        turn_id: str | None = None,
        response_id: str | None = None,
    ) -> None:
        """Append bounded raw PCM only for explicit debug-content auditing.

        Audio is deliberately outside JSONL and uses a non-reversible session
        filename. Operators need the documented PCM format to replay it.
        """
        if self._mode != "debug_content" or not self._available or not pcm:
            return
        key = (session_id, direction)
        async with self._lock:
            used = self._audio_bytes.get(key, 0)
            remaining = self._max_audio_bytes_per_session - used
            if remaining <= 0:
                return
            chunk = pcm[:remaining]
            now = datetime.now(UTC)
            digest = hashlib.sha256(session_id.encode()).hexdigest()[:24]
            filename = f"audio-{now.date().isoformat()}-{digest}-{direction}.pcm"
            path = self._directory / filename
            entry = {
                "timestamp": now.isoformat(),
                "session_id": session_id,
                "event": "debug.audio_captured",
                **({"turn_id": turn_id} if turn_id is not None else {}),
                **({"response_id": response_id} if response_id is not None else {}),
                "audio_file": filename,
                "direction": direction,
                "encoding": "pcm_s16le",
                "sample_rate": sample_rate,
                "channels": channels,
                "bytes": len(chunk),
                "offset_bytes": used,
                "cumulative_bytes": used + len(chunk),
                "truncated": len(chunk) != len(pcm),
            }
            audit_path = self._directory / f"sessions-{now.date().isoformat()}.jsonl"
            try:
                await asyncio.to_thread(_append_bytes, path, chunk)
                await asyncio.to_thread(
                    _append,
                    audit_path,
                    json.dumps(entry, separators=(",", ":"), sort_keys=True),
                )
            except OSError as err:
                self._disable_after_io_failure(err)
                return
            self._audio_bytes[key] = used + len(chunk)

    async def finish_session(self, session_id: str) -> None:
        """Release bounded in-memory audio accounting after terminal cleanup."""
        async with self._lock:
            for key in tuple(self._audio_bytes):
                if key[0] == session_id:
                    del self._audio_bytes[key]

    async def prune(self, *, now: datetime | None = None) -> None:
        if not self._available:
            return
        now = now or datetime.now(UTC)
        cutoff = now.date() - timedelta(days=self._retention_days)
        try:
            async with self._lock:
                await asyncio.to_thread(_prune, self._directory, cutoff.isoformat())
        except OSError as err:
            self._disable_after_io_failure(err)

    def _disable_after_io_failure(self, err: OSError) -> None:
        if not self._available:
            return
        self._available = False
        _LOGGER.warning("Audit logging disabled after I/O failure: %s", err)


def _append(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(line)
        file.write("\n")


def _prune(directory: Path, cutoff_date: str) -> None:
    if not directory.exists():
        return
    for pattern, prefix in (
        ("sessions-????-??-??.jsonl", "sessions-"),
        ("audio-????-??-??-*.pcm", "audio-"),
    ):
        for path in directory.glob(pattern):
            date = path.name.removeprefix(prefix)[:10]
            if date < cutoff_date:
                path.unlink(missing_ok=True)


def _append_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as file:
        file.write(payload)


def _json_safe(value: Any) -> Any:
    """Convert provider/Pydantic structures to JSON values before audit writes."""
    to_default_dict = getattr(value, "to_default_dict", None)
    if callable(to_default_dict):
        return _json_safe(to_default_dict())
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_safe(model_dump(mode="json"))
        except TypeError:
            return _json_safe(model_dump())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _redact(value: Any, key: str = "") -> Any:
    if _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(item): _redact(nested, str(item)) for item, nested in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str) and ("?token=" in value or "?signature=" in value):
        return _QUERY_STRING.sub("?[REDACTED]", value)
    return value
