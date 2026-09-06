"""Opt-in, redacted, daily JSONL audit logs for external conversations."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

SessionAuditMode = Literal["off", "metadata", "debug_content"]
_SENSITIVE_KEY = re.compile(
    r"token|secret|password|authorization|api[_-]?key|cookie", re.I
)
_QUERY_STRING = re.compile(r"\?[^\s]*")


class SessionAuditLog:
    """Append redacted session events to daily files owned by deployment policy."""

    def __init__(
        self, directory: Path, *, mode: SessionAuditMode, retention_days: int
    ) -> None:
        self._directory = directory
        self._mode = mode
        self._retention_days = retention_days
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._mode == "off":
            return
        self._directory.mkdir(parents=True, exist_ok=True)
        await self.prune()

    async def record(self, session_id: str, event: str, **fields: Any) -> None:
        if self._mode == "off":
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
            if self._mode == "metadata" and key in {
                "transcript",
                "arguments",
                "result",
            }:
                continue
            entry[key] = _redact(value, key)
        encoded = json.dumps(entry, separators=(",", ":"), sort_keys=True)
        path = self._directory / f"sessions-{now.date().isoformat()}.jsonl"
        async with self._lock:
            await asyncio.to_thread(_append, path, encoded)

    async def prune(self, *, now: datetime | None = None) -> None:
        if self._mode == "off":
            return
        now = now or datetime.now(UTC)
        cutoff = now.date() - timedelta(days=self._retention_days)
        async with self._lock:
            await asyncio.to_thread(_prune, self._directory, cutoff.isoformat())


def _append(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(line)
        file.write("\n")


def _prune(directory: Path, cutoff_date: str) -> None:
    if not directory.exists():
        return
    for path in directory.glob("sessions-????-??-??.jsonl"):
        date = path.stem.removeprefix("sessions-")
        if date < cutoff_date:
            path.unlink(missing_ok=True)


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
