"""Bounded, redacted tool lifecycle payloads for untrusted transport clients."""

from __future__ import annotations

import json
from typing import Any

_MAX_ARGUMENT_BYTES = 4_096
_MAX_RESULT_BYTES = 8_192
_SENSITIVE_KEYS = frozenset(
    {"authorization", "cookie", "password", "secret", "token", "api_key", "apikey"}
)


def public_arguments(value: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return a JSON-safe, bounded copy suitable for lifecycle events."""
    safe = _redact(value)
    if _encoded_size(safe) <= _MAX_ARGUMENT_BYTES:
        return safe, False
    return {"_truncated": True}, True


def public_result(value: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Return a JSON-safe, bounded result while retaining the MCP list shape."""
    safe = _redact(value)
    if _encoded_size(safe) <= _MAX_RESULT_BYTES:
        return safe, False
    return ([{"type": "text", "text": "[tool result truncated]"}], True)


def _redact(value: Any, key: str | None = None) -> Any:
    if key is not None and _is_sensitive(key):
        return "[redacted]"
    if isinstance(value, dict):
        return {
            str(item_key): _redact(item_value, str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        # Signed audio URLs must never be forwarded through tool diagnostics.
        if value.startswith(("http://", "https://")):
            return value.split("?", 1)[0]
        return value
    if value is None or isinstance(value, bool | int | float):
        return value
    return str(value)


def _is_sensitive(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith(
        ("_token", "_secret", "_password", "_key")
    )


def _encoded_size(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode())
