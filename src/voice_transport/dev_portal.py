"""Small authenticated, server-rendered audit viewer for development deployments."""

from __future__ import annotations

import asyncio
import html
import io
import json
import secrets
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, Response

_DEV_COOKIE = "voice_transport_dev"


def require_bearer(request: Request, expected_token: str) -> None:
    """Require the transport bearer without accepting URL/query-string secrets."""
    scheme, _, header_token = request.headers.get("authorization", "").partition(" ")
    token = (
        header_token
        if scheme.lower() == "bearer"
        else request.cookies.get(_DEV_COOKIE, "")
    )
    if not token or not secrets.compare_digest(token, expected_token):
        raise HTTPException(
            status_code=401,
            detail="Bearer authentication is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def set_dev_cookie(response: Response, request: Request, expected_token: str) -> None:
    """Persist a header-authenticated developer session for HTML audio requests."""
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and secrets.compare_digest(token, expected_token):
        response.set_cookie(
            _DEV_COOKIE,
            token,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            path="/dev",
        )


async def load_events(directory: Path) -> list[dict[str, Any]]:
    """Load bounded, valid audit JSONL events ordered chronologically."""
    return await asyncio.to_thread(_load_events, directory)


def _load_events(directory: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not directory.is_dir():
        return events
    for path in sorted(directory.glob("sessions-????-??-??.jsonl")):
        try:
            with path.open(encoding="utf-8") as file:
                for line in file:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict) and isinstance(
                        item.get("session_id"), str
                    ):
                        events.append(item)
        except OSError:
            continue
    return sorted(events, key=lambda item: str(item.get("timestamp", "")))


def sessions_by_recency(
    events: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[event["session_id"]].append(event)
    return sorted(
        grouped.items(),
        key=lambda item: str(item[1][-1].get("timestamp", "")),
        reverse=True,
    )


def index_page(
    config: str, sessions: list[tuple[str, list[dict[str, Any]]]]
) -> HTMLResponse:
    def session_row(session_id: str, events: list[dict[str, Any]]) -> str:
        escaped_id = html.escape(session_id, quote=True)
        session_link = (
            f"<a href='/dev/sessions/{escaped_id}'>"
            f"{html.escape(session_id)}</a>"
        )
        return (
            "<tr>"
            f"<td>{session_link}</td>"
            f"<td>{html.escape(str(events[0].get('timestamp', '')))}</td>"
            f"<td>{html.escape(str(events[-1].get('timestamp', '')))}</td>"
            f"<td>{len(events)}</td>"
            "</tr>"
        )

    rows = (
        "".join(session_row(session_id, events) for session_id, events in sessions)
        or "<tr><td colspan='4'>No audit sessions found.</td></tr>"
    )
    body = (
        "<h1>Voice Transport developer portal</h1>"
        "<p>Bearer authentication is required. "
        "This page contains development audit data.</p>"
        "<h2>Trusted tool configuration</h2>"
        f"<pre>{html.escape(config)}</pre>"
        "<h2>Audit sessions (newest first)</h2>"
        "<table><tr><th>Session</th><th>Started</th>"
        "<th>Last event</th><th>Events</th></tr>"
        f"{rows}</table>"
    )
    return HTMLResponse(_document(body))


def detail_page(session_id: str, events: list[dict[str, Any]]) -> HTMLResponse:
    rows: list[str] = []
    for index, event in enumerate(events):
        detail = {
            key: value
            for key, value in event.items()
            if key not in {"timestamp", "session_id", "event"}
        }
        audio = ""
        if event.get("event") == "debug.audio_captured":
            if isinstance(event.get("offset_bytes"), int):
                escaped_id = html.escape(session_id, quote=True)
                audio = (
                    "<audio controls preload='none' "
                    f"src='/dev/sessions/{escaped_id}/audio/{index}'></audio>"
                )
            else:
                audio = "<em>Legacy audio event: per-event offset unavailable.</em>"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(event.get('timestamp', '')))}</td>"
            f"<td>{html.escape(str(event.get('event', '')))}</td>"
            f"<td><pre>{html.escape(json.dumps(detail, indent=2, sort_keys=True))}"
            "</pre></td>"
            f"<td>{audio}</td></tr>"
        )
    body = (
        "<p><a href='/dev'>&larr; sessions</a></p>"
        f"<h1>Session {html.escape(session_id)}</h1>"
        "<table><tr><th>Timestamp</th><th>Event</th><th>Data</th><th>Audio</th></tr>"
        f"{''.join(rows) or '<tr><td colspan=4>No events.</td></tr>'}</table>"
    )
    return HTMLResponse(_document(body))


def wav_clip(directory: Path, event: dict[str, Any]) -> Response:
    """Return one bounded debug PCM audit event as a browser-playable WAV."""
    filename = event.get("audio_file")
    offset = event.get("offset_bytes")
    byte_count = event.get("bytes")
    sample_rate = event.get("sample_rate")
    channels = event.get("channels")
    if not (
        isinstance(filename, str)
        and Path(filename).name == filename
        and isinstance(offset, int)
        and isinstance(byte_count, int)
        and isinstance(sample_rate, int)
        and isinstance(channels, int)
        and offset >= 0
        and byte_count > 0
        and sample_rate > 0
        and channels > 0
    ):
        raise HTTPException(status_code=404, detail="Audio clip is unavailable.")
    try:
        with (directory / filename).open("rb") as file:
            file.seek(offset)
            pcm = file.read(byte_count)
    except OSError as err:
        raise HTTPException(
            status_code=404, detail="Audio sidecar is unavailable."
        ) from err
    if len(pcm) != byte_count:
        raise HTTPException(status_code=404, detail="Audio clip is incomplete.")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return Response(
        output.getvalue(), media_type="audio/wav", headers={"Cache-Control": "no-store"}
    )


def _document(body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Voice Transport dev</title>"
        "<style>body{font:14px sans-serif;margin:2rem}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #bbb;padding:.4rem;text-align:left;vertical-align:top}"
        "pre{white-space:pre-wrap;max-width:80rem;margin:0}</style></head><body>"
        f"{body}</body></html>"
    )
