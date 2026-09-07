"""Small authenticated, server-rendered audit viewer for development deployments."""

from __future__ import annotations

import asyncio
import base64
import binascii
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


def require_basic(request: Request, username: str, password: str) -> None:
    """Require separate HTTP Basic credentials for the developer portal."""
    scheme, _, encoded = request.headers.get("authorization", "").partition(" ")
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
        supplied_username, supplied_password = decoded.split(":", 1)
    except (ValueError, UnicodeDecodeError, binascii.Error):
        supplied_username = supplied_password = ""
    if not (
        scheme.lower() == "basic"
        and secrets.compare_digest(supplied_username, username)
        and secrets.compare_digest(supplied_password, password)
    ):
        raise HTTPException(
            status_code=401,
            detail="Developer portal authentication is required.",
            headers={
                "WWW-Authenticate": 'Basic realm="Voice Transport developer portal"'
            },
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
            f"<a href='/dev/sessions/{escaped_id}'>{html.escape(session_id)}</a>"
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
        "<p>HTTP Basic authentication is required. "
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
    event_types = sorted(
        {str(event.get("event", "")) for event in events if event.get("event")}
    )
    options = "".join(
        f"<option value='{html.escape(event_type, quote=True)}' selected>"
        f"{html.escape(event_type)}</option>"
        for event_type in event_types
    )
    rows: list[str] = []
    for index, event in enumerate(events):
        detail = {
            key: value
            for key, value in event.items()
            if key not in {"timestamp", "session_id", "event"}
        }
        audio = ""
        if event.get("event") == "debug.audio_captured":
            prior = events[index - 1] if index else None
            is_new_run = not isinstance(prior, dict) or (
                prior.get("event") != "debug.audio_captured"
                or prior.get("audio_file") != event.get("audio_file")
            )
            if is_new_run and isinstance(event.get("offset_bytes"), int):
                escaped_id = html.escape(session_id, quote=True)
                audio = (
                    "<audio controls preload='none' "
                    f"src='/dev/sessions/{escaped_id}/audio/{index}'></audio>"
                )
            elif is_new_run:
                audio = "<em>Legacy audio event: offset unavailable.</em>"
        rows.append(
            f"<tr data-event='{html.escape(str(event.get('event', '')), quote=True)}'>"
            f"<td>{html.escape(str(event.get('timestamp', '')))}</td>"
            f"<td>{html.escape(str(event.get('event', '')))}</td>"
            f"<td><pre>{html.escape(json.dumps(detail, indent=2, sort_keys=True))}"
            "</pre></td>"
            f"<td>{audio}</td></tr>"
        )
    body = (
        "<p><a href='/dev'>&larr; sessions</a></p>"
        f"<h1>Session {html.escape(session_id)}</h1>"
        "<section class='filters' aria-label='Timeline filters'>"
        "<label>Search <input id='timeline-search' type='search' "
        "placeholder='Search event data'></label>"
        "<label>Event types <select id='event-types' multiple size='8'>"
        f"{options}</select></label>"
        "<button id='clear-filters' type='button'>Clear filters</button>"
        "<output id='filter-count'></output></section>"
        "<table><thead><tr><th>Timestamp</th><th>Event</th><th>Data</th>"
        "<th>Audio</th></tr></thead><tbody id='timeline-events'>"
        f"{''.join(rows) or '<tr><td colspan=4>No events.</td></tr>'}"
        "</tbody></table><script>"
        "(()=>{const q=document.querySelector('#timeline-search'),"
        "types=document.querySelector('#event-types'),"
        "rows=[...document.querySelectorAll('#timeline-events tr[data-event]')],"
        "count=document.querySelector('#filter-count');"
        "const apply=()=>{const selected=new Set([...types.selectedOptions]"
        ".map(option=>option.value)),query=q.value.trim().toLowerCase();"
        "let visible=0;for(const row of rows){const matchType="
        "selected.has(row.dataset.event),matchText=!query||"
        "row.textContent.toLowerCase().includes(query);"
        "row.hidden=!(matchType&&matchText);if(!row.hidden)visible++;}"
        "count.textContent=`${visible} of ${rows.length} events shown`;};"
        "q.addEventListener('input',apply);types.addEventListener('change',apply);"
        "document.querySelector('#clear-filters').addEventListener('click',()=>{"
        "q.value='';for(const option of types.options)option.selected=true;"
        "apply();});apply();})();"
        "</script>"
    )
    return HTMLResponse(_document(body))


def wav_clip(
    directory: Path, events: list[dict[str, Any]], event_index: int
) -> Response:
    """Return one contiguous debug-audio run as a browser-playable WAV."""
    event = events[event_index]
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
    end_offset = offset + byte_count
    for candidate in events[event_index + 1 :]:
        if (
            candidate.get("event") != "debug.audio_captured"
            or candidate.get("audio_file") != filename
        ):
            break
        candidate_offset = candidate.get("offset_bytes")
        candidate_bytes = candidate.get("bytes")
        if isinstance(candidate_offset, int) and isinstance(candidate_bytes, int):
            end_offset = max(end_offset, candidate_offset + candidate_bytes)
    try:
        with (directory / filename).open("rb") as file:
            file.seek(offset)
            pcm = file.read(end_offset - offset)
    except OSError as err:
        raise HTTPException(
            status_code=404, detail="Audio sidecar is unavailable."
        ) from err
    if len(pcm) < byte_count:
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
        "pre{white-space:pre-wrap;max-width:80rem;margin:0}"
        ".filters{display:flex;gap:1rem;align-items:start;margin:1rem 0}"
        ".filters label{display:grid;gap:.3rem}.filters select{min-width:16rem}"
        "tr[hidden]{display:none}</style></head><body>"
        f"{body}</body></html>"
    )
