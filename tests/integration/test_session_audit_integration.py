import json

import pytest
from test_transport import _TIMEOUT_SECONDS, running_server, start
from websockets.asyncio.client import connect

from voice_transport.config import Settings


@pytest.mark.asyncio
async def test_debug_content_audit_records_transcript_and_session_lifecycle(
    tmp_path,
) -> None:
    settings = Settings(
        "token",
        session_audit_mode="debug_content",
        session_audit_log_path=str(tmp_path),
        session_audit_retention_days=7,
    )
    async with running_server(settings) as (base_url, _app):
        ws_url = base_url.replace("http", "ws", 1) + "/transport/v1"
        async with connect(
            ws_url,
            additional_headers={"authorization": "Bearer token"},
            open_timeout=_TIMEOUT_SECONDS,
            close_timeout=_TIMEOUT_SECONDS,
        ) as websocket:
            await websocket.send(json.dumps(start()))
            assert json.loads(await websocket.recv())["type"] == "session.ready"
            await websocket.send(
                '{"type":"turn.start","turn_id":"text-1","input":"text"}'
            )
            await websocket.send(
                '{"type":"input.text","turn_id":"text-1","text":"hello audit"}'
            )
            await websocket.recv()  # user.transcript.final
            await websocket.send('{"type":"session.cancel"}')
            assert json.loads(await websocket.recv())["type"] == "session.finished"

    entries = [
        json.loads(line)
        for line in next(tmp_path.glob("sessions-*.jsonl")).read_text().splitlines()
    ]
    assert any(entry["event"] == "session.started" for entry in entries)
    transcript = next(
        entry for entry in entries if entry["event"] == "user.transcript.final"
    )
    assert transcript["transcript"] == "hello audit"
    assert any(entry["event"] == "session.finished" for entry in entries)
