"""Opt-in live OpenAI Realtime + Home Assistant MCP E2E validation.

Run locally only with explicitly injected test credentials:

  RUN_LIVE_OPENAI_MCP_TEST=1 OPENAI_API_KEY=... HOMEASSISTANT_MCP_TOKEN=... \
  .venv/bin/pytest -q tests/integration/test_live_openai_mcp.py

The sole allowed MCP tool is the read-only Home Assistant context lookup. This
must never run in CI or with a production control-tool allowlist.
"""

import asyncio
import json
import os
import secrets
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
import uvicorn
from websockets.asyncio.client import connect

from voice_transport.app import create_app
from voice_transport.config import Settings

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_OPENAI_MCP_TEST") != "1",
    reason="set RUN_LIVE_OPENAI_MCP_TEST=1 to run live billable OpenAI/MCP validation",
)

_TIMEOUT_SECONDS = 90


@asynccontextmanager
async def running_server(settings: Settings) -> AsyncIterator[str]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(settings),
            host="127.0.0.1",
            port=port,
            access_log=False,
            lifespan="off",
            log_level="warning",
        )
    )
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(0.01)
        else:
            raise TimeoutError("live E2E server did not start")
        yield f"ws://127.0.0.1:{port}/transport/v1"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 5)


def session_start() -> dict[str, object]:
    return {
        "type": "session.start",
        "protocol_version": 1,
        "session_id": "live-mcp-e2e",
        "satellite": {"entity_id": "assist_satellite.live_test", "name": "Live Test"},
        "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1},
        "conversation": {"id": None, "wake_word": None},
    }


@pytest.mark.asyncio
async def test_live_realtime_model_calls_read_only_home_assistant_mcp_tool(
    tmp_path,
) -> None:
    api_key = os.environ["OPENAI_API_KEY"]
    ha_token = os.environ["HOMEASSISTANT_MCP_TOKEN"]
    tools = tmp_path / "voice-tools.json"
    tools.write_text(
        json.dumps(
            {
                "mcp_servers": [
                    {
                        "name": "home-assistant",
                        "transport": "streamable_http",
                        "url": "https://homeassistant.lan.brennandouglas.com/api/mcp",
                        "bearer_token_env": "HOMEASSISTANT_MCP_TOKEN",
                        "allowed_tools": ["homeassistant__GetLiveContext"],
                        "request_timeout_seconds": 15,
                        "max_concurrent_calls": 1,
                    }
                ]
            }
        )
    )
    original_token = os.environ.get("HOMEASSISTANT_MCP_TOKEN")
    os.environ["HOMEASSISTANT_MCP_TOKEN"] = ha_token
    settings = Settings(
        transport_token=secrets.token_urlsafe(24),
        realtime_provider="openai_realtime",
        openai_api_key=api_key,
        openai_realtime_model=os.environ.get(
            "LIVE_OPENAI_REALTIME_MODEL", "gpt-realtime-2.1-mini"
        ),
        openai_realtime_voice="ballad",
        public_base_url="http://127.0.0.1:8080",
        audio_url_signing_key=secrets.token_urlsafe(32),
        trusted_tool_config_path=str(tools),
        session_audit_mode="debug_content",
        session_audit_log_path=str(tmp_path / "audit"),
        session_audit_retention_days=7,
    )
    try:
        async with running_server(settings) as ws_url:
            async with connect(
                ws_url,
                additional_headers={
                    "authorization": f"Bearer {settings.transport_token}"
                },
                open_timeout=10,
                close_timeout=10,
            ) as websocket:
                await websocket.send(json.dumps(session_start()))
                assert (
                    json.loads(await asyncio.wait_for(websocket.recv(), 15))["type"]
                    == "session.ready"
                )
                await websocket.send(
                    json.dumps(
                        {
                            "type": "turn.start",
                            "turn_id": "tool-turn",
                            "input": "text",
                        }
                    )
                )
                await websocket.send(
                    json.dumps(
                        {
                            "type": "input.text",
                            "turn_id": "tool-turn",
                            "text": (
                                "Use homeassistant__GetLiveContext now. Do not guess; "
                                "briefly summarize the current home context after "
                                "the tool returns."
                            ),
                        }
                    )
                )
                await websocket.send(
                    json.dumps({"type": "turn.end", "turn_id": "tool-turn"})
                )
                for _ in range(100):
                    event = json.loads(
                        await asyncio.wait_for(websocket.recv(), _TIMEOUT_SECONDS)
                    )
                    if event["type"] == "assistant.response_finished":
                        break
                else:
                    raise AssertionError("live model did not finish its tool response")
                await websocket.send(json.dumps({"type": "session.cancel"}))
                assert (
                    json.loads(await asyncio.wait_for(websocket.recv(), 10))["type"]
                    == "session.finished"
                )
    finally:
        if original_token is None:
            os.environ.pop("HOMEASSISTANT_MCP_TOKEN", None)
        else:
            os.environ["HOMEASSISTANT_MCP_TOKEN"] = original_token

    audit_entries = [
        json.loads(line)
        for line in next((tmp_path / "audit").glob("sessions-*.jsonl"))
        .read_text()
        .splitlines()
    ]
    assert any(entry["event"] == "tool.call_started" for entry in audit_entries)
    completed = next(
        entry for entry in audit_entries if entry["event"] == "tool.call_finished"
    )
    assert completed["tool_name"] == "homeassistant__GetLiveContext"
    assert completed["is_error"] is False
