"""Opt-in E2E regression for local calculator results reaching OpenAI Realtime."""

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
        "session_id": "live-calculator-e2e",
        "satellite": {"entity_id": "assist_satellite.live_test", "name": "Live Test"},
        "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1},
        "conversation": {"id": None, "wake_word": None},
    }

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_OPENAI_MCP_TEST") != "1",
    reason="set RUN_LIVE_OPENAI_MCP_TEST=1 for billable OpenAI validation",
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_live_calculator_result_reaches_model(tmp_path) -> None:
    """The model must answer with the completed local calculation, not wait."""
    tools = tmp_path / "voice-tools.json"
    tools.write_text(
        json.dumps(
            {
                "mcp_servers": [],
                "profiles": {
                    "calculator-only": {
                        "providers": [],
                        "allowed_tools": ["no-provider-tools"],
                    }
                },
                "default_profile": "calculator-only",
            }
        )
    )
    settings = Settings(
        transport_token=secrets.token_urlsafe(24),
        realtime_provider="openai_realtime",
        openai_api_key=os.environ["OPENAI_API_KEY"],
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
    async with running_server(settings) as ws_url:
        async with connect(
            ws_url,
            additional_headers={"authorization": f"Bearer {settings.transport_token}"},
            open_timeout=10,
            close_timeout=10,
        ) as websocket:
            start = session_start()
            start["conversation"] = {
                **start["conversation"],
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            }
            await websocket.send(json.dumps(start))
            ready = json.loads(await asyncio.wait_for(websocket.recv(), 15))
            assert ready["type"] == "session.ready"
            await websocket.send(
                json.dumps({"type": "turn.start", "turn_id": "math", "input": "text"})
            )
            await websocket.send(
                json.dumps(
                    {
                        "type": "input.text",
                        "turn_id": "math",
                        "text": (
                            "What is five plus seven? Use the calculator, then "
                            "answer only with the result."
                        ),
                    }
                )
            )
            await websocket.send(
                json.dumps({"type": "turn.end", "turn_id": "math"})
            )
            finals: list[str] = []
            deadline = asyncio.get_running_loop().time() + 60
            while asyncio.get_running_loop().time() < deadline:
                remaining = deadline - asyncio.get_running_loop().time()
                event = json.loads(
                    await asyncio.wait_for(websocket.recv(), remaining)
                )
                if event["type"] == "assistant.text.final":
                    finals.append(event["text"])
                    if "12" in event["text"]:
                        break
            assert any("12" in text for text in finals), finals
