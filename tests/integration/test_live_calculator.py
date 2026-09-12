"""Opt-in E2E regression for local calculator results reaching OpenAI Realtime."""

import asyncio
import json
import os
import secrets

import pytest
from websockets.asyncio.client import connect

from tests.integration.test_live_openai_mcp import running_server, session_start
from voice_transport.config import Settings

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
