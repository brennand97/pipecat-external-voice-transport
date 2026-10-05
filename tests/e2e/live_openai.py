"""Billable, real OpenAI/Pipecat acceptance against ONLY a scratch HA instance.

Credentials are read in-process from the inventoried test key and private run.
Never uses the production HA MCP endpoint or production credentials.
"""

import argparse
import asyncio
import json
import os
import re
import secrets
import socket
from pathlib import Path

import uvicorn
from ha_testbed.api import HA
from websockets.asyncio.client import connect

from voice_transport.app import create_app
from voice_transport.config import Settings
from voice_transport.tools.mcp import MCPServerConfig, MCPToolProvider


async def main(run_path, key_file):
    run = json.loads(run_path.read_text())
    if (
        run["status"] != "ready"
        or not run["id"].startswith("hatb-")
        or not run["url"].startswith("http://localhost:")
    ):
        raise ValueError("Refusing non-scratch HA")
    credentials = json.loads((run_path.parent / "credentials.json").read_text())
    ha = HA(run["url"], credentials["tokens"]["access_token"])
    entity = run["fixture"]["entity"]
    device = ha.ws({"type": "config/entity_registry/get", "entity_id": entity})[
        "device_id"
    ]
    tools = json.loads((run_path.parent / "tools/tools.json").read_text())
    for server in tools["mcp_servers"]:
        server["url"] = run["url"] + "/api/mcp"
    config = run_path.parent / "live-tools.json"
    config.write_text(json.dumps(tools))
    config.chmod(0o600)
    os.environ["HOMEASSISTANT_MCP_TOKEN"] = credentials["tokens"]["access_token"]
    # Key never appears in argv, tools.json, reports, or shell output.
    key = key_file.read_text().strip()
    settings = Settings(
        transport_token=secrets.token_urlsafe(32),
        realtime_provider="openai_realtime",
        openai_api_key=key,
        openai_realtime_model="gpt-realtime-mini",
        openai_realtime_voice="marin",
        trusted_tool_config_path=str(config),
        public_base_url="http://127.0.0.1:8080",
        audio_url_signing_key=secrets.token_urlsafe(32),
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ready = asyncio.Event()

    class Server(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets)
            ready.set()

    server = Server(
        uvicorn.Config(
            create_app(settings),
            host="127.0.0.1",
            port=port,
            access_log=False,
            log_level="error",
            lifespan="off",
        )
    )
    task = asyncio.create_task(server.serve())
    truth = MCPToolProvider(
        MCPServerConfig(
            name="scratch-timer-truth",
            transport="streamable_http",
            url=run["url"] + "/api/mcp",
            bearer_token=credentials["tokens"]["access_token"],
            allowed_tools=frozenset({"voice_satellite__GetTimerStatus"}),
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), 15)
        async with connect(
            f"ws://127.0.0.1:{port}/transport/v1",
            additional_headers={"Authorization": "Bearer " + settings.transport_token},
        ) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "session.start",
                        "protocol_version": 1,
                        "session_id": "real-openai-" + secrets.token_hex(8),
                        "satellite": {"entity_id": entity, "name": "Scratch kitchen"},
                        "audio": {
                            "encoding": "pcm_s16le",
                            "sample_rate": 16000,
                            "channels": 1,
                        },
                        "conversation": {
                            "id": None,
                            "wake_word": None,
                            "profile": "e2e",
                            "device_id": device,
                            "input_modalities": ["text"],
                            "output_modalities": ["text"],
                        },
                    }
                )
            )
            event = json.loads(await asyncio.wait_for(ws.recv(), 30))
            assert event["type"] == "session.ready", event.get("code", event["type"])
            assert "voice_satellite__GetTimerStatus" in event.get(
                "effective_tools",
                event.get("capabilities", {}).get("effective_tools", []),
            ), "Timer tools missing"

            async def turn(index, text, required=(), forbidden=(), terminal=False):
                turn_id = f"live-{index}"
                for frame in [
                    {"type": "turn.start", "turn_id": turn_id, "input": "text"},
                    {"type": "input.text", "turn_id": turn_id, "text": text},
                    {"type": "turn.end", "turn_id": turn_id},
                ]:
                    await ws.send(json.dumps(frame))
                calls = []
                speech = []
                results = {}
                async with asyncio.timeout(60):
                    while True:
                        e = json.loads(await ws.recv())
                        typ = e["type"]
                        if typ in ("error", "session.error"):
                            raise AssertionError(
                                "Provider session error: " + e.get("code", "unknown")
                            )
                        if typ == "assistant.tool_call_finished":
                            name = e.get("tool_name") or e.get("tool", {}).get("name")
                            calls.append(name)
                            content = e.get("result", [])
                            if content and name.startswith("voice_satellite__"):
                                results[name] = json.loads(
                                    next(
                                        item["text"]
                                        for item in content
                                        if item["type"] == "text"
                                    )
                                )
                            assert not e.get("is_error", False), "Tool failed: " + str(
                                name
                            )
                        if typ == "assistant.text.final":
                            speech.append(e.get("text", ""))
                        if terminal and typ == "session.finished":
                            break
                        if (
                            not terminal
                            and typ == "assistant.response_finished"
                            and all(name in calls for name in required)
                        ):
                            break
                assert all(name in calls for name in required), (
                    "Required timer tool was not called"
                )
                assert not any(name in calls for name in forbidden), (
                    "Unsafe tool choice: " + str(calls)
                )
                if index == 5:
                    assert re.search(
                        r"\b(which|clarify|specify)\b", " ".join(speech), re.I
                    ), "Ambiguous request must elicit clarification"
                print(
                    json.dumps(
                        {
                            "case": index,
                            "tools": calls,
                            "answer": speech,
                            "passed": True,
                        }
                    ),
                    flush=True,
                )
                return results

            async def names():
                result = await truth.call_tool(
                    "voice_satellite__GetTimerStatus", {"device_id": device}
                )
                assert not result.is_error, "Independent timer-state lookup failed"
                data = json.loads(
                    next(
                        item["text"]
                        for item in result.content
                        if item["type"] == "text"
                    )
                )
                return sorted(t["name"] for t in data["timers"])

            assert not await names(), (
                "Use a fresh scratch device without existing timers"
            )
            await turn(
                1,
                "Set a five-minute timer named Pasta.",
                ("voice_satellite__StartTimer",),
            )
            assert [n.lower() for n in await names()] == ["pasta"]
            await turn(
                2,
                "How much time is left on the pasta timer?",
                ("voice_satellite__GetTimerStatus",),
            )
            results = await turn(
                3,
                "Add ten seconds to the pasta timer.",
                ("voice_satellite__ExtendTimer",),
            )
            changed = results["voice_satellite__ExtendTimer"]
            assert changed["relative_seconds"] == 10
            assert (
                9
                <= changed["timer"]["seconds_remaining"]
                - changed["previous"]["seconds_remaining"]
                <= 10
            )
            await turn(
                4,
                "Set a second five-minute timer named Tea.",
                ("voice_satellite__StartTimer",),
            )
            assert len(await names()) == 2
            # Fresh listing or conservative clarification are safe. A skipped
            # lookup never grants permission to guess and mutate a named timer.
            await turn(
                5,
                "Stop the timer.",
                (),
                (
                    "voice_satellite__StopTimer",
                    "voice_satellite__ExtendTimer",
                    "voice_satellite__ShortenTimer",
                    "voice_satellite__RenameTimer",
                    "transport__EndSession",
                ),
            )
            assert len(await names()) == 2
            await turn(
                6,
                "Stop the pasta timer, not our conversation.",
                ("voice_satellite__StopTimer",),
                ("transport__EndSession",),
            )
            assert [n.lower() for n in await names()] == ["tea"]
            await turn(
                7,
                "Please end our conversation, but leave the tea timer running.",
                (),
                ("voice_satellite__StopTimer",),
                terminal=True,
            )
            assert len(await names()) == 1
            print(
                "PASS real OpenAI/Pipecat: fresh timer state, relative extension, "
                "ambiguous-stop safety, named cancellation, agent-ended session "
                "preserves unrelated timer",
                flush=True,
            )
    finally:
        await truth.close()
        server.should_exit = True
        await asyncio.wait_for(task, 10)
        os.environ.pop("HOMEASSISTANT_MCP_TOKEN", None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path.home() / ".local/state/agents/.openai-test-key",
    )
    args = parser.parse_args()
    try:
        asyncio.run(main(args.run, args.api_key_file))
    except Exception as error:
        # No settings/credential repr in exception diagnostics.
        print(
            "FAIL live OpenAI:",
            type(error).__name__,
            str(error)
            if isinstance(error, AssertionError)
            else "See private provider diagnostics",
            flush=True,
        )
        raise SystemExit(1) from None
