"""Supervised real-provider scratch server over a private Unix socket.

A fixed Unix bridge keeps HA offline. Production app/provider/registry are
unchanged; instrumentation only observes frames and reads scratch MCP truth.
"""

import argparse
import asyncio
import json
import os
import secrets
import sys
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from fastapi import HTTPException, Request
from ha_testbed.api import HA
from loguru import logger

from voice_transport.app import create_app
from voice_transport.config import Settings
from voice_transport.tools.mcp import MCPServerConfig, MCPToolProvider


async def main(run_path, key_file):
    run = json.loads(run_path.read_text())
    private = run_path.parent
    url = urlsplit(run["url"])
    if (
        not run["id"].startswith("hatb-")
        or run["status"] != "ready"
        or url.hostname != "localhost"
        or url.username
    ):
        raise ValueError("Scratch loopback run required")
    creds = json.loads((private / "credentials.json").read_text())
    key = key_file.read_text().strip()

    def redact(record):
        for value in (
            key,
            creds["transport_token"],
            creds["tokens"]["access_token"],
            creds["signing_key"],
        ):
            record["message"] = record["message"].replace(value, "[REDACTED]")
        return True

    logger.remove()
    logger.add(sys.stderr, level="WARNING", filter=redact)
    tools = json.loads((private / "tools/tools.json").read_text())
    for server in tools["mcp_servers"]:
        server["url"] = run["url"] + "/api/mcp"
    tool_file = private / "live-tools.json"
    tool_file.write_text(json.dumps(tools))
    tool_file.chmod(0o600)
    os.environ["HOMEASSISTANT_MCP_TOKEN"] = creds["tokens"]["access_token"]
    ha = HA(run["url"], creds["tokens"]["access_token"])
    device = ha.ws(
        {"type": "config/entity_registry/get", "entity_id": run["fixture"]["entity"]}
    )["device_id"]
    socket_dir = private / "live-socket"
    socket_dir.mkdir(mode=0o700, exist_ok=True)
    settings = Settings(
        transport_token=creds["transport_token"],
        realtime_provider="openai_realtime",
        openai_api_key=key,
        openai_realtime_model="gpt-realtime-mini",
        openai_realtime_voice="marin",
        public_base_url=run["transport_url"],
        audio_url_signing_key=creds["signing_key"],
        trusted_tool_config_path=str(tool_file),
        session_audit_mode="metadata",
        session_audit_log_path=str(private / "artifacts/live-audit.jsonl"),
        input_idle_timeout_seconds=45,
    )
    app = create_app(settings)
    stats = {
        "created": 0,
        "input_bytes": 0,
        "closed": 0,
        "tool_calls": 0,
        "transcripts": [],
        "timer_starts": [],
        "answers": [],
        "audio_deliveries": [],
        "events": {},
        "session_contexts": [],
        "incoming_events": {},
        "termination": [],
        "provider_events": {},
        "provider_responses": [],
    }

    def authorize(request):
        if not secrets.compare_digest(
            request.headers.get("Authorization", ""),
            "Bearer " + creds["transport_token"],
        ):
            raise HTTPException(status_code=401)

    # Scratch-only observation of native provider lifecycle, without audio,
    # secrets, raw prompts or changing the production event handling.
    from voice_transport.providers.openai_realtime import realtime_events

    original_parser = realtime_events.parse_server_event

    def observe_provider_event(message):
        event = original_parser(message)
        stats["provider_events"][event.type] = (
            stats["provider_events"].get(event.type, 0) + 1
        )
        if event.type == "response.done":
            stats["provider_responses"].append({"status": event.response.status})
        return event

    realtime_events.parse_server_event = observe_provider_event

    @app.get("/e2e/stats")
    async def get_stats(request: Request):
        authorize(request)
        return stats

    @app.get("/e2e/timers")
    async def get_timers(request: Request):
        authorize(request)
        truth = MCPToolProvider(
            MCPServerConfig(
                name="scratch",
                transport="streamable_http",
                url=run["url"] + "/api/mcp",
                bearer_token=creds["tokens"]["access_token"],
                allowed_tools=frozenset({"voice_satellite__GetTimerStatus"}),
            )
        )
        try:
            result = await truth.call_tool(
                "voice_satellite__GetTimerStatus", {"device_id": device}
            )
            if result.is_error:
                raise HTTPException(status_code=502)
            return json.loads(
                next(item["text"] for item in result.content if item["type"] == "text")
            )
        finally:
            await truth.close()

    class Observed:
        async def __call__(self, scope, receive, send):
            async def read():
                message = await receive()
                if scope["type"] == "websocket":
                    stats["input_bytes"] += len(message.get("bytes") or b"")
                    if message.get("text"):
                        data = json.loads(message["text"])
                        typ = data.get("type")
                        stats["incoming_events"][typ] = (
                            stats["incoming_events"].get(typ, 0) + 1
                        )
                        if typ == "session.cancel":
                            stats["termination"].append(
                                {"direction": "client", "reason": data.get("reason")}
                            )
                        if data.get("type") == "session.start":
                            stats["created"] += 1
                            c = data.get("conversation", {})
                            stats["session_contexts"].append(
                                {
                                    k: c.get(k)
                                    for k in (
                                        "profile",
                                        "device_id",
                                        "input_modalities",
                                        "output_modalities",
                                    )
                                }
                            )
                return message

            async def write(message):
                if scope["type"] == "websocket" and message.get("text"):
                    data = json.loads(message["text"])
                    typ = data.get("type")
                    stats["events"][typ] = stats["events"].get(typ, 0) + 1
                    if typ == "assistant.audio":
                        stats["audio_deliveries"].append(
                            {
                                "response_id": data.get("response_id"),
                                "timer_starts_seen": len(stats["timer_starts"]),
                            }
                        )
                    if typ == "assistant.text.final":
                        stats["answers"].append(data.get("text", ""))
                    if typ == "session.finished":
                        stats["closed"] += 1
                        stats["termination"].append(
                            {
                                "direction": "server",
                                "reason": data.get("reason"),
                                "code": data.get("code"),
                            }
                        )
                    if typ == "user.transcript.final":
                        stats["transcripts"].append(data.get("text", ""))
                    if typ == "assistant.tool_call_finished":
                        stats["tool_calls"] += 1
                        if data.get("tool_name") == "voice_satellite__StartTimer":
                            content = data.get("result", [])
                            try:
                                stats["timer_starts"].append(
                                    json.loads(
                                        next(
                                            i["text"]
                                            for i in content
                                            if i["type"] == "text"
                                        )
                                    )
                                )
                            except (StopIteration, ValueError):
                                pass
                await send(message)

            await app(scope, read, write)

    ready = asyncio.Event()

    class Server(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets)
            ready.set()

    server = Server(
        uvicorn.Config(
            Observed(),
            uds=str(socket_dir / "live.sock"),
            access_log=False,
            log_level="error",
        )
    )
    task = asyncio.create_task(server.serve())
    await asyncio.wait_for(ready.wait(), 30)
    print(json.dumps({"endpoint": run["transport_url"]}), flush=True)
    await task


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument(
        "--api-key-file",
        type=Path,
        default=Path.home() / ".local/state/agents/.openai-test-key",
    )
    a = p.parse_args()
    asyncio.run(main(a.run, a.api_key_file))
