import asyncio
import json
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2 as httpx
import pytest
import uvicorn
from websockets.asyncio.client import connect

from voice_transport.app import create_app
from voice_transport.config import Settings

_TIMEOUT_SECONDS = 1


def start() -> dict:
    return {
        "type": "session.start",
        "protocol_version": 1,
        "session_id": "test-session",
        "satellite": {"entity_id": "assist_satellite.kitchen", "name": "Kitchen"},
        "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1},
        "conversation": {"id": None, "wake_word": None},
    }


@asynccontextmanager
async def running_server(settings: Settings) -> AsyncIterator[str]:
    """Run the ASGI app over a real bounded local TCP connection."""
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
            log_level="critical",
        )
    )
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(0.01)
        else:
            raise TimeoutError("test server did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, _TIMEOUT_SECONDS)


@pytest.mark.asyncio
async def test_health_and_ready() -> None:
    async with running_server(Settings("token")) as base_url:
        async with httpx.AsyncClient() as client:
            assert (await client.get(f"{base_url}/health")).json() == {"status": "ok"}
            assert (await client.get(f"{base_url}/ready")).json() == {"status": "ready"}


@pytest.mark.asyncio
async def test_transport_lifecycle() -> None:
    async with running_server(Settings("token")) as base_url:
        ws_url = base_url.replace("http", "ws", 1) + "/transport/v1"
        async with connect(
            ws_url,
            additional_headers={"authorization": "Bearer token"},
            open_timeout=_TIMEOUT_SECONDS,
            close_timeout=_TIMEOUT_SECONDS,
        ) as websocket:
            await asyncio.wait_for(
                websocket.send(json.dumps(start())), _TIMEOUT_SECONDS
            )
            assert (
                json.loads(await asyncio.wait_for(websocket.recv(), _TIMEOUT_SECONDS))[
                    "type"
                ]
                == "session.ready"
            )
            await asyncio.wait_for(websocket.send(b"\x00\x00"), _TIMEOUT_SECONDS)
            await asyncio.wait_for(
                websocket.send('{"type":"input.end"}'), _TIMEOUT_SECONDS
            )
            for expected in (
                "assistant.response_started",
                "assistant.text.final",
                "assistant.response_finished",
                "session.finished",
            ):
                assert (
                    json.loads(
                        await asyncio.wait_for(websocket.recv(), _TIMEOUT_SECONDS)
                    )["type"]
                    == expected
                )
