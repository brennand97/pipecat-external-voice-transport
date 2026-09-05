import asyncio
import json
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2 as httpx
import pytest
import uvicorn
from websockets.asyncio.client import connect

from voice_transport.agent.session import AgentEvent
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
async def running_server(settings: Settings) -> AsyncIterator[tuple[str, object]]:
    """Run the ASGI app over a real bounded local TCP connection."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    app = create_app(settings)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
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
        yield f"http://127.0.0.1:{port}", app
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, _TIMEOUT_SECONDS)


@pytest.mark.asyncio
async def test_health_and_ready() -> None:
    async with running_server(Settings("token")) as (base_url, _app):
        async with httpx.AsyncClient() as client:
            assert (await client.get(f"{base_url}/health")).json() == {"status": "ok"}
            assert (await client.get(f"{base_url}/ready")).json() == {"status": "ready"}


@pytest.mark.asyncio
async def test_transport_supports_persistent_audio_and_text_turns() -> None:
    async with running_server(Settings("token")) as (base_url, _app):
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
                '{"type":"turn.start","turn_id":"audio-1","input":"audio"}'
            )
            await websocket.send(b"\x00\x00")
            await websocket.send('{"type":"turn.end","turn_id":"audio-1"}')
            first = [json.loads(await websocket.recv()) for _ in range(3)]
            assert [event["type"] for event in first] == [
                "assistant.response_started",
                "assistant.text.final",
                "assistant.response_finished",
            ]
            assert {event["turn_id"] for event in first} == {"audio-1"}
            assert len({event["response_id"] for event in first}) == 1

            await websocket.send(
                '{"type":"turn.start","turn_id":"text-2","input":"text"}'
            )
            await websocket.send(
                '{"type":"input.text","turn_id":"text-2","text":"hello"}'
            )
            transcript = json.loads(await websocket.recv())
            assert transcript["source"] == "client_text"
            assert transcript["text"] == "hello"
            await websocket.send('{"type":"turn.end","turn_id":"text-2"}')
            second = [json.loads(await websocket.recv()) for _ in range(3)]
            assert {event["turn_id"] for event in second} == {"text-2"}
            await websocket.send('{"type":"session.cancel"}')
            assert json.loads(await websocket.recv())["type"] == "session.finished"


@pytest.mark.asyncio
async def test_response_cancel_is_nonterminal_and_revokes_active_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StreamingAgent:
        def __init__(self) -> None:
            self.events_queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
            self.cancelled = False
            self.interrupted = False

        async def start(self) -> None:
            pass

        async def submit_audio(self, turn_id: str, pcm: bytes) -> None:
            pass

        async def submit_text(self, turn_id: str, text: str) -> None:
            pass

        async def end_turn(self, turn_id: str) -> None:
            await self.events_queue.put(AgentEvent("assistant.response_started"))
            await self.events_queue.put(
                AgentEvent(
                    "assistant.audio.chunk",
                    audio=b"\x00\x00",
                    sample_rate=24_000,
                    channels=1,
                )
            )

        async def interrupt(self) -> None:
            self.interrupted = True

        async def cancel(self) -> None:
            await self.close()

        async def close(self) -> None:
            if not self.cancelled:
                self.cancelled = True
                await self.events_queue.put(None)

        async def events(self):
            while event := await self.events_queue.get():
                yield event

    agent = StreamingAgent()
    monkeypatch.setattr("voice_transport.app.create_agent_session", lambda _: agent)
    async with running_server(Settings("token")) as (base_url, _app):
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
                '{"type":"turn.start","turn_id":"audio-1","input":"audio"}'
            )
            await websocket.send(b"\x00\x00")
            await websocket.send('{"type":"turn.end","turn_id":"audio-1"}')
            started = json.loads(await websocket.recv())
            assert started["type"] == "assistant.response_started"
            audio = json.loads(await websocket.recv())
            assert audio["type"] == "assistant.audio"
            await websocket.send(
                json.dumps(
                    {
                        "type": "response.cancel",
                        "response_id": audio["response_id"],
                    }
                )
            )
            assert json.loads(await websocket.recv())["type"] == "assistant.interrupted"
            assert agent.interrupted
            await websocket.send(
                '{"type":"turn.start","turn_id":"text-2","input":"text"}'
            )
            await websocket.send(
                '{"type":"input.text","turn_id":"text-2","text":"next"}'
            )
            assert json.loads(await websocket.recv())["source"] == "client_text"
            await websocket.send('{"type":"turn.end","turn_id":"text-2"}')
            assert (
                json.loads(await websocket.recv())["type"]
                == "assistant.response_started"
            )
            second_audio = json.loads(await websocket.recv())
            assert second_audio["type"] == "assistant.audio"
            await websocket.send(
                '{"type":"turn.start","turn_id":"text-3","input":"text"}'
            )
            interrupted = json.loads(await websocket.recv())
            assert interrupted["type"] == "assistant.interrupted"
            assert interrupted["response_id"] == second_audio["response_id"]
            await websocket.send(
                '{"type":"input.text","turn_id":"text-3","text":"replacement"}'
            )
            assert json.loads(await websocket.recv())["source"] == "client_text"
            await websocket.send('{"type":"session.cancel"}')
            assert json.loads(await websocket.recv())["type"] == "session.finished"
        assert agent.cancelled
        async with httpx.AsyncClient() as client:
            revoked = await client.get(audio["url"])
            text_interrupted = await client.get(second_audio["url"])
        assert revoked.status_code == 404
        assert text_interrupted.status_code == 404


@pytest.mark.asyncio
async def test_signed_audio_endpoint_streams_wav_without_caching() -> None:
    async with running_server(Settings("token")) as (base_url, app):
        stream, token = app.state.audio_store.create(sample_rate=24_000)
        await stream.write(b"\x00\x00\x01\x00")
        await stream.close()
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_url}/audio/v1/{stream.stream_id}?token={token}"
            )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/wav")
        assert response.headers["cache-control"] == "no-store"
        assert response.content[:4] == b"RIFF"
        assert response.content[-4:] == b"\x00\x00\x01\x00"
        await app.state.audio_store.revoke(stream.stream_id)
        async with httpx.AsyncClient() as client:
            revoked = await client.get(
                f"{base_url}/audio/v1/{stream.stream_id}?token={token}"
            )
        assert revoked.status_code == 404
