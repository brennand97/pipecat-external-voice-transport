"""FastAPI application and Phase 1 authenticated protocol endpoint."""

from __future__ import annotations

import asyncio
import secrets
import time

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, StreamingResponse

from .audio_input import PCMInput
from .audio_output import AudioAccessError, AudioStream, AudioStreamStore
from .config import Settings
from .protocol import (
    ProtocolViolation,
    error_message,
    parse_json_message,
    parse_session_start,
    ready_message,
    validate_control,
)
from .providers import create_agent_session
from .session import Session, SessionRegistry


async def _send_error(
    websocket: WebSocket, error: ProtocolViolation, session_id: str | None = None
) -> None:
    await websocket.send_json(error_message(error, session_id))


async def _drain_pcm(input_pcm: PCMInput) -> None:
    """Phase 2 fake adapter; later replaced by the Pipecat input adapter."""
    async for _frame in input_pcm.frames():
        pass


async def _expire_audio_stream(audio_store: AudioStreamStore, stream_id: str) -> None:
    """Invalidate closed output after its short-lived signed URL expires."""
    await asyncio.sleep(audio_store.token_ttl_seconds)
    await audio_store.revoke(stream_id)


async def _cancel_task(task: asyncio.Task[object], timeout: float = 3.0) -> None:
    """Cancel a local task without allowing shutdown to block indefinitely."""
    if task.done():
        try:
            await task
        except asyncio.CancelledError:
            pass
        return
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=timeout)
    except asyncio.CancelledError:
        pass


def _is_authorized(websocket: WebSocket, expected_token: str) -> bool:
    header = websocket.headers.get("authorization", "")
    scheme, _, supplied = header.partition(" ")
    return (
        scheme.lower() == "bearer"
        and bool(supplied)
        and secrets.compare_digest(supplied, expected_token)
    )


def create_app(settings: Settings) -> FastAPI:
    """Create an application with injectable configuration for tests."""
    app = FastAPI(
        title="Pipecat External Voice Transport", docs_url=None, redoc_url=None
    )
    registry = SessionRegistry(settings.max_concurrent_sessions)
    signing_key = (
        settings.audio_url_signing_key.encode()
        if settings.audio_url_signing_key
        else secrets.token_bytes(32)
    )
    audio_store = AudioStreamStore(
        signing_key,
        token_ttl_seconds=settings.audio_url_token_ttl_seconds,
        max_buffered_chunks=settings.max_buffered_output_chunks,
        write_timeout_seconds=settings.audio_stream_write_timeout_seconds,
    )
    app.state.settings = settings
    app.state.audio_store = audio_store
    app.state.registry = registry
    app.state.ready = True

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        if app.state.ready:
            return JSONResponse({"status": "ready"})
        return JSONResponse(
            {"status": "not_ready"}, status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    @app.get("/audio/v1/{stream_id}")
    async def stream_audio(stream_id: str, token: str) -> StreamingResponse:
        try:
            stream = audio_store.open(stream_id, token)
        except AudioAccessError as err:
            raise HTTPException(
                status_code=404, detail="audio stream unavailable"
            ) from err
        return StreamingResponse(
            stream.wav_chunks(),
            media_type="audio/wav",
            headers={"Cache-Control": "no-store"},
        )

    @app.websocket("/transport/v1")
    async def transport(websocket: WebSocket) -> None:
        if not _is_authorized(websocket, settings.transport_token):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        await websocket.accept()
        session: Session | None = None
        input_pcm: PCMInput | None = None
        agent = None
        drain_task = None
        output_stream: AudioStream | None = None
        output_expiry_task: asyncio.Task[None] | None = None
        event_task: asyncio.Task[None] | None = None
        receive_task: asyncio.Task[object] | None = None
        retain_output = False
        output_disabled = False
        try:
            try:
                first = await asyncio.wait_for(
                    websocket.receive(), timeout=settings.session_start_timeout_seconds
                )
            except TimeoutError as err:
                raise ProtocolViolation(
                    "session_start_timeout",
                    "Timed out waiting for session.start.",
                ) from err
            if first.get("type") != "websocket.receive" or first.get("text") is None:
                raise ProtocolViolation(
                    "invalid_first_message",
                    "The first frame must be a session.start JSON message.",
                )
            start = parse_session_start(parse_json_message(first["text"]))
            session = await registry.create(start, settings.max_input_bytes)
            session.mark_ready()
            input_pcm = PCMInput(
                session,
                settings.max_audio_frame_bytes,
                settings.max_buffered_audio_frames,
            )
            # Provider implementation selection does not affect protocol handling.
            agent = create_agent_session(settings)
            await agent.start()
            drain_task = asyncio.create_task(_drain_pcm(input_pcm))
            await websocket.send_json(ready_message(start.session_id))
            session_started_at = time.monotonic()
            response_started = False

            async def emit_agent_events() -> None:
                nonlocal output_disabled, output_stream, response_started
                async for event in agent.events():
                    if event.type == "assistant.audio.chunk":
                        if output_disabled:
                            continue
                        if output_stream is None:
                            if not event.sample_rate or not event.channels:
                                raise ProtocolViolation(
                                    "invalid_provider_audio",
                                    "Provider returned audio without a format.",
                                )
                            output_stream, token = audio_store.create(
                                event.sample_rate, event.channels
                            )
                            base_url = settings.public_base_url or str(
                                websocket.base_url
                            ).rstrip("/")
                            await websocket.send_json(
                                {
                                    "type": "assistant.audio",
                                    "session_id": session.start.session_id,
                                    "url": (
                                        f"{base_url}/audio/v1/{output_stream.stream_id}"
                                        f"?token={token}"
                                    ),
                                    "content_type": "audio/wav",
                                }
                            )
                        try:
                            await output_stream.write(event.audio or b"")
                        except AudioAccessError:
                            # A client that never opens the signed stream must
                            # not block provider output or retain audio forever.
                            await audio_store.revoke(output_stream.stream_id)
                            output_stream = None
                            output_disabled = True
                        continue
                    if event.type == "assistant.response_started":
                        response_started = True
                    response = {
                        "type": event.type,
                        "session_id": session.start.session_id,
                    }
                    if event.text is not None:
                        response["text"] = event.text
                    await websocket.send_json(response)
                    if event.type == "assistant.response_finished" and output_stream:
                        await output_stream.close()

            while True:
                remaining = settings.max_session_seconds - (
                    time.monotonic() - session_started_at
                )
                if remaining <= 0:
                    raise ProtocolViolation(
                        "session_duration_exceeded",
                        "The maximum session duration was reached.",
                    )
                receive_timeout = min(settings.input_idle_timeout_seconds, remaining)
                timeout_code = (
                    "input_idle_timeout"
                    if receive_timeout == settings.input_idle_timeout_seconds
                    else "session_duration_exceeded"
                )
                try:
                    frame = await asyncio.wait_for(
                        websocket.receive(), timeout=receive_timeout
                    )
                except TimeoutError as err:
                    message = (
                        "No input was received before the idle timeout."
                        if timeout_code == "input_idle_timeout"
                        else "The maximum session duration was reached."
                    )
                    raise ProtocolViolation(timeout_code, message) from err
                if frame.get("type") == "websocket.disconnect":
                    return
                if frame.get("bytes") is not None:
                    await input_pcm.put(frame["bytes"])
                    await agent.push_audio(frame["bytes"])
                    continue
                if frame.get("text") is None:
                    raise ProtocolViolation(
                        "invalid_message",
                        "Only text controls and binary PCM frames are accepted.",
                    )
                control = validate_control(parse_json_message(frame["text"]))
                if control == "session.cancel":
                    session.cancel()
                    await input_pcm.close()
                    await drain_task
                    await agent.cancel()
                    session.finish()
                    await websocket.send_json(
                        {
                            "type": "session.finished",
                            "session_id": session.start.session_id,
                        }
                    )
                    return
                session.end_input()
                await input_pcm.close()
                await drain_task
                await agent.end_input()
                event_task = asyncio.create_task(emit_agent_events())
                receive_task = asyncio.create_task(websocket.receive())
                while True:
                    done, _pending = await asyncio.wait(
                        {event_task, receive_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if event_task in done:
                        await event_task
                        await _cancel_task(receive_task)
                        if output_stream is not None:
                            retain_output = True
                            output_expiry_task = asyncio.create_task(
                                _expire_audio_stream(
                                    audio_store, output_stream.stream_id
                                )
                            )
                        session.finish()
                        await websocket.send_json(
                            {
                                "type": "session.finished",
                                "session_id": session.start.session_id,
                            }
                        )
                        return
                    received = receive_task.result()
                    if received.get("type") == "websocket.disconnect":
                        await agent.cancel()
                        await _cancel_task(event_task)
                        return
                    if received.get("text") is None:
                        raise ProtocolViolation(
                            "invalid_state",
                            "Only session.cancel is accepted while responding.",
                        )
                    control = validate_control(parse_json_message(received["text"]))
                    if control != "session.cancel":
                        raise ProtocolViolation(
                            "invalid_state",
                            "Only session.cancel is accepted while responding.",
                        )
                    session.cancel()
                    await agent.cancel()
                    await _cancel_task(event_task)
                    if output_stream is not None:
                        await audio_store.revoke(output_stream.stream_id)
                    if response_started:
                        await websocket.send_json(
                            {
                                "type": "assistant.interrupted",
                                "session_id": session.start.session_id,
                            }
                        )
                    session.finish()
                    await websocket.send_json(
                        {
                            "type": "session.finished",
                            "session_id": session.start.session_id,
                        }
                    )
                    return
        except WebSocketDisconnect:
            return
        except ProtocolViolation as err:
            if session is not None:
                session.fail()
            await _send_error(
                websocket, err, session.start.session_id if session else None
            )
            await websocket.close(code=status.WS_1002_PROTOCOL_ERROR)
        finally:
            if receive_task is not None:
                await _cancel_task(receive_task)
            if event_task is not None:
                await _cancel_task(event_task)
            if input_pcm is not None:
                await input_pcm.close()
            if output_stream is not None and not retain_output:
                await audio_store.revoke(output_stream.stream_id)
            if output_expiry_task is not None and not retain_output:
                output_expiry_task.cancel()
            if agent is not None:
                await agent.close()
            if drain_task is not None:
                await drain_task
            if session is not None:
                await registry.remove(session.start.session_id)

    return app


def runtime_app() -> FastAPI:
    """Uvicorn application factory that reads production environment settings."""
    return create_app(Settings.from_environment())
