"""FastAPI application and Phase 1 authenticated protocol endpoint."""

from __future__ import annotations

import asyncio
import secrets
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse

from .agent.fake import FakeAgentSession
from .audio_input import PCMInput
from .config import Settings
from .protocol import (
    ProtocolViolation,
    error_message,
    parse_json_message,
    parse_session_start,
    ready_message,
    validate_control,
)
from .session import Session, SessionRegistry


async def _send_error(
    websocket: WebSocket, error: ProtocolViolation, session_id: str | None = None
) -> None:
    await websocket.send_json(error_message(error, session_id))


async def _drain_pcm(input_pcm: PCMInput) -> None:
    """Phase 2 fake adapter; later replaced by the Pipecat input adapter."""
    async for _frame in input_pcm.frames():
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
    app.state.settings = settings
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
            # The orchestrator depends only on the isolated agent lifecycle.
            # Provider implementation selection does not affect protocol handling.
            agent = FakeAgentSession()
            await agent.start()
            drain_task = asyncio.create_task(_drain_pcm(input_pcm))
            await websocket.send_json(ready_message(start.session_id))
            session_started_at = time.monotonic()

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
                async for event in agent.events():
                    response = {
                        "type": event.type,
                        "session_id": session.start.session_id,
                    }
                    if event.text is not None:
                        response["text"] = event.text
                    await websocket.send_json(response)
                session.finish()
                await websocket.send_json(
                    {"type": "session.finished", "session_id": session.start.session_id}
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
            if input_pcm is not None:
                await input_pcm.close()
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
