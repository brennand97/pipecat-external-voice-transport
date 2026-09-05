"""FastAPI application and Phase 1 authenticated protocol endpoint."""

from __future__ import annotations

import asyncio
import secrets

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse

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
        drain_task = None
        try:
            first = await websocket.receive()
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
            drain_task = asyncio.create_task(_drain_pcm(input_pcm))
            await websocket.send_json(ready_message(start.session_id))

            while True:
                frame = await websocket.receive()
                if frame.get("type") == "websocket.disconnect":
                    return
                if frame.get("bytes") is not None:
                    await input_pcm.put(frame["bytes"])
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
                # The deterministic fake agent validates framing, ordering, lifecycle,
                # framing, ordering, lifecycle, and cleanup without a billable provider.
                await websocket.send_json(
                    {
                        "type": "assistant.response_started",
                        "session_id": session.start.session_id,
                    }
                )
                await websocket.send_json(
                    {
                        "type": "assistant.text.final",
                        "session_id": session.start.session_id,
                        "text": "External Transport test session received audio.",
                    }
                )
                await websocket.send_json(
                    {
                        "type": "assistant.response_finished",
                        "session_id": session.start.session_id,
                    }
                )
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
            if drain_task is not None:
                await drain_task
            if session is not None:
                await registry.remove(session.start.session_id)

    return app


def runtime_app() -> FastAPI:
    """Uvicorn application factory that reads production environment settings."""
    return create_app(Settings.from_environment())
