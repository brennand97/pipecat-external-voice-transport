"""FastAPI application for External Transport Protocol v1."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, StreamingResponse

from .audio_output import AudioAccessError, AudioStream, AudioStreamStore
from .config import Settings
from .conversation import ConversationActor, ConversationError, TurnInput
from .protocol import (
    ProtocolViolation,
    error_message,
    parse_input_text,
    parse_json_message,
    parse_session_start,
    parse_turn_start,
    ready_message,
    validate_control,
)
from .providers import create_agent_session, prepare_provider
from .session import Session, SessionRegistry
from .session_audit import SessionAuditLog

_LOGGER = logging.getLogger(__name__)


async def _send_error(
    websocket: WebSocket, error: ProtocolViolation, session_id: str | None = None
) -> None:
    await websocket.send_json(error_message(error, session_id))


async def _expire_audio_stream(audio_store: AudioStreamStore, stream_id: str) -> None:
    await asyncio.sleep(audio_store.token_ttl_seconds)
    await audio_store.revoke(stream_id)


async def _cancel_task(task: asyncio.Task | None, timeout: float = 3.0) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    try:
        await asyncio.wait_for(task, timeout=timeout)
    except (asyncio.CancelledError, TimeoutError):
        pass


def _is_authorized(websocket: WebSocket, expected_token: str) -> bool:
    header = websocket.headers.get("authorization", "")
    scheme, _, supplied = header.partition(" ")
    return (
        scheme.lower() == "bearer"
        and bool(supplied)
        and secrets.compare_digest(supplied, expected_token)
    )


def _protocol_error(error: ConversationError) -> ProtocolViolation:
    return ProtocolViolation("invalid_turn_state", str(error))


def create_app(settings: Settings) -> FastAPI:
    prepare_provider(settings)
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
    app.state.audit = SessionAuditLog(
        Path(settings.session_audit_log_path),
        mode=settings.session_audit_mode,  # type: ignore[arg-type]
        retention_days=settings.session_audit_retention_days,
    )
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
        await app.state.audit.initialize()
        session: Session | None = None
        actor: ConversationActor | None = None
        writer_task: asyncio.Task | None = None
        streams: dict[str, AudioStream] = {}
        issued_stream_ids: set[str] = set()
        expiry_tasks: set[asyncio.Task] = set()
        implicit_turn_number = 0

        async def emit_events() -> None:
            assert actor is not None and session is not None
            async for event in actor.events():
                response_id = event.response_id
                await app.state.audit.record(
                    session.start.session_id,
                    event.type,
                    turn_id=event.turn_id,
                    response_id=response_id,
                    transcript=event.text,
                    source=event.source,
                )
                if event.type == "assistant.audio.chunk":
                    if response_id is None:
                        continue
                    stream = streams.get(response_id)
                    if stream is None:
                        if not event.sample_rate or not event.channels:
                            raise ProtocolViolation(
                                "invalid_provider_audio",
                                "Provider returned audio without a format.",
                            )
                        stream, token = audio_store.create(
                            event.sample_rate, event.channels
                        )
                        streams[response_id] = stream
                        issued_stream_ids.add(stream.stream_id)
                        base_url = settings.public_base_url or str(
                            websocket.base_url
                        ).rstrip("/")
                        await websocket.send_json(
                            {
                                "type": "assistant.audio",
                                "session_id": session.start.session_id,
                                "turn_id": event.turn_id,
                                "response_id": response_id,
                                "url": (
                                    f"{base_url}/audio/v1/{stream.stream_id}"
                                    f"?token={token}"
                                ),
                                "content_type": "audio/wav",
                            }
                        )
                    try:
                        await stream.write(event.audio or b"")
                    except AudioAccessError:
                        await audio_store.revoke(stream.stream_id)
                        streams.pop(response_id, None)
                    continue
                if event.type == "assistant.interrupted" and response_id:
                    stream = streams.pop(response_id, None)
                    if stream is not None:
                        await audio_store.revoke(stream.stream_id)
                response = {
                    "type": event.type,
                    "session_id": session.start.session_id,
                    "turn_id": event.turn_id,
                }
                if response_id is not None:
                    response["response_id"] = response_id
                if event.text is not None:
                    response["text"] = event.text
                if event.source is not None:
                    response["source"] = event.source
                await websocket.send_json(response)
                if event.type == "assistant.response_finished" and response_id:
                    stream = streams.pop(response_id, None)
                    if stream is not None:
                        await stream.close()
                        task = asyncio.create_task(
                            _expire_audio_stream(audio_store, stream.stream_id)
                        )
                        expiry_tasks.add(task)
                        task.add_done_callback(expiry_tasks.discard)

        try:
            try:
                first = await asyncio.wait_for(
                    websocket.receive(), timeout=settings.session_start_timeout_seconds
                )
            except TimeoutError as err:
                raise ProtocolViolation(
                    "session_start_timeout", "Timed out waiting for session.start."
                ) from err
            if first.get("type") != "websocket.receive" or first.get("text") is None:
                raise ProtocolViolation(
                    "invalid_first_message",
                    "The first frame must be a session.start JSON message.",
                )
            start = parse_session_start(parse_json_message(first["text"]))
            session = await registry.create(start, settings.max_input_bytes)
            await app.state.audit.record(
                start.session_id,
                "session.started",
                satellite_entity_id=start.satellite_entity_id,
                satellite_name=start.satellite_name,
            )
            agent = create_agent_session(
                settings, audit=app.state.audit, session_id=start.session_id
            )
            actor = ConversationActor(agent)
            await actor.start()
            session.mark_ready()
            await websocket.send_json(ready_message(start.session_id))
            writer_task = asyncio.create_task(emit_events())
            started_at = time.monotonic()

            while True:
                remaining = settings.max_session_seconds - (
                    time.monotonic() - started_at
                )
                if remaining <= 0:
                    raise ProtocolViolation(
                        "session_duration_exceeded",
                        "The maximum session duration was reached.",
                    )
                receive_task = asyncio.create_task(websocket.receive())
                done, _ = await asyncio.wait(
                    {receive_task, writer_task},
                    timeout=min(settings.input_idle_timeout_seconds, remaining),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    await _cancel_task(receive_task)
                    if actor.active_response_id is not None:
                        continue
                    raise ProtocolViolation(
                        "input_idle_timeout",
                        "No input was received before the idle timeout.",
                    )
                if writer_task in done:
                    await writer_task
                    raise ProtocolViolation(
                        "provider_closed", "The realtime provider closed unexpectedly."
                    )
                frame = receive_task.result()
                if frame.get("type") == "websocket.disconnect":
                    return
                if frame.get("bytes") is not None:
                    if actor.open_turn_id is None:
                        implicit_turn_number += 1
                        await actor.start_turn(
                            f"implicit-{implicit_turn_number}", TurnInput.AUDIO
                        )
                    session.add_audio(frame["bytes"], settings.max_audio_frame_bytes)
                    await actor.submit_audio(actor.open_turn_id, frame["bytes"])
                    continue
                if frame.get("text") is None:
                    raise ProtocolViolation(
                        "invalid_message",
                        "Only text controls and binary PCM frames are accepted.",
                    )
                message = parse_json_message(frame["text"])
                message_type = message["type"]
                try:
                    if message_type == "turn.start":
                        turn = parse_turn_start(message)
                        await actor.start_turn(turn.turn_id, TurnInput(turn.input_type))
                    elif message_type == "input.text":
                        turn_id, text = parse_input_text(message)
                        await actor.submit_text(turn_id, text)
                    else:
                        control = validate_control(message)
                        if control == "session.cancel":
                            await actor.cancel()
                            await websocket.send_json(
                                {
                                    "type": "session.finished",
                                    "session_id": session.start.session_id,
                                }
                            )
                            return
                        if control == "response.cancel":
                            await actor.cancel_response(message["response_id"])
                        else:
                            turn_id = message.get("turn_id") or actor.open_turn_id
                            if turn_id is None:
                                raise ConversationError("no input turn is active")
                            await actor.end_turn(turn_id)
                except ConversationError as err:
                    raise _protocol_error(err) from err
        except WebSocketDisconnect:
            return
        except ProtocolViolation as err:
            if session is not None:
                session.fail()
            await _send_error(
                websocket, err, session.start.session_id if session else None
            )
            await websocket.close(code=status.WS_1002_PROTOCOL_ERROR)
        except Exception:  # noqa: BLE001 - never leak provider/tool internals to clients
            _LOGGER.exception("Provider session startup or execution failed")
            if session is not None:
                session.fail()
            await _send_error(
                websocket,
                ProtocolViolation(
                    "provider_failure",
                    "The configured assistant tools are unavailable.",
                ),
                session.start.session_id if session else None,
            )
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        finally:
            if actor is not None:
                await actor.close()
            await _cancel_task(writer_task)
            for stream_id in issued_stream_ids:
                await audio_store.revoke(stream_id)
            for task in tuple(expiry_tasks):
                await _cancel_task(task)
            if session is not None:
                await app.state.audit.record(
                    session.start.session_id, "session.finished"
                )
                await registry.remove(session.start.session_id)

    return app


def runtime_app() -> FastAPI:
    return create_app(Settings.from_environment())
