import asyncio
import time

from voice_transport.providers.openai_realtime import (
    _cancel_bounded,
    _ready_openai_service,
)


async def test_provider_ready_gate_follows_session_update() -> None:
    ready = asyncio.Event()
    events: asyncio.Queue = asyncio.Queue()

    class Service:
        def __init__(self) -> None:
            self.updated = False

        async def _handle_evt_session_updated(self, event) -> None:
            self.updated = event == "updated"

        async def handle_evt_input_audio_transcription_completed(self, event) -> None:
            pass

    service = _ready_openai_service(Service, ready, events)()
    await service._handle_evt_session_updated("updated")

    assert service.updated
    assert ready.is_set()


async def test_provider_emits_only_native_final_transcripts() -> None:
    events: asyncio.Queue = asyncio.Queue()

    class Service:
        async def _handle_evt_session_updated(self, event) -> None:
            pass

        async def handle_evt_input_audio_transcription_completed(self, event) -> None:
            self.event = event

    event = type("Transcription", (), {"transcript": "Hello there"})()
    service = _ready_openai_service(Service, asyncio.Event(), events)()
    await service.handle_evt_input_audio_transcription_completed(event)

    transcript = await events.get()
    assert transcript.type == "user.transcript.final"
    assert transcript.text == "Hello there"


async def test_provider_cancellation_is_bounded() -> None:
    blocked = asyncio.Event()

    async def never_finishes() -> None:
        await blocked.wait()

    started = time.monotonic()
    await _cancel_bounded(never_finishes(), timeout=0.01)
    assert time.monotonic() - started < 0.2
