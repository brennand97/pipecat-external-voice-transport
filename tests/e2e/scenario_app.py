"""Test-only provider injection; production transports/actors/audio store stay real.

Never ship this module/entrypoint as the production image. No provider credentials.
"""

import asyncio
import json
import math
import secrets
import struct

from fastapi import HTTPException, Request

from voice_transport.agent.fake import FakeAgentSession
from voice_transport.agent.session import AgentEvent
from voice_transport.app import create_app
from voice_transport.config import Settings
from voice_transport.providers.fake import FakeRealtimeProvider

STATS = {
    "created": 0,
    "closed": 0,
    "input_bytes": 0,
    "turns": 0,
    "output_bytes": 0,
    "interrupts": 0,
    "tool_calls": 0,
    "end_requests": 0,
}
NEXT_AUDIO_TEXT = None


class ScenarioAgent(FakeAgentSession):
    """Deterministic text/audio and delayed responses, cancellable in one session."""

    def __init__(self, config):
        super().__init__()
        self.registry = config.tool_registry
        self.output_audio = "audio" in config.output_modalities
        self._response_task = None
        self._audio_bytes = {}
        self._responded = set()
        self._pending_audio_response = None
        self._auto_audio_fired = False
        STATS["created"] += 1

    @property
    def effective_profile(self):
        return self.registry.profile_name if self.registry else None

    @property
    def effective_tool_names(self):
        return self.registry.tool_names if self.registry else ()

    async def start(self):
        await super().start()
        if self.registry:
            await self.registry.discover()

    async def submit_audio(self, turn_id, pcm):
        global NEXT_AUDIO_TEXT
        await super().submit_audio(turn_id, pcm)
        STATS["input_bytes"] += len(pcm)
        self._audio_bytes[turn_id] = self._audio_bytes.get(turn_id, 0) + len(pcm)
        # Browser external mode leaves the audio turn open while provider VAD
        # decides to respond. Model that boundary without claiming real STT.
        if (
            self._audio_bytes[turn_id] >= 64000
            and turn_id not in self._responded
            and (not self._auto_audio_fired or NEXT_AUDIO_TEXT is not None)
        ):
            self._auto_audio_fired = True
            self._responded.add(turn_id)
            STATS["turns"] += 1
            await self._events.put(AgentEvent("user.speech_started"))
            await self._events.put(
                AgentEvent("user.transcript.final", "fixture microphone input")
            )
            text, NEXT_AUDIO_TEXT = NEXT_AUDIO_TEXT, None
            self._pending_audio_response = (text,)
            # events() starts the reply only after the real actor handles VAD;
            # otherwise its interrupt would cancel the new reply, not the old one.

    async def events(self):
        async for event in super().events():
            yield event
            if (
                event.type == "user.transcript.final"
                and self._pending_audio_response
                and not self._closed
            ):
                text = self._pending_audio_response[0]
                self._pending_audio_response = None
                self._response_task = asyncio.create_task(self._respond(text))

    async def end_turn(self, turn_id):
        self._ensure_started()
        text = self._text.pop(turn_id, None)
        if turn_id in self._responded:
            return
        self._responded.add(turn_id)
        STATS["turns"] += 1
        await self.interrupt()
        self._response_task = asyncio.create_task(self._respond(text))

    async def _respond(self, text):
        await self._events.put(AgentEvent("assistant.response_started"))
        if text and "slow" in text:
            await asyncio.sleep(3)
        if text == "please end session":
            if self.registry is None:
                raise RuntimeError("End-session scenario requires the real registry")
            result = await self.registry.call("transport__EndSession", {})
            if result.is_error:
                raise RuntimeError("End-session tool failed")
            STATS["end_requests"] += 1
            return  # Real transport session-control event owns termination.
        reply = "Test response: " + (text or "received microphone audio")
        if text and text.startswith("timer:"):
            command = json.loads(text[6:])
            name = "voice_satellite__" + command["operation"]
            arguments = command.get("arguments", {})
            await self._events.put(
                AgentEvent(
                    "assistant.tool_call_started",
                    tool_call_id="timer-fixture",
                    tool_name=name,
                    tool_arguments=arguments,
                )
            )
            result = await self.registry.call(name, arguments)
            STATS["tool_calls"] += 1
            STATS["last_timer"] = {
                "name": name,
                "is_error": result.is_error,
                "content": result.content,
            }
            await self._events.put(
                AgentEvent(
                    "assistant.tool_call_finished",
                    tool_call_id="timer-fixture",
                    tool_name=name,
                    tool_arguments=arguments,
                    tool_result=result.content,
                    is_error=result.is_error,
                )
            )
            reply = "Timer tool rejected" if result.is_error else "Timer tool succeeded"
        if text == "test tool":
            if self.registry is None:
                raise RuntimeError("Tool scenario requires a real trusted registry")
            name = "intent__HassTurnOn"
            arguments = {"name": "Testbed Light", "domain": ["light"]}
            await self._events.put(
                AgentEvent(
                    "assistant.tool_call_started",
                    tool_call_id="fixture-1",
                    tool_name=name,
                    tool_arguments=arguments,
                )
            )
            result = await self.registry.call(name, arguments)
            STATS["tool_calls"] += 1
            await self._events.put(
                AgentEvent(
                    "assistant.tool_call_finished",
                    tool_call_id="fixture-1",
                    tool_name=name,
                    tool_arguments=arguments,
                    tool_result=result.content,
                    is_error=result.is_error,
                )
            )
            reply = (
                "Test light turned on" if not result.is_error else "Test tool failed"
            )
        await self._events.put(AgentEvent("assistant.text.final", reply))
        if self.output_audio:
            # Long playback exposes cancellation while real audio is streaming.
            rate = 24000
            samples = rate * 8 if text and "long" in text else 14400
            pcm = b"".join(
                struct.pack("<h", int(4000 * math.sin(2 * math.pi * 440 * i / rate)))
                for i in range(samples)
            )
            for offset in range(0, len(pcm), 4800):
                chunk = pcm[offset : offset + 4800]
                await self._events.put(
                    AgentEvent(
                        "assistant.audio.chunk",
                        audio=chunk,
                        sample_rate=rate,
                        channels=1,
                    )
                )
                STATS["output_bytes"] += len(chunk)
                await asyncio.sleep(0.05)
        await self._events.put(AgentEvent("assistant.response_finished"))

    async def interrupt(self):
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass
            STATS["interrupts"] += 1
        self._response_task = None

    async def close(self):
        if not self._closed:
            await self.interrupt()
            await super().close()
            STATS["closed"] += 1
            if self.registry:
                await self.registry.close()


def app():
    # Process-local override of the documented provider boundary, never source patching.
    FakeRealtimeProvider.create_session = lambda self, config: ScenarioAgent(config)
    settings = Settings.from_environment()
    if settings.realtime_provider != "fake" or settings.openai_api_key:
        raise RuntimeError("E2E scenario app refuses real provider credentials")
    application = create_app(settings)

    @application.get("/e2e/stats")
    async def stats(request: Request):
        header = request.headers.get("authorization", "")
        if not secrets.compare_digest(header, "Bearer " + settings.transport_token):
            raise HTTPException(401)
        return dict(STATS)

    @application.post("/e2e/next-audio")
    async def next_audio(request: Request):
        header = request.headers.get("authorization", "")
        if not secrets.compare_digest(header, "Bearer " + settings.transport_token):
            raise HTTPException(401)
        body = await request.json()
        if set(body) != {"text"} or body["text"] not in {
            "please end session",
            "test second utterance",
        }:
            raise HTTPException(400)
        # Serial browser scenarios only; scripts the model decision, not HA/UI.
        global NEXT_AUDIO_TEXT
        NEXT_AUDIO_TEXT = body["text"]
        return {"armed": True}

    return application
