"""OpenAI Realtime implementation of the generic provider contract."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    TextFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.openai.realtime import events as realtime_events
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    AudioOutput,
    InputAudioNoiseReduction,
    InputAudioTranscription,
    SemanticTurnDetection,
    SessionProperties,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.workers.runner import WorkerRunner

from voice_transport.agent.session import AgentEvent, AgentSession
from voice_transport.tools.pipecat_bridge import PipecatToolBridge

from .base import RealtimeProviderConfig


async def _cancel_bounded(awaitable, timeout: float) -> None:
    task = asyncio.create_task(awaitable)
    await _wait_task_bounded(task, timeout)


async def _wait_task_bounded(task: asyncio.Task, timeout: float) -> None:
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if done:
        # Retrieve exceptions so provider shutdown failures cannot become
        # unobserved task warnings during session cleanup.
        _consume_task_result(task)
        return
    task.cancel()
    task.add_done_callback(_consume_task_result)


def _consume_task_result(task: asyncio.Task) -> None:
    try:
        task.result()
    except (asyncio.CancelledError, Exception):
        pass


def _normalize_text_chunk(previous_parts: list[str], chunk: str) -> str:
    """Repair a missing space at an unambiguous streamed sentence boundary."""
    if not previous_parts or not chunk or chunk[0].isspace():
        return chunk
    previous = previous_parts[-1]
    if previous.endswith((".", "?", "!")) and chunk[0].isupper():
        return f" {chunk}"
    return chunk


def _session_properties(
    config: RealtimeProviderConfig, model: str, voice: str, tool_schemas: list[object]
) -> SessionProperties:
    """Build the initial OpenAI session.update without unused audio capability."""
    audio: AudioConfiguration | None = None
    if "audio" in config.input_modalities or "audio" in config.output_modalities:
        audio = AudioConfiguration(
            input=(
                AudioInput(
                    transcription=InputAudioTranscription(),
                    turn_detection=SemanticTurnDetection(),
                    noise_reduction=InputAudioNoiseReduction(type="near_field"),
                )
                if "audio" in config.input_modalities
                else None
            ),
            output=AudioOutput(voice=voice)
            if "audio" in config.output_modalities
            else None,
        )
    return SessionProperties(
        model=model,
        output_modalities=sorted(config.output_modalities),
        tools=tool_schemas,
        audio=audio,
    )


def _ready_openai_service(
    service_class,
    ready_event: asyncio.Event,
    events: asyncio.Queue[AgentEvent | None],
):
    """Create a provider-local service with explicit readiness/transcripts."""

    class ReadyOpenAIRealtimeService(service_class):
        async def _handle_evt_session_updated(self, event):
            await super()._handle_evt_session_updated(event)
            ready_event.set()

        async def handle_evt_input_audio_transcription_completed(self, event):
            await super().handle_evt_input_audio_transcription_completed(event)
            transcript = getattr(event, "transcript", None)
            if isinstance(transcript, str) and transcript:
                await events.put(AgentEvent("user.transcript.final", text=transcript))

        async def _handle_context(self, context) -> None:
            # In a native realtime conversation, the first context arrives
            # upstream from the assistant aggregator after OpenAI has already
            # completed a response. Treat it as a mirror, not a request for an
            # unsolicited additional response.
            if self._context is None:
                self._context = context
                # The generic realtime implementation would create a response
                # here. External audio turns are server-VAD driven, so do not
                # create one yet—but do send the context's tool schemas and
                # system instruction to OpenAI before the first user turn.
                await self._process_completed_function_calls(send_new_results=False)
                await self._send_session_update()
                self._llm_needs_conversation_setup = False
                return
            await super()._handle_context(context)

        async def submit_text(self, text: str) -> None:
            # Audio-first realtime sessions don't receive an LLMContextFrame,
            # but Pipecat's response helper requires local context bookkeeping.
            # OpenAI already owns the live conversation, so initialize an empty
            # mirror without replaying prior server-side items.
            if self._context is None:
                self._context = LLMContext([])
                self._llm_needs_conversation_setup = False
            self._llm_needs_conversation_setup = False
            item = realtime_events.ConversationItem(
                type="message",
                role="user",
                content=[realtime_events.ItemContent(type="input_text", text=text)],
            )
            self._messages_added_manually[item.id] = True
            await self.send_client_event(
                realtime_events.ConversationItemCreateEvent(item=item)
            )
            if self._context is not None:
                self._context.add_message({"role": "user", "content": text})
            await self._create_response()

    return ReadyOpenAIRealtimeService


class OpenAIRealtimeProvider:
    """Create Pipecat-backed sessions for OpenAI's Realtime API."""

    def __init__(self, api_key: str, model: str, voice: str) -> None:
        self._api_key = api_key
        self._model = model
        self._voice = voice

    def create_session(self, config: RealtimeProviderConfig) -> AgentSession:
        return OpenAIRealtimeAgentSession(
            self._api_key, self._model, config.output_voice or self._voice, config
        )


class OpenAIRealtimeAgentSession:
    """One isolated Pipecat pipeline and worker for an OpenAI turn."""

    def __init__(
        self,
        api_key: str,
        model: str,
        voice: str,
        config: RealtimeProviderConfig,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._config = config
        self._events: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        self._pipeline_ready = asyncio.Event()
        self._provider_ready = asyncio.Event()
        self._source = None
        self._llm = None
        self._runner = None
        self._runner_task: asyncio.Task[None] | None = None
        self._closed = False
        self._events_finished = False
        self._pending_text: dict[str, str] = {}
        self._tool_bridge: PipecatToolBridge | None = None

    async def start(self) -> None:
        """Start the pre-imported compact Pipecat pipeline."""
        self._source = _PipecatPCMSource(
            self._config.input_sample_rate, self._config.input_channels
        )
        tool_schemas = []
        if self._config.tool_registry is not None:
            self._tool_bridge = PipecatToolBridge(
                self._config.tool_registry, emit_event=self._events.put
            )
            tool_schemas = await self._tool_bridge.function_schemas()
        context = LLMContext([], tools=tool_schemas)
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

        sink = _PipecatEventSink(self._events, self._pipeline_ready)
        ready_service_class = _ready_openai_service(
            OpenAIRealtimeLLMService, self._provider_ready, self._events
        )
        llm = ready_service_class(
            api_key=self._api_key,
            settings=OpenAIRealtimeLLMService.Settings(
                model=self._model,
                system_instruction=self._config.system_instruction,
                session_properties=_session_properties(
                    self._config, self._model, self._voice, tool_schemas
                ),
            ),
        )
        # Text-first turns can reach OpenAI before the first LLMContextFrame.
        # Synchronize the handler-carrying schemas now through Pipecat's own
        # schema-managed registration path. This makes handlers available for
        # the initial session.update without legacy duplicate registration.
        llm._sync_registered_tool_handlers(tool_schemas)
        self._llm = llm
        worker = PipelineWorker(
            Pipeline(
                [
                    self._source,
                    user_aggregator,
                    llm,
                    sink,
                    assistant_aggregator,
                ]
            ),
            idle_timeout_secs=300,
            params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        )
        self._runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
        await self._runner.add_workers(worker)
        self._runner_task = asyncio.create_task(self._runner.run(auto_end=False))
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    self._pipeline_ready.wait(), self._provider_ready.wait()
                ),
                timeout=10,
            )
        except TimeoutError:
            await self.close()
            raise RuntimeError("Pipecat pipeline did not become ready") from None

    async def submit_audio(self, turn_id: str, pcm: bytes) -> None:
        if self._source is None:
            raise RuntimeError("OpenAI Realtime session has not started")
        await self._source.push_audio(pcm)

    async def submit_text(self, turn_id: str, text: str) -> None:
        if self._llm is None:
            raise RuntimeError("OpenAI Realtime session has not started")
        self._pending_text[turn_id] = text

    async def end_turn(self, turn_id: str) -> None:
        """Audio uses provider VAD; dispatch a completed text turn explicitly."""
        text = self._pending_text.pop(turn_id, None)
        if text is not None:
            await self._llm.submit_text(text)

    async def interrupt(self) -> None:
        if self._source is None:
            raise RuntimeError("OpenAI Realtime session has not started")
        await self._source.interrupt()

    async def cancel(self) -> None:
        await self._stop_pipeline("external_transport_cancelled")
        if self._tool_bridge is not None:
            await self._tool_bridge.close()
        await self._finish_events()

    async def close(self) -> None:
        if self._closed:
            return
        await self._stop_pipeline("external_transport_closed")
        if self._tool_bridge is not None:
            await self._tool_bridge.close()
        await self._finish_events()

    async def _stop_pipeline(self, reason: str) -> None:
        """Stop provider tasks without allowing a remote close to block cleanup."""
        if self._runner is not None:
            await _cancel_bounded(self._runner.cancel(reason), timeout=3)
        if self._runner_task is not None and not self._runner_task.done():
            self._runner_task.cancel()
            await _wait_task_bounded(self._runner_task, timeout=3)

    async def events(self) -> AsyncIterator[AgentEvent]:
        while event := await self._events.get():
            yield event

    async def _finish_events(self) -> None:
        if self._events_finished:
            return
        self._events_finished = True
        self._closed = True
        await self._events.put(None)


class _PipecatPCMSource:
    """Lazy wrapper so Pipecat remains private to the OpenAI provider module."""

    def __new__(cls, sample_rate: int, channels: int):
        class PCMSource(FrameProcessor):
            async def process_frame(self, frame, direction) -> None:
                await super().process_frame(frame, direction)
                await self.push_frame(frame, direction)

            async def push_audio(self, pcm: bytes) -> None:
                await self.push_frame(
                    InputAudioRawFrame(
                        audio=pcm,
                        sample_rate=sample_rate,
                        num_channels=channels,
                    )
                )

            async def interrupt(self) -> None:
                await self.push_frame(InterruptionFrame())

        return PCMSource(name="external-pcm-source")


class _PipecatEventSink:
    """Translate Pipecat lifecycle/text frames without leaking them outward."""

    def __new__(
        cls,
        events: asyncio.Queue[AgentEvent | None],
        pipeline_ready: asyncio.Event,
    ):
        class EventSink(FrameProcessor):
            def __init__(self, **kwargs) -> None:
                super().__init__(**kwargs)
                self._text_parts: list[str] = []
                self._response_active = False

            async def _ensure_response_started(self) -> None:
                if self._response_active:
                    return
                # OpenAI Realtime can begin the post-tool continuation with a
                # text/audio frame rather than another LLMFullResponseStartFrame.
                self._response_active = True
                self._text_parts = []
                await events.put(AgentEvent("assistant.response_started"))

            async def process_frame(self, frame, direction) -> None:
                await super().process_frame(frame, direction)
                if isinstance(frame, StartFrame):
                    pipeline_ready.set()
                elif isinstance(frame, UserStartedSpeakingFrame):
                    await events.put(AgentEvent("user.speech_started"))
                elif isinstance(frame, InterruptionFrame):
                    self._response_active = False
                    self._text_parts = []
                elif isinstance(frame, LLMFullResponseStartFrame):
                    await self._ensure_response_started()
                elif isinstance(frame, TextFrame):
                    await self._ensure_response_started()
                    # Realtime can surface the same output transcript through
                    # more than one provider event. Preserve audio unchanged,
                    # but avoid presenting duplicated adjacent text chunks.
                    if not self._text_parts or self._text_parts[-1] != frame.text:
                        chunk = _normalize_text_chunk(self._text_parts, frame.text)
                        self._text_parts.append(chunk)
                        await events.put(AgentEvent("assistant.text.delta", chunk))
                elif isinstance(frame, TTSAudioRawFrame):
                    await self._ensure_response_started()
                    await events.put(
                        AgentEvent(
                            "assistant.audio.chunk",
                            audio=frame.audio,
                            sample_rate=frame.sample_rate,
                            channels=frame.num_channels,
                        )
                    )
                elif isinstance(frame, LLMFullResponseEndFrame):
                    if not self._response_active:
                        await self.push_frame(frame, direction)
                        return
                    self._response_active = False
                    if self._text_parts:
                        await events.put(
                            AgentEvent(
                                "assistant.text.final", "".join(self._text_parts)
                            )
                        )
                    await events.put(AgentEvent("assistant.response_finished"))
                await self.push_frame(frame, direction)

        return EventSink(name="external-event-sink")
