"""OpenAI Realtime implementation of the generic provider contract."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

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

    return ReadyOpenAIRealtimeService


class OpenAIRealtimeProvider:
    """Create Pipecat-backed sessions for OpenAI's Realtime API."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def create_session(self, config: RealtimeProviderConfig) -> AgentSession:
        return OpenAIRealtimeAgentSession(self._api_key, self._model, config)


class OpenAIRealtimeAgentSession:
    """One isolated Pipecat pipeline and worker for an OpenAI turn."""

    def __init__(
        self,
        api_key: str,
        model: str,
        config: RealtimeProviderConfig,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._config = config
        self._events: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        self._pipeline_ready = asyncio.Event()
        self._provider_ready = asyncio.Event()
        self._source = None
        self._runner = None
        self._runner_task: asyncio.Task[None] | None = None
        self._closed = False
        self._events_finished = False
        self._tool_bridge: PipecatToolBridge | None = None

    async def start(self) -> None:
        """Start the same compact pipeline shape used by Pipecat examples."""
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.worker import PipelineParams, PipelineWorker
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMContextAggregatorPair,
        )
        from pipecat.services.openai.realtime.events import (
            AudioConfiguration,
            AudioInput,
            InputAudioNoiseReduction,
            InputAudioTranscription,
            SemanticTurnDetection,
            SessionProperties,
        )
        from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
        from pipecat.workers.runner import WorkerRunner

        self._source = _PipecatPCMSource(
            self._config.input_sample_rate, self._config.input_channels
        )
        tool_schemas = []
        if self._config.tool_registry is not None:
            self._tool_bridge = PipecatToolBridge(self._config.tool_registry)
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
                session_properties=SessionProperties(
                    audio=AudioConfiguration(
                        input=AudioInput(
                            transcription=InputAudioTranscription(),
                            turn_detection=SemanticTurnDetection(),
                            noise_reduction=InputAudioNoiseReduction(type="near_field"),
                        )
                    )
                ),
            ),
        )
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

    async def push_audio(self, pcm: bytes) -> None:
        if self._source is None:
            raise RuntimeError("OpenAI Realtime session has not started")
        await self._source.push_audio(pcm)

    async def end_input(self) -> None:
        """Let provider-side semantic VAD close the user turn."""

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
        from pipecat.processors.frame_processor import FrameProcessor

        class PCMSource(FrameProcessor):
            async def process_frame(self, frame, direction) -> None:
                await super().process_frame(frame, direction)
                await self.push_frame(frame, direction)

            async def push_audio(self, pcm: bytes) -> None:
                from pipecat.frames.frames import InputAudioRawFrame

                await self.push_frame(
                    InputAudioRawFrame(
                        audio=pcm,
                        sample_rate=sample_rate,
                        num_channels=channels,
                    )
                )

        return PCMSource(name="external-pcm-source")


class _PipecatEventSink:
    """Translate Pipecat lifecycle/text frames without leaking them outward."""

    def __new__(
        cls,
        events: asyncio.Queue[AgentEvent | None],
        pipeline_ready: asyncio.Event,
    ):
        from pipecat.processors.frame_processor import FrameProcessor

        class EventSink(FrameProcessor):
            def __init__(self, **kwargs) -> None:
                super().__init__(**kwargs)
                self._text_parts: list[str] = []

            async def process_frame(self, frame, direction) -> None:
                await super().process_frame(frame, direction)
                from pipecat.frames.frames import (
                    LLMFullResponseEndFrame,
                    LLMFullResponseStartFrame,
                    StartFrame,
                    TextFrame,
                    TTSAudioRawFrame,
                )

                if isinstance(frame, StartFrame):
                    pipeline_ready.set()
                elif isinstance(frame, LLMFullResponseStartFrame):
                    await events.put(AgentEvent("assistant.response_started"))
                elif isinstance(frame, TextFrame):
                    # Realtime can surface the same output transcript through
                    # more than one provider event. Preserve audio unchanged,
                    # but avoid presenting duplicated adjacent text chunks.
                    if not self._text_parts or self._text_parts[-1] != frame.text:
                        self._text_parts.append(frame.text)
                        await events.put(AgentEvent("assistant.text.delta", frame.text))
                elif isinstance(frame, TTSAudioRawFrame):
                    await events.put(
                        AgentEvent(
                            "assistant.audio.chunk",
                            audio=frame.audio,
                            sample_rate=frame.sample_rate,
                            channels=frame.num_channels,
                        )
                    )
                elif isinstance(frame, LLMFullResponseEndFrame):
                    if self._text_parts:
                        await events.put(
                            AgentEvent(
                                "assistant.text.final", "".join(self._text_parts)
                            )
                        )
                    await events.put(AgentEvent("assistant.response_finished"))
                    await events.put(None)
                await self.push_frame(frame, direction)

        return EventSink(name="external-event-sink")
