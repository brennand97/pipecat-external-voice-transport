import asyncio
from collections.abc import AsyncIterator

from voice_transport.agent.session import AgentEvent
from voice_transport.conversation import ConversationActor, TurnInput


class Provider:
    def __init__(self) -> None:
        self.events_queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        self.calls: list[tuple] = []

    async def start(self) -> None:
        pass

    async def submit_audio(self, turn_id: str, pcm: bytes) -> None:
        self.calls.append(("audio", turn_id, pcm))

    async def submit_text(self, turn_id: str, text: str) -> None:
        self.calls.append(("text", turn_id, text))

    async def end_turn(self, turn_id: str) -> None:
        self.calls.append(("end", turn_id))

    async def interrupt(self) -> None:
        self.calls.append(("interrupt",))

    async def close(self) -> None:
        await self.events_queue.put(None)

    async def events(self) -> AsyncIterator[AgentEvent]:
        while event := await self.events_queue.get():
            yield event


async def test_close_runs_provider_cleanup_in_the_calling_task() -> None:
    class TaskBoundProvider(Provider):
        def __init__(self) -> None:
            super().__init__()
            self.close_task: asyncio.Task[object] | None = None

        async def close(self) -> None:
            self.close_task = asyncio.current_task()
            await super().close()

    provider = TaskBoundProvider()
    actor = ConversationActor(provider)
    await actor.start()
    caller = asyncio.current_task()

    await actor.close()

    assert provider.close_task is caller


async def test_text_turn_interrupts_response_and_echoes_transcript() -> None:
    provider = Provider()
    actor = ConversationActor(provider)
    await actor.start()
    await actor.start_turn("one", TurnInput.AUDIO)
    await actor.end_turn("one")
    await provider.events_queue.put(AgentEvent("assistant.response_started"))
    first = await anext(actor.events())
    assert first.type == "assistant.response_started"
    await actor.start_turn("two", TurnInput.TEXT)
    await actor.submit_text("two", "hello")
    events = [await anext(actor.events()), await anext(actor.events())]
    assert [event.type for event in events] == [
        "assistant.interrupted",
        "user.transcript.final",
    ]
    assert events[1].source == "client_text"
    await actor.close()


async def test_two_sequential_turns_get_distinct_responses() -> None:
    provider = Provider()
    actor = ConversationActor(provider)
    await actor.start()
    events = actor.events()
    response_ids = []
    for turn_id in ("one", "two"):
        await actor.start_turn(turn_id, TurnInput.AUDIO)
        await actor.submit_audio(turn_id, b"\x00\x00")
        await actor.end_turn(turn_id)
        await provider.events_queue.put(AgentEvent("assistant.response_started"))
        started = await anext(events)
        response_ids.append(started.response_id)
        await provider.events_queue.put(AgentEvent("assistant.response_finished"))
        assert (await anext(events)).turn_id == turn_id
    assert response_ids == ["1", "2"]
    await actor.close()


async def test_audio_does_not_interrupt_until_provider_detects_speech() -> None:
    provider = Provider()
    actor = ConversationActor(provider)
    await actor.start()
    events = actor.events()
    await actor.start_turn("one", TurnInput.TEXT)
    await actor.submit_text("one", "first")
    await anext(events)  # client text transcript
    await actor.end_turn("one")
    await provider.events_queue.put(AgentEvent("assistant.response_started"))
    started = await anext(events)

    await actor.start_turn("two", TurnInput.AUDIO)
    await actor.submit_audio("two", b"\x00\x00")
    assert provider.calls[-1] == ("audio", "two", b"\x00\x00")
    assert actor.active_response_id == started.response_id

    await provider.events_queue.put(AgentEvent("user.speech_started"))
    interrupted = await anext(events)
    speech = await anext(events)
    assert interrupted.type == "assistant.interrupted"
    assert interrupted.turn_id == "one"
    assert speech.type == "user.speech_started"
    assert speech.turn_id == "two"
    await actor.close()


async def test_late_interrupted_response_events_are_fenced() -> None:
    provider = Provider()
    actor = ConversationActor(provider)
    await actor.start()
    events = actor.events()
    await actor.start_turn("one", TurnInput.TEXT)
    await actor.submit_text("one", "first")
    await anext(events)
    await actor.end_turn("one")
    await provider.events_queue.put(AgentEvent("assistant.response_started"))
    await anext(events)

    await actor.start_turn("two", TurnInput.TEXT)
    assert (await anext(events)).type == "assistant.interrupted"
    await provider.events_queue.put(
        AgentEvent(
            "assistant.audio.chunk", audio=b"old", sample_rate=24_000, channels=1
        )
    )
    await provider.events_queue.put(AgentEvent("assistant.response_finished"))
    await actor.submit_text("two", "second")
    transcript = await anext(events)
    assert transcript.type == "user.transcript.final"
    await actor.end_turn("two")
    await provider.events_queue.put(AgentEvent("assistant.response_started"))
    await provider.events_queue.put(AgentEvent("assistant.text.final", "new"))
    assert (await anext(events)).turn_id == "two"
    final = await anext(events)
    assert final.text == "new"
    await actor.close()
