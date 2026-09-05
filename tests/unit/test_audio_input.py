import asyncio

import pytest

from voice_transport.audio_input import PCMInput
from voice_transport.protocol import ProtocolViolation, SessionStart
from voice_transport.session import Session

_TIMEOUT_SECONDS = 1


def make_input(max_frames: int = 2) -> PCMInput:
    session = Session(
        SessionStart("id", "assist_satellite.kitchen", "Kitchen", None, None), 100
    )
    session.mark_ready()
    return PCMInput(session, max_frame_bytes=4, max_frames=max_frames)


async def test_pcm_frames_are_consumed_in_arrival_order() -> None:
    pcm_input = make_input()

    async def collect() -> list[bytes]:
        return [frame async for frame in pcm_input.frames()]

    consumer = asyncio.create_task(collect())
    await asyncio.wait_for(pcm_input.put(b"\x01\x00"), _TIMEOUT_SECONDS)
    await asyncio.wait_for(pcm_input.put(b"\x02\x00"), _TIMEOUT_SECONDS)
    await asyncio.wait_for(pcm_input.close(), _TIMEOUT_SECONDS)
    assert await asyncio.wait_for(consumer, _TIMEOUT_SECONDS) == [
        b"\x01\x00",
        b"\x02\x00",
    ]


async def test_pcm_input_backpressures_when_queue_is_full() -> None:
    pcm_input = make_input(max_frames=1)
    await asyncio.wait_for(pcm_input.put(b"\x01\x00"), _TIMEOUT_SECONDS)
    blocked_put = asyncio.create_task(pcm_input.put(b"\x02\x00"))
    await asyncio.sleep(0)
    assert not blocked_put.done()

    frames = pcm_input.frames()
    assert await asyncio.wait_for(anext(frames), _TIMEOUT_SECONDS) == b"\x01\x00"
    await asyncio.wait_for(blocked_put, _TIMEOUT_SECONDS)
    assert await asyncio.wait_for(anext(frames), _TIMEOUT_SECONDS) == b"\x02\x00"
    await asyncio.wait_for(pcm_input.close(), _TIMEOUT_SECONDS)


async def test_closed_pcm_input_rejects_more_frames() -> None:
    pcm_input = make_input()
    await asyncio.wait_for(pcm_input.close(), _TIMEOUT_SECONDS)
    with pytest.raises(ProtocolViolation, match="closed"):
        await pcm_input.put(b"\x00\x00")
