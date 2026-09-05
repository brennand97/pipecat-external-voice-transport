import asyncio
import time

from voice_transport.providers.openai_realtime import _cancel_bounded


async def test_provider_cancellation_is_bounded() -> None:
    blocked = asyncio.Event()

    async def never_finishes() -> None:
        await blocked.wait()

    started = time.monotonic()
    await _cancel_bounded(never_finishes(), timeout=0.01)
    assert time.monotonic() - started < 0.2
