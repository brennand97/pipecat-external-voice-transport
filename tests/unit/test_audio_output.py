import asyncio

import pytest

from voice_transport.audio_output import AudioAccessError, AudioStreamStore


async def test_signed_audio_stream_emits_wav_header_and_pcm() -> None:
    store = AudioStreamStore(b"test-signing-key", token_ttl_seconds=60)
    stream, token = store.create(sample_rate=24_000)
    assert store.open(stream.stream_id, token) is stream
    await stream.write(b"\x00\x00\x01\x00")
    await stream.close()
    chunks = [chunk async for chunk in stream.wav_chunks()]
    assert chunks[0][:4] == b"RIFF"
    assert chunks[0][8:12] == b"WAVE"
    assert chunks[1] == b"\x00\x00\x01\x00"


async def test_audio_stream_rejects_expired_or_revoked_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1_000.0
    monkeypatch.setattr("voice_transport.audio_output.time.time", lambda: now)
    store = AudioStreamStore(b"test-signing-key", token_ttl_seconds=1)
    stream, token = store.create(sample_rate=24_000)

    monkeypatch.setattr("voice_transport.audio_output.time.time", lambda: now + 2)
    with pytest.raises(AudioAccessError, match="expired"):
        store.open(stream.stream_id, token)

    monkeypatch.setattr("voice_transport.audio_output.time.time", lambda: now)
    await store.revoke(stream.stream_id)
    with pytest.raises(AudioAccessError, match="unavailable"):
        store.open(stream.stream_id, token)


async def test_revoking_an_open_stream_stops_the_consumer() -> None:
    store = AudioStreamStore(b"test-signing-key")
    stream, _ = store.create(sample_rate=24_000)
    chunks = stream.wav_chunks()
    assert (await anext(chunks))[:4] == b"RIFF"
    await stream.write(b"\x00\x00")
    assert await anext(chunks) == b"\x00\x00"

    await store.revoke(stream.stream_id)
    with pytest.raises(StopAsyncIteration):
        await anext(chunks)


def test_audio_stream_rejects_invalid_token() -> None:
    store = AudioStreamStore(b"test-signing-key")
    stream, _ = store.create(sample_rate=24_000)
    with pytest.raises(AudioAccessError, match="malformed"):
        store.open(stream.stream_id, "not-a-token")


async def test_audio_stream_write_is_bounded_without_a_consumer() -> None:
    store = AudioStreamStore(
        b"test-signing-key", max_buffered_chunks=1, write_timeout_seconds=0.01
    )
    stream, _ = store.create(sample_rate=24_000)
    await stream.write(b"\x00\x00")
    with pytest.raises(AudioAccessError, match="not keeping up"):
        await stream.write(b"\x00\x00")


async def test_closing_a_full_audio_queue_does_not_block() -> None:
    store = AudioStreamStore(b"test-signing-key", max_buffered_chunks=1)
    stream, _ = store.create(sample_rate=24_000)
    await stream.write(b"\x00\x00")
    await asyncio.wait_for(stream.close(), timeout=0.1)
