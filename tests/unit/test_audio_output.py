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


def test_audio_stream_rejects_invalid_token() -> None:
    store = AudioStreamStore(b"test-signing-key")
    stream, _ = store.create(sample_rate=24_000)
    with pytest.raises(AudioAccessError, match="malformed"):
        store.open(stream.stream_id, "not-a-token")


async def test_closing_a_full_audio_queue_does_not_block() -> None:
    store = AudioStreamStore(b"test-signing-key", max_buffered_chunks=1)
    stream, _ = store.create(sample_rate=24_000)
    await stream.write(b"\x00\x00")
    await asyncio.wait_for(stream.close(), timeout=0.1)
