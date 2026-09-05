import struct
from pathlib import Path

FIXTURE = Path(__file__).parents[1] / "fixtures" / "tell-me-a-short-joke.pcm"


def test_synthetic_speech_fixture_is_protocol_pcm16() -> None:
    pcm = FIXTURE.read_bytes()
    assert len(pcm) == 84_224
    assert len(pcm) % 2 == 0
    assert len(pcm) / 32_000 == 2.632
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    assert max(abs(sample) for sample in samples) == 20_764
    assert pcm[-32_000:] == b"\x00" * 32_000
