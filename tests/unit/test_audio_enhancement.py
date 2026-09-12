from __future__ import annotations

import asyncio
from array import array

import pytest

from voice_transport.audio_enhancement import (
    AudioEnhancementError,
    AudioInputEnhancementConfig,
    DisabledAudioEnhancer,
    GtcrnAudioEnhancer,
    parse_audio_input_config,
)


class FakeModel:
    sample_rate = 16_000
    frame_shift_in_samples = 256

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[list[float]] = []

    def process(self, samples: list[float]) -> list[float]:
        self.calls.append(samples)
        if self.fail:
            raise RuntimeError("model failed")
        return samples

    def flush(self) -> None:
        return None


def pcm(samples: list[int]) -> bytes:
    values = array("h", samples)
    if __import__("sys").byteorder != "little":
        values.byteswap()
    return values.tobytes()


@pytest.mark.asyncio
async def test_disabled_enhancer_is_exact_pass_through() -> None:
    source = pcm([-32768, -1, 0, 1, 32767])
    result, metrics = await DisabledAudioEnhancer().process(source, 16_000, 1)
    assert result is source
    assert metrics.limiter_samples == 0


def test_audio_config_rejects_unsafe_values() -> None:
    for value in (
        {"enhancer": "unknown"},
        {"unknown": True},
        {"output_gain_db": True},
        {"output_gain_db": 12.1},
        {"limiter_dbfs": float("nan")},
        {"inference_deadline_ms": 3},
        {"provider_noise_reduction": "magic"},
    ):
        with pytest.raises(ValueError):
            parse_audio_input_config(value)


def test_audio_config_parses_trusted_gtcrn_policy() -> None:
    config = parse_audio_input_config(
        {
            "enhancer": "gtcrn",
            "output_gain_db": 3,
            "limiter_dbfs": -4,
            "provider_noise_reduction": "far_field",
            "failure_mode": "closed",
            "inference_deadline_ms": 20,
        }
    )
    assert config == AudioInputEnhancementConfig(
        enhancer="gtcrn",
        output_gain_db=3.0,
        limiter_dbfs=-4.0,
        provider_noise_reduction="far_field",
        failure_mode="closed",
        inference_deadline_ms=20,
    )


@pytest.mark.asyncio
async def test_gtcrn_processes_each_complete_hop_and_keeps_frame_size() -> None:
    model = FakeModel()
    enhancer = GtcrnAudioEnhancer(AudioInputEnhancementConfig(enhancer="gtcrn"), model)
    source = pcm([1000] * 1280)
    result, metrics = await enhancer.process(source, 16_000, 1)
    assert len(result) == len(source)
    assert len(model.calls) == 5
    assert all(len(call) == 256 for call in model.calls)
    assert metrics.frames == 1
    await enhancer.close()


@pytest.mark.asyncio
async def test_gtcrn_partial_hop_is_transparent() -> None:
    enhancer = GtcrnAudioEnhancer(
        AudioInputEnhancementConfig(enhancer="gtcrn"), FakeModel()
    )
    source = pcm([1000] * 257)
    assert (await enhancer.process(source, 16_000, 1))[0] == source


@pytest.mark.asyncio
async def test_gtcrn_fail_open_is_one_way() -> None:
    enhancer = GtcrnAudioEnhancer(
        AudioInputEnhancementConfig(enhancer="gtcrn", failure_mode="open"),
        FakeModel(fail=True),
    )
    source = pcm([1000] * 256)
    result, first = await enhancer.process(source, 16_000, 1)
    again, second = await enhancer.process(source, 16_000, 1)
    assert result == source == again
    assert first.failed_open and second.failed_open


@pytest.mark.asyncio
async def test_gtcrn_fail_closed_raises() -> None:
    enhancer = GtcrnAudioEnhancer(
        AudioInputEnhancementConfig(enhancer="gtcrn", failure_mode="closed"),
        FakeModel(fail=True),
    )
    with pytest.raises(AudioEnhancementError, match="model failed"):
        await enhancer.process(pcm([1000] * 256), 16_000, 1)


@pytest.mark.asyncio
async def test_limiter_prevents_pcm_overflow() -> None:
    enhancer = GtcrnAudioEnhancer(
        AudioInputEnhancementConfig(
            enhancer="gtcrn", output_gain_db=12, limiter_dbfs=-3
        ),
        FakeModel(),
    )
    output, metrics = await enhancer.process(pcm([30_000] * 256), 16_000, 1)
    values = array("h")
    values.frombytes(output)
    assert max(abs(value) for value in values) < 32767
    assert metrics.limiter_samples > 0


@pytest.mark.asyncio
async def test_enhancer_serializes_concurrent_calls() -> None:
    model = FakeModel()
    enhancer = GtcrnAudioEnhancer(AudioInputEnhancementConfig(enhancer="gtcrn"), model)
    source = pcm([1000] * 256)
    await asyncio.gather(
        enhancer.process(source, 16_000, 1), enhancer.process(source, 16_000, 1)
    )
    assert len(model.calls) == 2
