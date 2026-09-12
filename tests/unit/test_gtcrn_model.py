from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sherpa_onnx")

from voice_transport.audio_enhancement import (
    AudioInputEnhancementConfig,
    create_audio_enhancer,
    verify_gtcrn_model,
)


@pytest.mark.asyncio
async def test_pinned_gtcrn_model_loads_and_preserves_normal_transport_frame() -> None:
    model = Path(__file__).parents[2] / "models" / "gtcrn_simple.onnx"
    assert verify_gtcrn_model(str(model)) == model
    enhancer = create_audio_enhancer(
        AudioInputEnhancementConfig(enhancer="gtcrn"), gtcrn_model_path=str(model)
    )
    output, metrics = await enhancer.process(b"\x00\x00" * 1280, 16_000, 1)
    assert len(output) == 2560
    assert metrics.failed_open is False
    await enhancer.close()
