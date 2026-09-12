"""Trusted, session-local audio input enhancement.

The optional GTCRN backend is deliberately isolated here so the transport and
provider layers can remain usable without native ML dependencies.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

EXPECTED_GTCRN_SHA256 = (
    "e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534"
)


class AudioEnhancementError(RuntimeError):
    """A configured audio enhancement policy cannot be honored."""


@dataclass(frozen=True, slots=True)
class AudioInputEnhancementConfig:
    """Server-authoritative input enhancement policy for one session."""

    enhancer: Literal["disabled", "gtcrn"] = "disabled"
    output_gain_db: float = 0.0
    limiter_dbfs: float = -3.0
    provider_noise_reduction: Literal["near_field", "far_field", "disabled"] = (
        "near_field"
    )
    failure_mode: Literal["open", "closed"] = "open"
    inference_deadline_ms: int = 12


@dataclass(frozen=True, slots=True)
class EnhancementMetrics:
    """Per-frame non-content measurements."""

    frames: int = 0
    input_rms_dbfs: float = -120.0
    output_rms_dbfs: float = -120.0
    output_peak_dbfs: float = -120.0
    limiter_samples: int = 0
    inference_ms: float = 0.0
    deadline_missed: bool = False
    failed_open: bool = False


class AudioEnhancer(Protocol):
    async def process(
        self, pcm: bytes, sample_rate: int, channels: int
    ) -> tuple[bytes, EnhancementMetrics]: ...

    async def close(self) -> None: ...


class DisabledAudioEnhancer:
    """Exact pass-through used by every profile that does not opt in."""

    async def process(
        self, pcm: bytes, sample_rate: int, channels: int
    ) -> tuple[bytes, EnhancementMetrics]:
        del sample_rate, channels
        return pcm, EnhancementMetrics(
            frames=1,
            input_rms_dbfs=_rms_dbfs(pcm),
            output_rms_dbfs=_rms_dbfs(pcm),
            output_peak_dbfs=_peak_dbfs(pcm),
        )

    async def close(self) -> None:
        return None


class _StreamingModel(Protocol):
    sample_rate: int
    frame_shift_in_samples: int

    def process(self, samples: list[float]) -> list[float]: ...

    def flush(self) -> None: ...


class GtcrnAudioEnhancer:
    """Ordered GTCRN enhancer with a one-way fail-open circuit breaker."""

    def __init__(
        self,
        config: AudioInputEnhancementConfig,
        model: _StreamingModel,
    ) -> None:
        self._config = config
        self._model = model
        self._failed_open = False
        self._closed = False
        self._initial_delay_samples = 0
        self._lock = asyncio.Lock()

    async def process(
        self, pcm: bytes, sample_rate: int, channels: int
    ) -> tuple[bytes, EnhancementMetrics]:
        if self._closed:
            raise AudioEnhancementError("audio enhancer is closed")
        if sample_rate != self._model.sample_rate or channels != 1 or len(pcm) % 2:
            return await self._failure_or_raise(
                pcm,
                "GTCRN requires even-length 16 kHz mono PCM input",
            )
        # The transport normally supplies 80 ms (five 16 ms hops). Refuse to
        # invent timing for partial model hops: passing that frame through is
        # safer than dropping, duplicating, or delaying user speech.
        samples = _pcm_to_floats(pcm)
        hop = self._model.frame_shift_in_samples
        if len(samples) % hop:
            return pcm, EnhancementMetrics(
                frames=1,
                input_rms_dbfs=_rms_dbfs(pcm),
                output_rms_dbfs=_rms_dbfs(pcm),
                output_peak_dbfs=_peak_dbfs(pcm),
            )
        async with self._lock:
            if self._failed_open:
                return pcm, EnhancementMetrics(
                    frames=1,
                    input_rms_dbfs=_rms_dbfs(pcm),
                    output_rms_dbfs=_rms_dbfs(pcm),
                    output_peak_dbfs=_peak_dbfs(pcm),
                    failed_open=True,
                )
            started = time.perf_counter()
            try:
                enhanced = await asyncio.to_thread(self._process_sync, samples)
            except Exception as err:  # native/model failures are policy controlled
                return await self._failure_or_raise(pcm, str(err))
            elapsed_ms = (time.perf_counter() - started) * 1000
            # GTCRN emits its first enhanced hop on the following call. Keep
            # transport frame length invariant by introducing that documented
            # one-hop (16 ms) delay with leading silence exactly once.
            if len(enhanced) < len(samples):
                missing = len(samples) - len(enhanced)
                if self._initial_delay_samples:
                    return await self._failure_or_raise(
                        pcm, "GTCRN unexpectedly withheld output"
                    )
                enhanced = [0.0] * missing + enhanced
                self._initial_delay_samples = missing
            elif len(enhanced) > len(samples):
                return await self._failure_or_raise(
                    pcm, "GTCRN changed PCM frame length"
                )
            output, limited = _floats_to_pcm(
                enhanced,
                gain_db=self._config.output_gain_db,
                limiter_dbfs=self._config.limiter_dbfs,
            )
            return output, EnhancementMetrics(
                frames=1,
                input_rms_dbfs=_rms_dbfs(pcm),
                output_rms_dbfs=_rms_dbfs(output),
                output_peak_dbfs=_peak_dbfs(output),
                limiter_samples=limited,
                inference_ms=elapsed_ms,
                deadline_missed=elapsed_ms > self._config.inference_deadline_ms,
            )

    def _process_sync(self, samples: list[float]) -> list[float]:
        hop = self._model.frame_shift_in_samples
        output: list[float] = []
        for offset in range(0, len(samples), hop):
            chunk = self._model.process(samples[offset : offset + hop])
            if len(chunk) not in {0, hop}:
                raise AudioEnhancementError("GTCRN returned an invalid hop length")
            output.extend(chunk)
        return output

    async def _failure_or_raise(
        self, pcm: bytes, message: str
    ) -> tuple[bytes, EnhancementMetrics]:
        if self._config.failure_mode == "closed":
            raise AudioEnhancementError(message)
        self._failed_open = True
        return pcm, EnhancementMetrics(
            frames=1,
            input_rms_dbfs=_rms_dbfs(pcm),
            output_rms_dbfs=_rms_dbfs(pcm),
            output_peak_dbfs=_peak_dbfs(pcm),
            failed_open=True,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(self._model.flush)


class _SherpaGtcrnModel:
    """Small adapter over sherpa-onnx's stateful online denoiser."""

    def __init__(self, path: Path) -> None:
        try:
            import sherpa_onnx
        except ImportError as err:  # pragma: no cover - exercised in container test
            raise AudioEnhancementError(
                "sherpa-onnx enhancement dependency is unavailable"
            ) from err
        config = sherpa_onnx.OnlineSpeechDenoiserConfig(
            model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(
                    model=str(path)
                ),
                num_threads=1,
                provider="cpu",
                debug=False,
            )
        )
        if not config.validate():
            raise AudioEnhancementError("GTCRN configuration validation failed")
        self._denoiser = sherpa_onnx.OnlineSpeechDenoiser(config)
        self.sample_rate = int(self._denoiser.sample_rate)
        self.frame_shift_in_samples = int(self._denoiser.frame_shift_in_samples)

    def process(self, samples: list[float]) -> list[float]:
        result = self._denoiser(samples, self.sample_rate)
        return list(result.samples)

    def flush(self) -> None:
        self._denoiser.flush()


def verify_gtcrn_model(path: str) -> Path:
    """Verify the pinned immutable artifact before native model construction."""
    model = Path(path)
    try:
        digest = hashlib.file_digest(model.open("rb"), "sha256").hexdigest()
    except OSError as err:
        raise AudioEnhancementError("GTCRN model cannot be read") from err
    if digest != EXPECTED_GTCRN_SHA256:
        raise AudioEnhancementError(
            "GTCRN model checksum does not match pinned artifact"
        )
    return model


def create_audio_enhancer(
    config: AudioInputEnhancementConfig, *, gtcrn_model_path: str = ""
) -> AudioEnhancer:
    """Create isolated enhancement state after trusted configuration selection."""
    if config.enhancer == "disabled":
        return DisabledAudioEnhancer()
    if config.enhancer != "gtcrn":  # defensive boundary for future config changes
        raise AudioEnhancementError("unsupported audio enhancer")
    if not gtcrn_model_path:
        raise AudioEnhancementError("GTCRN model path is not configured")
    return GtcrnAudioEnhancer(
        config, _SherpaGtcrnModel(verify_gtcrn_model(gtcrn_model_path))
    )


def parse_audio_input_config(value: object) -> AudioInputEnhancementConfig:
    """Strictly parse a trusted JSON profile's optional audio policy."""
    if value is None:
        return AudioInputEnhancementConfig()
    if not isinstance(value, dict):
        raise ValueError("audio_input must be an object")
    allowed = {
        "enhancer",
        "output_gain_db",
        "limiter_dbfs",
        "provider_noise_reduction",
        "failure_mode",
        "inference_deadline_ms",
    }
    if set(value) - allowed:
        raise ValueError("audio_input contains unknown keys")
    enhancer = value.get("enhancer", "disabled")
    noise = value.get("provider_noise_reduction", "near_field")
    failure = value.get("failure_mode", "open")
    if enhancer not in {"disabled", "gtcrn"}:
        raise ValueError("audio_input enhancer is invalid")
    if noise not in {"near_field", "far_field", "disabled"}:
        raise ValueError("audio_input provider_noise_reduction is invalid")
    if failure not in {"open", "closed"}:
        raise ValueError("audio_input failure_mode is invalid")
    gain = _number(value.get("output_gain_db", 0.0), "output_gain_db", 0.0, 12.0)
    limiter = _number(value.get("limiter_dbfs", -3.0), "limiter_dbfs", -12.0, -1.0)
    deadline = value.get("inference_deadline_ms", 12)
    if (
        isinstance(deadline, bool)
        or not isinstance(deadline, int)
        or not 4 <= deadline <= 40
    ):
        raise ValueError("audio_input inference_deadline_ms is invalid")
    return AudioInputEnhancementConfig(
        enhancer, gain, limiter, noise, failure, deadline
    )


def _number(value: object, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"audio_input {name} is invalid")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"audio_input {name} is invalid")
    return result


def _pcm_to_floats(pcm: bytes) -> list[float]:
    import array

    values = array.array("h")
    values.frombytes(pcm)
    if values.itemsize != 2:
        raise AudioEnhancementError("platform PCM word size is unsupported")
    if __import__("sys").byteorder != "little":
        values.byteswap()
    return [sample / 32768.0 for sample in values]


def _floats_to_pcm(
    samples: list[float], *, gain_db: float, limiter_dbfs: float
) -> tuple[bytes, int]:
    import array

    limit = 10 ** (limiter_dbfs / 20)
    gain = 10 ** (gain_db / 20)
    output = array.array("h")
    limited = 0
    for sample in samples:
        value = sample * gain
        if value > limit:
            value = limit
            limited += 1
        elif value < -limit:
            value = -limit
            limited += 1
        output.append(max(-32768, min(32767, round(value * 32767))))
    if __import__("sys").byteorder != "little":
        output.byteswap()
    return output.tobytes(), limited


def _rms_dbfs(pcm: bytes) -> float:
    samples = _pcm_to_floats(pcm)
    if not samples:
        return -120.0
    return 20 * math.log10(
        max(math.sqrt(sum(x * x for x in samples) / len(samples)), 1e-6)
    )


def _peak_dbfs(pcm: bytes) -> float:
    samples = _pcm_to_floats(pcm)
    if not samples:
        return -120.0
    return 20 * math.log10(max(max(abs(x) for x in samples), 1e-6))
