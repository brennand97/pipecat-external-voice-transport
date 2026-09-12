# Profile-scoped ML audio enhancement — low-level implementation plan

Status: design only; implementation has not started.

Audience: an implementation agent with no prior conversation context.

## 1. Goal

Add optional server-side, profile-scoped speech enhancement for quiet/far-field 16 kHz mono Satellite PCM. Use a maintained open-source streaming model instead of implementing a denoiser or adaptive “focus” algorithm in this repository.

The implementation must:

- be server-authoritative;
- add tightly bounded latency;
- fit Loki's one-CPU/1 GiB container limit;
- preserve frame order and continuous timing;
- be exactly transparent when disabled;
- expose objective quality/performance metrics;
- fail explicitly or fail open according to trusted policy;
- remain independent from Satellite media ducking so each feature can be tested alone.

## 2. Baseline evidence

Reviewed raw inputs are signed 16-bit, 16 kHz, mono PCM:

| Capture | Duration | Overall RMS | 90th-percentile 20 ms RMS | Windows above -35 dBFS |
|---|---:|---:|---:|---:|
| `2026-09-12-7b8a…` | 63.44 s | -26.3 dBFS | -20.8 dBFS | 36.7% |
| `2026-09-08-42ef…` | 81.20 s | -30.5 dBFS | -27.4 dBFS | 19.3% |
| `2026-09-08-e755…` | 21.28 s | -26.2 dBFS | -21.4 dBFS | 23.8% |
| `2026-09-08-2522…` | 31.92 s | -27.9 dBFS | -22.1 dBFS | 23.7% |

None clipped. The latest session combines recognition errors with persistent program audio, so gain without enhancement is unsafe.

## 3. Model selection

### 3.1 Primary implementation: sherpa-onnx streaming GTCRN

Use `k2-fsa/sherpa-onnx`'s `OnlineSpeechDenoiser` with `gtcrn_simple.onnx`.

Verified properties at design time:

- sherpa-onnx: Apache-2.0.
- GTCRN reference/model: MIT.
- model URL: GitHub release `k2-fsa/sherpa-onnx`, tag `speech-enhancement-models`, asset `gtcrn_simple.onnx`.
- SHA-256: `e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534`.
- asset size: approximately 535 KB.
- sample rate: 16 kHz.
- hop length: 256 samples, or 16 ms.
- streaming recurrent state and learned per-frame spectral mask.
- CPU provider and one-thread mode.
- sherpa-onnx 1.13.8 publishes CPython 3.12, 3.13, and 3.14 Linux x86-64 wheels, matching development and container Python versions.

The model already adapts every 16 ms. Do not create a home-grown controller that changes model hyperparameters per frame; that is unstable, unvalidated, and duplicates the recurrent mask estimator.

### 3.2 What GTCRN does not do

GTCRN enhances generic speech. It does not identify a household member. Music vocals, television dialogue, and another speaker may survive because they are speech-like.

A moving target requires either:

- synchronized multichannel audio plus beamforming/array geometry; or
- an enrolled speaker embedding plus a target-speaker extraction model.

A mono stream cannot infer direction. Target-speaker toolkits reviewed (including WeSep and ClearerVoice-Studio) are research/broad toolkit stacks rather than drop-in low-latency preprocessors; target enrollment also introduces biometric privacy, guest failure behavior, and additional compute. WeSep did not expose a clear repository license through GitHub at review time. Do not add target-speaker extraction in this implementation.

Name the feature **ML speech enhancement**, not voice isolation or acoustic echo cancellation.

### 3.3 Alternatives retained for benchmark only

- RNNoise: BSD-3-Clause, 10 ms frames, but standard 48 kHz API requires two resampling stages.
- DTLN: MIT, 16 kHz, 8 ms shift, <1 million parameters, demonstrated on Raspberry Pi 3 B+, but older split-state ONNX integration.
- DeepFilterNet: MIT/Apache-2.0, high-quality low-latency model, but designed for 48 kHz.
- sherpa-onnx DPDFNet: Apache-2.0 implementation and 16 kHz streaming models, but 9–15 MB model assets.

Do not implement multiple backends before GTCRN's acceptance test. The offline harness may gain adapters later.

## 4. Trusted configuration schema

The existing mounted `voice-tools.json` already defines the trusted named profiles selected by `conversation.profile`. Extend those profile records rather than accepting filter settings over the WebSocket.

Example:

```json
{
  "profiles": {
    "home": {
      "providers": ["home-assistant", "music-assistant"],
      "allowed_tools": ["..."],
      "audio_input": {
        "enhancer": "disabled",
        "provider_noise_reduction": "near_field"
      }
    },
    "home-far-field": {
      "providers": ["home-assistant", "music-assistant"],
      "allowed_tools": ["..."],
      "audio_input": {
        "enhancer": "gtcrn",
        "output_gain_db": 3.0,
        "limiter_dbfs": -3.0,
        "provider_noise_reduction": "far_field",
        "failure_mode": "open"
      }
    }
  }
}
```

Allowed fields and bounds:

| Field | Default | Values/bounds |
|---|---:|---|
| `enhancer` | `disabled` | `disabled`, `gtcrn` |
| `output_gain_db` | 0.0 | 0.0–12.0 |
| `limiter_dbfs` | -3.0 | -12.0 to -1.0 |
| `provider_noise_reduction` | `near_field` | `near_field`, `far_field`, `disabled` if supported by provider API |
| `failure_mode` | `open` | `open`, `closed` |
| `inference_deadline_ms` | 12 | 4–40 |

Reject booleans where numeric values are expected. Reject NaN/infinity. Unknown keys fail startup. A profile that requests `gtcrn` fails application readiness if the model cannot be loaded or verified; do not silently convert the configured profile to disabled.

The client continues to send only a profile name. It never sends model paths, gain, limiter, or failure policy.

## 5. Code structure

### 5.1 New module

Create `src/voice_transport/audio_enhancement.py` with no imports from OpenAI-specific modules.

Public types:

```python
@dataclass(frozen=True, slots=True)
class AudioInputEnhancementConfig:
    enhancer: Literal["disabled", "gtcrn"] = "disabled"
    output_gain_db: float = 0.0
    limiter_dbfs: float = -3.0
    provider_noise_reduction: Literal["near_field", "far_field", "disabled"] = "near_field"
    failure_mode: Literal["open", "closed"] = "open"
    inference_deadline_ms: int = 12

@dataclass(frozen=True, slots=True)
class EnhancementMetrics:
    frames: int
    input_rms_dbfs: float
    output_rms_dbfs: float
    output_peak_dbfs: float
    limiter_samples: int
    inference_ms: float
    failed_open: bool

class AudioEnhancer(Protocol):
    async def process(self, pcm: bytes, sample_rate: int, channels: int) -> tuple[bytes, EnhancementMetrics]
    async def close(self) -> None
```

Implement:

- `DisabledAudioEnhancer`: returns the exact same `bytes` object/value and zero processing metrics.
- `GtcrnAudioEnhancer`: owns one sherpa-onnx online denoiser stream and one ordered execution worker.
- `AudioEnhancerFactory`: validates the model once and creates session-local enhancer state.

Keep model inference behind a small adapter protocol so unit tests use a fake model without importing/loading sherpa-onnx.

### 5.2 Profile parser refactor

Current profile parsing in `src/voice_transport/tools/config.py` returns positional tuples. Replace the tuple with a named immutable dataclass, for example:

```python
@dataclass(frozen=True, slots=True)
class TrustedProfileConfig:
    providers: frozenset[str]
    allowed_patterns: tuple[ToolNamePattern, ...]
    context_injections: dict[str, dict[str, str]]
    audio_input: AudioInputEnhancementConfig
```

Tasks:

1. Add `audio_input` to the profile allowed-key set.
2. Parse it through one strict `_audio_input_config` function in `audio_enhancement.py` or a dependency-neutral config module.
3. Add `audio_input_config` to `ToolRegistry`, or return a small `PreparedSessionProfile` containing registry plus audio config. Prefer the latter if it avoids making tool behavior own audio behavior.
4. Ensure legacy synthesized profiles receive disabled enhancement and near-field provider noise reduction.
5. Ensure calculator-only/no-provider profiles still work.
6. Parse the trusted JSON once per session until a separate cached immutable config loader is introduced.

Do not overload `requested_tools`; audio policy is never client-filterable.

### 5.3 Settings and model path

Add to `src/voice_transport/config.py`:

```python
gtcrn_model_path: str = ""
gtcrn_model_sha256: str = EXPECTED_GTCRN_SHA256
```

Environment variables:

- `GTCRN_MODEL_PATH`, default `/app/models/gtcrn_simple.onnx` in the container.
- Do not make the expected checksum operator-configurable in normal production; pin it in source. If an override is required for development, name it explicitly as unsafe/test-only and reject it outside tests.

Startup/readiness behavior:

- If no trusted profile requests GTCRN, absence of the model and optional library does not break fake/minimal installations.
- If any configured profile requests GTCRN, verify file type/size/checksum and instantiate a probe denoiser during application startup before `/ready` returns success.
- Never download model files at runtime.

### 5.4 Dependency and model vendoring

In `pyproject.toml` add:

```toml
enhancement = [
  "sherpa-onnx==1.13.8",
]
```

Before merging, verify the exact package's transitive footprint and licenses in CI.

Preferred model layout:

```text
models/gtcrn_simple.onnx
models/README.md
THIRD_PARTY_NOTICES.md
```

Commit the small immutable model artifact or fetch it in a dedicated build stage from the exact release URL and verify SHA-256 before copying it into the runtime image. Committing is preferred for reproducible/offline image builds, provided repository policy accepts third-party binaries. Include upstream URL, tag, checksum, and MIT notice.

Update `Dockerfile` to install `.[realtime,tools,enhancement]` and copy only the model/notice files. Preserve read-only runtime, non-root user, dropped capabilities, one CPU, and 1 GiB limits.

Add container tests proving the model exists, checksum matches, module imports under Python 3.14, and `/ready` succeeds for a GTCRN profile.

## 6. Runtime pipeline integration

### 6.1 Provider-neutral config

Add `audio_input: AudioInputEnhancementConfig` to `RealtimeProviderConfig` in `src/voice_transport/providers/base.py`.

In `src/voice_transport/providers/__init__.py`:

1. Prepare the trusted profile once.
2. Pass its tool registry and audio config into `RealtimeProviderConfig`.
3. Do not infer audio config from Satellite entity, prompt, or model output.

Expose a redacted effective configuration in `session.ready`, for example:

```json
"audio_input": {
  "enhancer": "gtcrn",
  "provider_noise_reduction": "far_field"
}
```

Do not expose model filesystem paths.

### 6.2 Pipecat processor placement

In `src/voice_transport/providers/openai_realtime.py`, insert a processor immediately after `_PipecatPCMSource` and before `user_aggregator`:

```text
_PipecatPCMSource
  -> _PipecatAudioEnhancementProcessor (only when enabled)
  -> user_aggregator
  -> OpenAIRealtimeLLMService
  -> sink
  -> assistant_aggregator
```

When disabled, omit the processor entirely. This gives the strongest byte-for-byte transparency guarantee.

`_PipecatAudioEnhancementProcessor` behavior:

- Match only `InputAudioRawFrame`; forward every other frame unchanged.
- Validate 16 kHz mono signed-16 input. A configured unsupported format follows `failure_mode` rather than silently resampling.
- Await ordered processing, construct one replacement `InputAudioRawFrame` with the same sample rate/channels and processed bytes, then push it in the original direction.
- Maintain no model state in class/static globals.
- `cleanup()`/terminal handling closes the enhancer and worker exactly once.
- Never flush model-tail audio into a closed/cancelled provider session.

### 6.3 Chunking and timing

Incoming native chunks are normally 80 ms (1280 samples). GTCRN uses 256-sample/16 ms hops, so each input frame contains exactly five hops.

The adapter must still support arbitrary even-length chunks permitted by protocol:

1. Append samples to a session-local remainder buffer.
2. Process each complete 256-sample hop.
3. Preserve any remainder for the next frame.
4. Because withholding a partial hop changes frame length/timing, return processed complete samples plus a conservative passthrough/delayed strategy defined by tests. Preferred strategy is a fixed 256-sample streaming delay: emit exactly as many samples as received using initial zeros and queued processed output.
5. Document the exact delay and compensate only in metrics, not by dropping samples.

For normal 80 ms input, output frame byte length must equal input frame byte length. Across the whole session, total output samples must equal total input samples until cancellation. Do not append flush output to OpenAI after session close.

### 6.4 Inference execution

Do not block the asyncio event loop with native inference unless measurement proves the binding releases the GIL and remains below 1 ms.

Use one bounded, ordered worker per active enhanced session:

- a dedicated single-worker executor or an asyncio worker task consuming a queue of size 1;
- processing backpressure propagates to the existing bounded PCM/WebSocket path;
- no unbounded inference queue;
- max two sessions follows current deployment concurrency.

Measure wall and CPU time. A deadline miss is diagnostic; do not cancel native inference mid-call.

Failure policy:

- `open`: on first inference exception or repeated deadline breach, emit `audio.enhancement_failed_open`, close/bypass model for the rest of that session, and forward the original current/subsequent PCM. Preserve ordering and sample count.
- `closed`: emit provider/session error and terminate cleanly.
- Never alternate processed/raw audio repeatedly after failure; use a one-way circuit breaker.

### 6.5 Gain and limiter

The ML library owns denoising. Post-gain is intentionally simple:

1. Convert output to contiguous float32/array representation already returned by sherpa-onnx.
2. Multiply by `10 ** (output_gain_db / 20)`.
3. Apply a bounded limiter before signed-16 conversion; no integer wrap is permitted.
4. Count limited samples.

Do not add a custom adaptive AGC/noise tracker in version one. Kiosk Satellite already has device-level gain controls, and another dynamic gain loop could pump music/noise. A later maintained AGC library requires separate A/B evidence.

### 6.6 OpenAI noise reduction

`_session_properties()` currently hard-codes `InputAudioNoiseReduction(type="near_field")`.

Map trusted config:

- `near_field` -> `InputAudioNoiseReduction(type="near_field")`;
- `far_field` -> `InputAudioNoiseReduction(type="far_field")`;
- `disabled` -> omit `noise_reduction` if the installed OpenAI/Pipecat schema supports omission.

Add a compatibility unit test against the pinned Pipecat version. Do not send an undocumented string.

Double denoising may damage consonants. Test GTCRN with provider reduction disabled, near-field, and far-field independently.

## 7. Audit and developer portal

### 7.1 Metadata events

Use `SessionAuditLog.record()` for non-content events:

- `audio.enhancement_started`
- `audio.enhancement_metrics` every five seconds and at terminal cleanup
- `audio.enhancement_failed_open`
- `audio.enhancement_failed`

Fields:

```text
enhancer, model_version, model_sha256_prefix,
provider_noise_reduction, frame_count,
input_rms_dbfs, output_rms_dbfs, output_peak_dbfs,
limiter_samples, inference_ms_p50/p95/p99,
realtime_factor, deadline_misses, status
```

No PCM, transcript, media URL, full local path, or credential in metadata mode.

### 7.2 Processed debug audio

Extend `SessionAuditLog.record_audio` direction typing/naming to support `input_processed` only in `debug_content` mode. Keep existing `input` as raw transport PCM and `output` as assistant PCM.

Update prune patterns and `src/voice_transport/dev_portal.py` so raw and processed clips are separately labeled and rendered as contiguous WAV timelines. Do not enable processed recording when raw debug recording is disabled. Apply the same per-session byte cap independently or explicitly divide one documented cap between raw and processed data.

Tests must prove processed PCM cannot be written in `off` or `metadata` mode.

## 8. Automated tests

### 8.1 Configuration tests

Extend `tests/unit/test_tool_config.py` or create `tests/unit/test_audio_enhancement_config.py`:

- defaults/legacy profiles are disabled and near-field;
- valid GTCRN profile parses;
- unknown enhancer/key/provider reduction/failure mode rejected;
- every numeric bound tested on both sides;
- booleans, NaN, infinity, strings-as-numbers rejected;
- calculator-only profile remains valid;
- client requested tools cannot affect audio policy.

### 8.2 Enhancer unit tests

Create `tests/unit/test_audio_enhancement.py` with a fake streaming model adapter:

1. Disabled enhancer returns exact original bytes.
2. Five 256-sample hops are processed for one 80 ms frame.
3. Fragmented/non-80 ms frames preserve order and total sample count.
4. State is isolated between sessions.
5. Gain conversion is correct within one PCM quantization step.
6. Positive/negative full-scale transients never wrap or exceed limiter.
7. No NaN/inf reaches PCM output.
8. Model exception switches fail-open permanently and forwards original audio.
9. Fail-closed raises the expected typed error.
10. Close is idempotent and no frames emit afterward.
11. Queue is bounded and backpressure is observable.
12. Deadline misses increment metrics without reordering.

### 8.3 Pipecat integration tests

Create `tests/unit/test_openai_audio_enhancement.py`:

- processor appears directly after source only for enabled profiles;
- only input audio frames are transformed;
- control/interruption/end frames pass unchanged;
- processed frame metadata is retained;
- cleanup closes enhancer after normal, cancel, timeout, and provider-error paths;
- OpenAI session properties map near/far/disabled correctly.

Extend `tests/integration/test_transport.py` with fake provider/enhancer injection to prove real WebSocket PCM reaches the provider transformed once and disabled PCM arrives unchanged.

### 8.4 Real-model tests

Mark real GTCRN tests separately, but run them in CI/container validation:

- model checksum and load;
- 16 kHz sample rate and 256-sample shift;
- deterministic output for a fixed fixture within documented tolerance;
- silence does not become material noise;
- 60 seconds of PCM processes faster than realtime;
- memory remains bounded over repeated session creation/destruction.

Do not make billable OpenAI tests mandatory CI.

### 8.5 Full validation

```bash
cd /home/brennan/repos/pipecat-external-voice-transport
.venv/bin/ruff check .
.venv/bin/python -m pytest -q
podman build -t voice-transport:gtcrn-test .   # docker is also acceptable
```

Run the built container read-only with the same one-CPU/1 GiB constraints as production.

## 9. Local offline measurement harness

### 9.1 Deliverable

Add `scripts/audio_enhancement_harness.py` plus `scripts/README-audio-harness.md`.

Subcommands:

```text
benchmark --input <pcm-or-wav> --profile <profile> --repeat 10 --json <path>
enhance   --input <pcm-or-wav> --profile <profile> --output <wav> --json <path>
compare   --manifest <json> --profiles raw,gtcrn-far --output-dir <dir>
live      --manifest <json> --profiles raw,gtcrn-far --output-dir <dir>
```

The harness imports the production parser/enhancer; it must not duplicate signal processing.

### 9.2 Fixture manifest

Do not commit private debug audio by default. Add the containing directory to `.gitignore` and accept `VOICE_AUDIO_FIXTURE_DIR`.

Manifest example:

```json
{
  "cases": [
    {
      "id": "far-field-play-request",
      "input": "far-field-play-request.pcm",
      "sample_rate": 16000,
      "channels": 1,
      "expected_transcript": "play Sabrina Carpenter in the kitchen",
      "kind": "speech"
    },
    {
      "id": "music-only-tail",
      "input": "music-only-tail.pcm",
      "sample_rate": 16000,
      "channels": 1,
      "expected_transcript": "",
      "kind": "non_speech"
    }
  ]
}
```

Private manifests/results remain outside git. Repository fixtures must be synthetic or explicitly licensed/consented.

### 9.3 Offline metrics

For every profile/case, output machine-readable JSON with:

- samples in/out and algorithmic delay;
- input/output RMS and peak;
- limiter count;
- inference p50/p95/p99 and max;
- wall/CPU realtime factor;
- model/library versions/checksum;
- peak RSS where available.

If a clean reference is supplied, calculate SI-SDR improvement using a tested library or a clearly isolated metrics-only implementation. Do not use PESQ without confirming its licensing. Objective signal scores do not replace transcription tests.

The `enhance` command writes WAV only under the requested output directory and never overwrites without `--force`.

## 10. Live OpenAI replay harness on the dev box

### 10.1 Safety and credential handling

The live subcommand is opt-in and billable:

- require `RUN_LIVE_AUDIO_ENHANCEMENT_TEST=1`;
- read `OPENAI_API_KEY` from environment only;
- never accept/print the key;
- load credentials according to the machine's non-secret credential inventory;
- use a calculator-only/no-external-provider tool profile so replay cannot control HA/MA;
- generate a random transport bearer token in memory;
- default to loopback and audio output disabled/text output where supported;
- cap each case duration and total run cost/time.

### 10.2 Execution

The harness starts the real FastAPI app with uvicorn on an ephemeral `127.0.0.1` port, mirroring existing live integration tests. For each case/profile:

1. Open a fresh transport session.
2. Send `session.start` with audio input and the trusted profile name.
3. Wait for `session.ready` and assert returned effective audio policy.
4. Send `turn.start`.
5. Replay PCM in original 80 ms cadence using monotonic scheduling. Do not burst-send; provider VAD behavior depends on realtime timing.
6. Continue any non-speech/music tail while assistant response begins when the fixture scenario requires false-barge testing.
7. Collect `user.speech_started`, final transcripts, response start/finish, interruption, error, and enhancement audit metrics.
8. Cancel and verify `session.finished` plus enhancer cleanup.
9. Repeat raw and enhanced profiles with fresh sessions.

Compute normalized transcript text and word error rate using a pinned dev-only library such as `jiwer`, or a small separately tested metrics helper. Report blank transcripts explicitly; never coerce them to success.

### 10.3 End-to-end route from HA/Satellite to `kratos`

After direct replay passes, run a candidate server reachable by HA:

```bash
EXTERNAL_TRANSPORT_TOKEN='<injected>' \
REALTIME_PROVIDER=openai_realtime \
OPENAI_API_KEY='<injected>' \
TRUSTED_TOOL_CONFIG_PATH=/tmp/voice-tools-audio-test.json \
GTCRN_MODEL_PATH="$PWD/models/gtcrn_simple.onnx" \
SESSION_AUDIT_MODE=debug_content \
SESSION_AUDIT_LOG_PATH=/tmp/voice-transport-audio-test \
.venv/bin/uvicorn voice_transport.app:runtime_app --factory \
  --host 0.0.0.0 --port 8765 --no-access-log
```

Do not put literal secrets in shell history; the shown placeholders mean environment injection by the operator. Restrict port 8765 to the trusted LAN and stop it after testing.

In HA:

1. Create a temporary External Conversation Service URL `ws://10.1.0.40:8765/transport/v1` (or current dev-box LAN IP).
2. Add raw and GTCRN test profiles with no control tools.
3. Assign only a test Satellite.
4. Run matched utterances at fixed marked distances (0.5 m, 2 m, 4 m) in quiet and with the configured MA player ducked to known levels.
5. Restore original Satellite assignment and remove temporary service afterward.

This validates native Kiosk/browser PCM, HA relay, local enhancement, OpenAI VAD/transcription, and response timing without deploying the candidate image to Loki.

## 11. Measurement protocol and acceptance gates

Use at least:

- 10 quiet near-field utterances;
- 10 quiet far-field utterances at each chosen distance;
- 10 far-field utterances with music at temporary volume;
- 10 music/background-only tails during assistant output;
- three speakers if available, including one not used during tuning.

Keep wording and playback source/volume fixed across raw/enhanced runs. Alternate profile order to reduce learning/order bias.

Release gates:

### Performance

- p99 inference <8 ms per 16 ms GTCRN hop on Loki-equivalent one CPU.
- mean realtime factor <0.25 for one session and <0.50 with two concurrent sessions.
- p95 added end-to-end transcript latency ≤25 ms relative to raw.
- peak container RSS <750 MiB with two enhanced sessions under the 1 GiB limit.
- no event-loop heartbeat delay >20 ms attributable to inference.

### Correctness

- disabled path byte-for-byte identical.
- zero sample loss/duplication/reordering in all fixture chunkings.
- zero integer wrap/clipping beyond configured limiter.
- all terminal paths close model workers and leave no growing thread/task count.

### Quality

- ≥20% relative WER improvement on the far-field set, or a pre-agreed statistically meaningful equivalent if baseline WER is already low.
- no more than 5% relative WER degradation on clean near-field speech.
- ≥50% reduction in false `user.speech_started`/assistant interruption events on non-speech/music-tail fixtures.
- no increase in missed speech turns greater than one case across the evaluation corpus.
- GTCRN+provider denoising must beat GTCRN alone before double denoising is enabled.

If performance passes but quality does not, leave production profiles disabled and retain harness results. Do not tune against only the five reviewed sessions.

## 12. Rollout and rollback

1. Merge code with all production profiles set to `enhancer: disabled`.
2. Build/publish an immutable candidate image.
3. Test locally on `kratos` through direct replay and HA-routed live sessions.
4. Deploy to Loki with enhancement still disabled; verify readiness/resource baseline.
5. Enable GTCRN on one test profile/Satellite only.
6. Observe at least 20 sessions and compare audit metrics against raw controls.
7. Expand only after acceptance gates remain satisfied.

Rollback is configuration-only: select the raw profile or set `enhancer: disabled`, then restart/recreate sessions. Keep the previous immutable image available. A model/runtime failure must never require changing Satellite firmware.

## 13. Implementation sequence for independent agents

Use fail-first tests and one concern per commit:

1. Add strict audio config dataclass/parser and tests.
2. Add model artifact/notices/checksum/container validation.
3. Add model adapter and disabled/fake enhancer tests.
4. Implement GTCRN chunk/state adapter and real-model tests.
5. Add bounded ordered execution and failure circuit breaker.
6. Add Pipecat processor placement and cleanup tests.
7. Add OpenAI near/far/disabled mapping tests.
8. Add audit metrics and debug processed-audio tests.
9. Add offline benchmark/compare harness with redaction tests.
10. Add opt-in live replay harness/test.
11. Run full validation and container benchmark.
12. Run HA-to-`kratos` live protocol, attach redacted measurements to the PR, and release only after gates pass.

Do not implement Satellite media ducking in this repository or in the same PR.
