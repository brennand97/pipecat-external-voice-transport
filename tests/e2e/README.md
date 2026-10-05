# Deterministic E2E provider adapter

This directory owns transport-specific test behavior. The reusable HA/browser harness lives in the separate `ha-integration-testbed` checkout; HA satellite/browser fixtures live in `voice-satellite-card-integration/tests/e2e`.

`Dockerfile` builds a small runtime **without Pipecat/OpenAI/ML dependencies**. Its Dockerfile-specific ignore file includes only source and this test directory, leaving the production Docker context policy unchanged. The production Dockerfile/entrypoint and fake provider source are not modified.

`scenario_app.py` injects `ScenarioAgent` at the existing fake provider factory **inside the E2E process only**. Authentication, WebSockets, conversation actor, bounds, signed audio storage/streaming and tool registry use the real implementations. The app refuses a real provider/API key. It emits deterministic text, PCM tone chunks, cancellable delayed responses and a scripted synthetic-light tool request. `/e2e/stats` requires the same test bearer token and reports counts, never credentials/audio/transcripts.

```sh
# Existing development environment; no Docker/HA needed for provider tests:
PYTHONPATH=src:tests/e2e .venv/bin/python -m unittest discover -s tests/e2e -p 'test_*.py' -v

# Real-container acceptance (the satellite wrapper sets up HA/MCP/transport):
# In the satellite repo:
npm run e2e:verify -- <run-id>
```

`smoke.py` uses the shared testbed Python environment and private run manifest/credentials. It checks invalid-token rejection, persistent text/audio turns, input PCM, valid signed WAV streaming and session cleanup. With the configured test HA MCP fixture, it also verifies `intent__HassTurnOn` through the real registry changes only the exposed synthetic light. No cloud/provider credential or production HA token is loaded.

Slow-response cancellation is independently tested at the provider boundary. This is not a speech recognition or physical audio-quality test. Browser consumption of the provider's signed audio URL is asserted by the owning satellite suite, not substituted by this direct backend test. The current external path does not wrap these URLs in HA's separate media-player proxy.

## Credentialed and acoustic probes (separate acceptance gates)

- `live_openai.py <private-run.json>` runs the production Pipecat/OpenAI stack with real scratch MCP: creation, live remaining-time lookup, relative extension, ambiguous clarification without mutation, named cancellation and agent-ended conversation preserving another timer. It uses the inventoried `.openai-test-key` in-process only. This is billable, bounded and opt-in; deterministic tests never load that key.
- `acoustics.py <private-directory>` creates cached public TTS phrases and deterministic distance, SNR, clipping, tablet bandwidth/quantization, reverberation, competing-voice and negative clips. `--continuous --source-directory <cached-directory>` reuses speech without another TTS request and trims padding for a 120ms wake/command gap. These are synthetic stress profiles, not calibration of the intended tablet.
- `live_server.py <private-run.json>` is a **supervised**, real-provider Unix-socket producer for browser probes. Its telemetry observes production WebSocket frames; authenticated `/e2e/timers` independently queries actual scratch MCP. Mount only its private socket directory into the fixed shared `unix_bridge.py`, on the scratch internal network, replacing only the exact owned transport container. HA stays offline; the host process alone can contact OpenAI. Never use an arbitrary upstream host or production token. Stop the producer before normal `down`.

OpenAI has timer-tool access, not an automatically refreshed view of all timers. Prompt guidance improves safe behavior but cannot guarantee obedience or recognition. Tests verify authoritative tool effects and preservation independently of model prose. Conservative clarification is safe even when the model does not perform another listing; a time-left answer must actually query status. Keep that distinction explicit in reports.
