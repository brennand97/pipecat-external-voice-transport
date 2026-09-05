# Pipecat External Voice Transport

A network service implementing the provider-neutral Voice Satellite External Transport Protocol v1. Provider integration is intentionally isolated here; the Voice Satellite fork only relays protocol control frames and native PCM.

## Phase 1 status

The service currently provides an authenticated protocol skeleton with health/readiness endpoints, strict v1 validation, bounded PCM accounting, concurrency limits, cancellation, and deterministic fake-agent completion. It **does not yet include Pipecat, OpenAI Realtime, Home Assistant tools, or streaming output audio**. Do not select it for production satellite use until later phases are complete.

## Local development

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
ruff check .
EXTERNAL_TRANSPORT_TOKEN=development-token \
  uvicorn voice_transport.app:runtime_app --factory --reload
```

- `GET /health` verifies process/event-loop liveness.
- `GET /ready` reports whether this process is accepting sessions.
- `WSS /transport/v1` requires `Authorization: Bearer <EXTERNAL_TRANSPORT_TOKEN>`.

Copy `.env.example` to an untracked deployment secret file before using `docker-compose.yml`. The compose file binds the service to loopback; put it behind the existing HTTPS/WSS reverse proxy. Never log, commit, or place transport, provider, Home Assistant, or signing credentials in an image.

## Protocol compatibility

Implemented protocol version: **v1**. The canonical protocol specification is maintained in the Voice Satellite fork at `docs/external-transport-protocol.md`; this repository maintains validation and contract tests without a runtime dependency on that repository.

Input is binary, 16 kHz mono signed PCM16 little-endian. The first control frame must be `session.start`; audio is accepted only after `session.ready`. The fake implementation emits text lifecycle events on `input.end` solely to exercise the client lifecycle.
