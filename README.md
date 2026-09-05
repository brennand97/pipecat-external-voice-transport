# Pipecat External Voice Transport

A network service implementing the provider-neutral Voice Satellite External Transport Protocol v1. Provider integration is intentionally isolated here; the Voice Satellite fork only relays protocol control frames and native PCM.

## Status

The service provides authenticated protocol handling, strict v1 validation, bounded ordered PCM ingress, concurrency limits, cancellation, and deterministic fake-agent coverage. Its provider boundary supports `fake` and Pipecat-backed `openai_realtime` with an explicit model. OpenAI readiness is gated on its completed session update; assistant PCM is exposed through a signed, short-lived streaming WAV capability.

A response can be cancelled while audio is streaming. The server cancels the provider, revokes the active stream, emits `assistant.interrupted`, and finishes the session. Output buffering is bounded; if no client consumes an announced stream, audio is discarded after a bounded wait rather than blocking the provider indefinitely.

The live server-to-OpenAI transport path is validated with the bundled PCM fixture. HA/Kiosk hardware acceptance, provider-native transcript consistency, and multi-turn barge-in remain separate integration work; do not select it for production satellite use yet.

## Local development

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[realtime,tools,dev]'
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

## Trusted tools

Set `TRUSTED_TOOL_CONFIG_PATH` to a read-only deployment-mounted JSON file to enable tools. The file declares only fixed MCP endpoints or fixed executable argv vectors; no shell expressions or literal secrets are accepted. MCP entries require a non-empty `allowed_tools` allowlist. `env_names` copies only named values already injected into the service environment.

```json
{
  "mcp_servers": [{
    "name": "home-assistant",
    "transport": "streamable_http",
    "url": "https://home-assistant.example/mcp",
    "allowed_tools": ["get_state", "set_light"]
  }],
  "script_tools": [{
    "name": "calendar",
    "command": ["/usr/local/bin/calendar-tool"],
    "timeout_seconds": 15
  }]
}
```

Tool configuration is trusted administrator input and must be mounted outside Git. It must not contain credentials; provide required child-process values through the environment and list only their names in `env_names`.
