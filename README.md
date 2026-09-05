# Pipecat External Voice Transport

A network service implementing the provider-neutral Voice Satellite External Transport Protocol v1. Provider integration is intentionally isolated here; the Voice Satellite fork only relays protocol control frames and native PCM.

## Status

The service provides authenticated protocol handling, strict development-v1 validation, bounded ordered PCM ingress, concurrency limits, and deterministic fake-agent coverage. One persistent provider session accepts correlated audio and text turns. Its provider boundary supports `fake` and Pipecat-backed `openai_realtime` with an explicit model. OpenAI readiness is gated on its completed session update; its module is loaded during application construction rather than on the first turn.

Every response has a server-generated `response_id`. Text input interrupts active output immediately; audio waits for provider/VAD genuine-speech detection. Interruption cancels provider output, revokes the matching signed WAV stream, and emits `assistant.interrupted` without closing the conversation. Output buffering is bounded; unread audio is discarded after a bounded wait.

The live server-to-OpenAI audio and text provider paths are validated, including provider-native final transcription. HA/Kiosk hardware acceptance remains separate integration work; do not select it for production satellite use yet.

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
