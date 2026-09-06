# Pipecat External Voice Transport

A self-hosted realtime voice server for the Voice Satellite External Transport Protocol v1. It accepts native PCM audio or client text over an authenticated WebSocket, runs a persistent provider conversation, and returns transcripts, text, and signed streaming WAV audio.

The server currently supports:

- Pipecat with OpenAI Realtime (`gpt-realtime-mini` by default)
- Persistent audio and text turns over one provider session
- Provider-native audio transcription and echoed client-text transcripts
- Server-owned interruption and non-terminal response cancellation
- Short-lived, revocable streaming audio URLs
- Explicitly trusted asynchronous MCP and JSON-line script tools
- A deterministic fake provider for development and protocol testing

> The server protocol is still a development-stage v1. Deploy matching Voice Satellite client support before enabling persistent turns on a satellite. Hardware acceptance on Kiosk Satellite remains outstanding.

## Requirements

### Deployment

- Docker Engine with Compose v2
- An HTTPS/WSS reverse proxy
- A hostname reachable by both Home Assistant and playback devices
- An OpenAI API key when using `openai_realtime`

### Local development

- Python 3.12 or newer
- A C/C++ runtime compatible with Pipecat's dependencies

## Quick start with Docker Compose

1. Clone the repository and enter it:

   ```bash
   git clone https://github.com/brennand97/pipecat-external-voice-transport.git
   cd pipecat-external-voice-transport
   ```

2. Create an untracked environment file:

   ```bash
   cp .env.example .env
   chmod 600 .env
   ```

3. Set at least these values in `.env`:

   ```dotenv
   EXTERNAL_TRANSPORT_TOKEN=replace-with-a-long-random-token
   REALTIME_PROVIDER=openai_realtime
   OPENAI_API_KEY=replace-with-your-openai-api-key
   OPENAI_REALTIME_MODEL=gpt-realtime-mini
   OPENAI_REALTIME_VOICE=marin
   PUBLIC_BASE_URL=https://voice-agent.example
   AUDIO_URL_SIGNING_KEY=replace-with-a-different-long-random-secret
   ```

4. Select a published immutable release tag and start the service:

   ```bash
   # Use the v* tag created from a passing GitHub release-tag workflow.
   IMAGE_TAG=vX.Y.Z docker compose pull
   IMAGE_TAG=vX.Y.Z docker compose up -d
   docker compose ps
   ```

The included Compose file requires `IMAGE_TAG`; it deliberately has no mutable
`main` or `latest` default. A matching `sha-<full-commit-sha>` tag is also
published for every release and is suitable for exact rollback/reproducibility.

The included Compose file binds the service to `127.0.0.1:8080`. Publish it through a reverse proxy rather than exposing port 8080 directly.

## Reverse proxy requirements

Proxy all three routes to `http://127.0.0.1:8080`:

| Route | Purpose |
| --- | --- |
| `/health` and `/ready` | Health checks |
| `/transport/v1` | Long-lived WebSocket sessions |
| `/audio/v1/` | Chunked streaming WAV responses |

The proxy must:

- terminate TLS and support WebSocket upgrades;
- disable response buffering for `/audio/v1/`;
- permit chunked responses with long enough playback timeouts;
- avoid logging query strings on audio routes because they contain short-lived capability tokens;
- restrict access to expected Home Assistant and playback networks where practical.

`PUBLIC_BASE_URL` must be the external HTTPS origin reachable from the playback device. Do not set it to the container name or loopback address in a remote deployment.

## Home Assistant configuration

Configure the matching Voice Satellite integration with:

- **Transport URL:** `wss://voice-agent.example/transport/v1`
- **Transport token:** the value of `EXTERNAL_TRANSPORT_TOKEN`

Keep External Transport disabled until the endpoint, TLS certificate, token, and audio URL are reachable from the Home Assistant host and Kiosk Satellite device.

## Configuration

All runtime settings are environment variables.

### Required

| Variable | Description |
| --- | --- |
| `EXTERNAL_TRANSPORT_TOKEN` | Dedicated bearer token used by Voice Satellite. |

### Provider and output

| Variable | Default | Description |
| --- | --- | --- |
| `REALTIME_PROVIDER` | `fake` | `fake` or `openai_realtime`. |
| `OPENAI_API_KEY` | — | Required for `openai_realtime`. |
| `OPENAI_REALTIME_MODEL` | `gpt-realtime-mini` | Explicit Realtime model. |
| `OPENAI_REALTIME_VOICE` | `marin` | OpenAI Realtime output voice. Select a voice supported by the configured model. |
| `PUBLIC_BASE_URL` | — | Required for OpenAI mode; public HTTPS origin used in audio URLs. |
| `AUDIO_URL_SIGNING_KEY` | — | Required for OpenAI mode; separate secret for audio capabilities. |
| `AUDIO_URL_TOKEN_TTL_SECONDS` | `60` | Lifetime of a signed audio capability. |

### Limits

| Variable | Default | Description |
| --- | --- | --- |
| `MAX_CONCURRENT_SESSIONS` | `2` | Simultaneous provider conversations. |
| `MAX_AUDIO_FRAME_BYTES` | `32000` | Maximum binary PCM WebSocket frame size. |
| `MAX_INPUT_BYTES` | `9600000` | Maximum cumulative input audio per session. |
| `MAX_BUFFERED_AUDIO_FRAMES` | `64` | Bounded input buffering limit. |
| `MAX_BUFFERED_OUTPUT_CHUNKS` | `64` | Bounded assistant-audio buffering limit. |
| `AUDIO_STREAM_WRITE_TIMEOUT_SECONDS` | `1` | Maximum wait for a slow output consumer. |
| `SESSION_START_TIMEOUT_SECONDS` | `5` | Time allowed to receive `session.start`. |
| `INPUT_IDLE_TIMEOUT_SECONDS` | `20` | Idle time allowed while no response is active. |
| `MAX_SESSION_SECONDS` | `300` | Maximum lifetime of one persistent conversation. |

Invalid or missing required configuration fails startup rather than silently falling back.

## Trusted tools

Tools are disabled unless `TRUSTED_TOOL_CONFIG_PATH` names a read-only JSON file mounted by the administrator.

```json
{
  "mcp_servers": [
    {
      "name": "home-assistant",
      "transport": "streamable_http",
      "url": "https://home-assistant.example/api/mcp",
      "bearer_token_env": "HOMEASSISTANT_MCP_TOKEN",
      "allowed_tools": ["get_state", "set_light"],
      "request_timeout_seconds": 15,
      "max_concurrent_calls": 2
    }
  ],
  "script_tools": [
    {
      "name": "calendar",
      "command": ["/usr/local/bin/calendar-tool"],
      "timeout_seconds": 15,
      "max_concurrent_calls": 2
    }
  ]
}
```

Security rules:

- MCP tools must appear in a non-empty server-side `allowed_tools` list.
- Script commands are fixed argv arrays; the model cannot select a command or shell expression.
- Do not put credentials in this JSON file. For authenticated network MCP servers, `bearer_token_env` names an environment variable injected into the service; its value is sent only as an HTTP `Authorization: Bearer` header. For stdio MCP children, `env_names` may copy specifically named variables already injected into the service environment.
- Mount executables and configuration read-only and grant only the permissions required by each tool.

## Protocol overview

The client authenticates with:

```http
Authorization: Bearer <EXTERNAL_TRANSPORT_TOKEN>
```

It sends `session.start`, waits for `session.ready`, and then submits explicit turns. A client may optionally override the default OpenAI system instruction and voice for that one session via `conversation.initial_prompt` and `conversation.initial_voice`; neither is persisted or applied to later sessions:

```json
{
  "type":"session.start",
  "protocol_version":1,
  "session_id":"prompt-test-1",
  "satellite":{"entity_id":"assist_satellite.kitchen","name":"Kitchen"},
  "audio":{"encoding":"pcm_s16le","sample_rate":16000,"channels":1},
  "conversation":{
    "id":null,
    "wake_word":null,
    "initial_prompt":"You are concise. Call home tools silently and state only the result.",
    "initial_voice":"ballad"
  }
}
```

`initial_prompt` must be a non-empty string up to 16,000 UTF-8 bytes; `initial_voice` must be a non-empty string up to 128 bytes. Omit either field (or supply `null`) to use the deployment default. Then start an audio turn:

```json
{"type":"turn.start","turn_id":"turn-1","input":"audio"}
```

Binary input for an audio turn is PCM16 little-endian, 16 kHz, mono. End it with:

```json
{"type":"turn.end","turn_id":"turn-1"}
```

A text turn is:

```json
{"type":"turn.start","turn_id":"turn-2","input":"text"}
{"type":"input.text","turn_id":"turn-2","text":"Turn on the kitchen lights"}
{"type":"turn.end","turn_id":"turn-2"}
```

Response events include `turn_id` and a server-generated `response_id`. `response.cancel` interrupts only that response and leaves the conversation open; `session.cancel` terminates the session. The server decides whether new input represents an interruption: text is immediate intent, while audio waits for provider/VAD speech detection.

See [`docs/protocol-compatibility.md`](docs/protocol-compatibility.md) for the implemented contract and [`docs/turn-session-architecture.md`](docs/turn-session-architecture.md) for lifecycle details.

## Health and operations

```bash
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8080/ready
docker compose logs --tail=100 pipecat-external-voice-transport
docker compose restart pipecat-external-voice-transport
```

- `/health` confirms that the process and event loop are available.
- `/ready` confirms that the process is accepting sessions; it does not open a billable provider session.
- Raw audio is not persisted.
- Access logging is disabled in the supplied container command so signed audio query tokens are not recorded.
- Active provider sessions, streams, and tasks are released on cancellation, disconnect, timeout, or process shutdown.

## Session audit logs

Session audit logging is disabled by default. Mount a writable, access-restricted
host directory at `/var/log/voice-transport`, then configure:

```env
SESSION_AUDIT_MODE=metadata # or debug_content
SESSION_AUDIT_LOG_PATH=/var/log/voice-transport
SESSION_AUDIT_RETENTION_DAYS=7
```

Daily `sessions-YYYY-MM-DD.jsonl` files are pruned after the configured
retention period. `metadata` records correlated lifecycle IDs, timing, and tool
outcomes but omits transcript, tool argument, and tool-result content.
`debug_content` additionally records transcripts and tool arguments/results so
an operator can reconstruct a conversation and its agent actions. Both modes
always redact common credential fields and signed URL query strings, but
`debug_content` remains sensitive personal/home data and must be enabled only
for deliberate diagnostics.

To roll back, set `IMAGE_TAG` to the prior tested tag and run:

```bash
IMAGE_TAG=<prior-tag> docker compose pull
IMAGE_TAG=<prior-tag> docker compose up -d
```

## Local development

Create an isolated environment and install all development extras:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[realtime,tools,dev]'
ruff format --check .
ruff check .
pytest
```

Run safely with the fake provider:

```bash
EXTERNAL_TRANSPORT_TOKEN=development-token \
REALTIME_PROVIDER=fake \
uvicorn voice_transport.app:runtime_app --factory --reload
```

Run against OpenAI only with an untracked environment file or process environment:

```bash
set -a
. ./.env
set +a
uvicorn voice_transport.app:runtime_app --factory --host 127.0.0.1 --port 8080
```

### Opt-in live MCP E2E test

The network-free suite is the default. A separate billable, read-only live test
proves that a Realtime model receives the configured function schema and calls
Home Assistant MCP's `homeassistant__GetLiveContext` tool. It is skipped unless
explicitly enabled and writes only temporary audit files:

```bash
RUN_LIVE_OPENAI_MCP_TEST=1 \
OPENAI_API_KEY=... \
HOMEASSISTANT_MCP_TOKEN=... \
LIVE_OPENAI_REALTIME_MODEL=gpt-realtime-2.1-mini \
.venv/bin/pytest -q tests/integration/test_live_openai_mcp.py
```

Never place provider keys, transport tokens, signing keys, Home Assistant credentials, or signed audio URLs in source control, command output, test fixtures, or bug reports.

## Validation status

The automated suite covers protocol parsing, persistent sequential turns, text interruption, provider/VAD-gated audio interruption, stale-event fencing, signed stream expiry/revocation, bounded queues, tool adapters, and cleanup. A bounded live test has validated an audio turn followed by a text turn over one OpenAI Realtime session, including both transcript sources and streamed WAV output.

Target-device playback, echo behavior, and end-to-end Kiosk Satellite acceptance must still be validated in the intended deployment environment.
