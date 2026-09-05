# Pipecat External Voice Transport Server — Implementation Plan

## 1. Purpose

Build and operate a network-hosted Pipecat service that implements the generic **Voice Satellite External Transport Protocol v1**. The service will receive Kiosk Satellite's native PCM and pre-roll through the Voice Satellite fork, run an OpenAI Realtime conversation, expose controlled Home Assistant tools, and return low-latency streaming response audio.

This repository is provider-specific. Pipecat and OpenAI implementation details belong here, while the Voice Satellite fork remains provider-neutral.

## 2. Goals

- Implement External Transport Protocol v1.
- Accept native 16 kHz mono PCM16 audio over an authenticated WebSocket.
- Preserve pre-roll ordering and support seamless wake phrases.
- Run OpenAI Realtime through Pipecat with an explicitly selected cost-effective model.
- Stream response audio with low startup latency to Kiosk Satellite's native player.
- Support transcripts, interruptions, conversation continuation, and clean cancellation.
- Provide narrowly scoped Home Assistant tools.
- Run as a hardened container on an existing Docker host.
- Publish tested, immutable images to GitHub Container Registry.
- Provide useful metrics without retaining private audio or secrets.

## 3. Non-goals

- Running inside Home Assistant OS or a Home Assistant custom component.
- Implementing wake-word detection.
- Exposing arbitrary Home Assistant service calls to the model.
- Persisting raw microphone or response audio by default.
- Depending on Voice Satellite's internal Python or JavaScript implementation.
- Storing deployment credentials in GitHub.

## 4. Architecture

```text
Android tablet / Kiosk Satellite
  native wake + native pre-roll
        |
        | binary PCM over HA WebSocket
        v
Voice Satellite External Transport client
        |
        | WSS External Transport Protocol v1
        v
Pipecat external voice server
  +-- session manager
  +-- PCM input adapter
  +-- OpenAI Realtime service
  +-- Home Assistant tools
  +-- event normalizer
  `-- signed streaming-audio endpoint
        |
        v
Kiosk Satellite native playSound(stream=true)
```

Home Assistant relays audio and protocol events only. It must not run Pipecat, transcode audio, or hold the OpenAI API key.

## 5. Proposed repository structure

```text
pipecat-external-voice-transport/
  src/
    app.py
    config.py
    protocol/
      __init__.py
      messages.py
      session.py
      validation.py
    transport/
      __init__.py
      websocket.py
      audio_input.py
      audio_output.py
    agent/
      __init__.py
      pipeline.py
      prompts.py
      context.py
    tools/
      __init__.py
      home_assistant.py
      allowlist.py
      confirmation.py
    observability/
      __init__.py
      logging.py
      metrics.py
  tests/
    unit/
    integration/
    contract/
    fixtures/
  docs/
    deployment.md
    operations.md
    security.md
    protocol-compatibility.md
  Dockerfile
  docker-compose.yml
  pyproject.toml
  README.md
  PLAN.md
  .github/
    workflows/
```

## 6. API surface

### 6.1 Liveness

```http
GET /health
```

Return success if the process and event loop are healthy. Do not call external providers.

### 6.2 Readiness

```http
GET /ready
```

Validate required configuration and internal components. Optional provider checks must avoid opening a billable realtime session.

### 6.3 Session transport

```text
WSS /transport/v1
Authorization: Bearer <external-transport-token>
```

Rules:

- Authenticate before accepting a session.
- Require `session.start` as the first message.
- Reject unsupported protocol versions before accepting PCM.
- Use JSON text frames for control/events.
- Use binary client-to-server frames for input PCM.
- Enforce message, frame, duration, and concurrency limits.

### 6.4 Response audio

```http
GET /audio/v1/{session_id}/{audio_id}?token=<short-lived-signature>
```

Requirements:

- Unpredictable identifiers
- Short expiration
- Signature bound to session and audio ID
- Invalidation when session/output closes
- No permanent credentials in the URL
- No signed query values in access logs
- Correct content type and no unsafe caching

## 7. Protocol implementation

The canonical specification is maintained by the Voice Satellite fork. This repository should include compatibility documentation, schemas/fixtures, and the implemented protocol version.

### 7.1 Start message

Accept:

```json
{
  "type": "session.start",
  "protocol_version": 1,
  "session_id": "01J...",
  "satellite": {
    "entity_id": "assist_satellite.kitchen",
    "name": "Kitchen"
  },
  "audio": {
    "encoding": "pcm_s16le",
    "sample_rate": 16000,
    "channels": 1
  },
  "conversation": {
    "id": null,
    "wake_word": "Okay Nabu"
  }
}
```

Validate every field and reject duplicate active session IDs.

### 7.2 Ready response

Send only after the internal Pipecat session can safely accept buffered audio:

```json
{
  "type": "session.ready",
  "session_id": "01J...",
  "capabilities": {
    "transcription": true,
    "streaming_audio_url": true,
    "interruptions": true,
    "conversation_continuation": true
  }
}
```

### 7.3 Control and event messages

Implement the v1 messages defined by the Voice Satellite fork, including:

- `input.end`
- `input.pause`
- `input.resume`
- `session.cancel`
- `user.speech_started`
- `user.speech_stopped`
- `user.transcript.partial`
- `user.transcript.final`
- `assistant.response_started`
- `assistant.text.delta`
- `assistant.text.final`
- `assistant.audio`
- `assistant.interrupted`
- `assistant.response_finished`
- `session.finished`
- `error`

Unknown optional fields should be tolerated where forward compatibility permits. Unknown message types must return a protocol error rather than being silently interpreted.

## 8. Session state machine

Use explicit state transitions:

```text
NEW
  -> CONNECTING
  -> READY
  -> LISTENING
  -> RESPONDING
  -> FINISHED

Active state
  -> CANCELLING
  -> FINISHED

Any state
  -> FAILED
```

Reject invalid transitions. Every terminal path must release:

- Pipecat worker/pipeline
- OpenAI connection
- input queues
- output queues
- signed audio URLs
- tasks and timers
- Home Assistant client resources scoped to the session

## 9. Input audio

Expected client audio:

```text
Encoding: signed PCM16 little-endian
Sample rate: 16000 Hz
Channels: 1
Byte rate: 32000 bytes/second
```

Requirements:

- Preserve binary frame order.
- Validate even byte length and frame-size limits.
- Track input bytes and derived duration.
- Apply bounded queues and backpressure.
- Reject binary input before `session.ready`.
- Enforce maximum input and total session duration.
- Do not persist raw PCM by default.
- Resample inside this service if Pipecat/OpenAI needs a different rate.

Kiosk pre-roll arrives as the first audio frames and does not need a provider-specific marker. It must be forwarded before live frames without delay or reordering.

## 10. Pipecat pipeline

### 10.0 Session boundary

The WebSocket layer is an orchestrator, not a Pipecat transport implementation.
Each connection creates one isolated provider session with this lifecycle:
`start()`, `push_audio(pcm)`, `end_input()`, `cancel()`, consume `events()`,
then `close()`. The Pipecat implementation owns its pipeline, worker, provider
connection, and feature-specific tasks behind that boundary. A deterministic
fake implementation uses the same boundary in contract tests.


Use Pipecat's OpenAI Realtime service and set the model explicitly. Do not rely on Pipecat's default model because it can change between versions.

Initial model target:

```python
OpenAIRealtimeLLMService(
    api_key=settings.openai_api_key,
    settings=OpenAIRealtimeLLMService.Settings(
        model="gpt-realtime-mini",
        # Session properties configured explicitly.
    ),
)
```

Initial behavior:

- OpenAI/server semantic VAD
- Near-field input noise reduction
- Concise spoken-response system prompt
- Audio output enabled
- Input transcription enabled
- Function calling enabled only for registered tools
- Usage metrics enabled
- Bounded conversation context
- Explicit idle, response, and total-session timeouts

Conceptual pipeline:

```text
External PCM input
  -> Pipecat input transport
  -> OpenAI Realtime
  -> transcript/event adapter
  -> output audio buffer/stream
  -> External Transport events
```

## 11. Custom Pipecat transport adapter

Implement a small Pipecat transport adapter around the protocol session instead of exposing Pipecat internals over the network.

Responsibilities:

- Convert binary PCM into Pipecat audio frames.
- Forward user and assistant lifecycle frames into protocol events.
- Receive generated output audio frames.
- Publish output into the session's streaming endpoint.
- Propagate cancellation in both directions.
- Apply backpressure rather than accumulating unbounded audio.

Keep protocol models independent from Pipecat classes so future framework upgrades affect only the adapter.

## 12. Output audio

OpenAI Realtime audio must be exposed as a stream Kiosk Satellite can begin playing before generation finishes.

Evaluate these formats on the target Android tablet:

1. Streaming WAV/PCM
2. Ogg/Opus
3. WebM/Opus
4. AAC if compatibility requires it

Measure:

- Time to first playable audio
- Encoder buffering latency
- CPU usage
- Android native-player support
- Behavior when the stream ends normally
- Behavior when stopped early
- Recovery after a dropped stream

Prefer no transcoding if native playback is reliable. Otherwise use a bounded streaming encoder process/library and monitor its resource consumption.

Emit:

```json
{
  "type": "assistant.audio",
  "id": "audio-123",
  "url": "https://voice-agent.example/audio/v1/...?...",
  "content_type": "audio/ogg"
}
```

## 13. Interruption and barge-in

During assistant playback:

1. Continue consuming microphone audio.
2. Detect genuine user speech using the selected Pipecat/OpenAI turn strategy.
3. Cancel the active OpenAI response.
4. Close or invalidate the remaining output stream.
5. Emit `assistant.interrupted` immediately.
6. Allow Voice Satellite to stop Kiosk Satellite's native playback.
7. Continue into the new user turn.

Test against speaker echo. The tablet's configured Android `Voice communication` capture path should provide hardware echo cancellation, but this must be verified on the actual device.

## 14. Conversation management

- Accept an optional prior conversation ID.
- Return a continuation ID when appropriate.
- Bound context growth.
- Truncate or summarize according to an explicit policy.
- End abandoned sessions promptly.
- Do not keep idle OpenAI Realtime sessions open indefinitely.
- Ensure one satellite cannot accidentally attach to another satellite's session.

Initial limits should be configurable, with conservative defaults:

```text
Ready timeout: 5 seconds
Input idle timeout: 20 seconds
Maximum response duration: 60 seconds
Maximum session duration: 5 minutes
Maximum concurrent sessions: 2
```

## 15. Home Assistant tools

### 15.1 Initial allowlist

Start with narrow tools:

- Read an exposed entity's state
- Turn allowlisted lights/switches on or off
- Set allowlisted light brightness
- Activate allowlisted scenes
- Set allowlisted climate targets
- Run explicitly allowlisted scripts

### 15.2 Excluded initially

- Locks
- Alarm panels
- Garage doors/covers with security impact
- Shell commands
- Arbitrary services
- Configuration changes
- Restarts, updates, or backups

### 15.3 Safety model

- Enforce allowlists server-side, never only through prompting.
- Validate domains, services, entity IDs, values, and ranges.
- Add confirmations for consequential actions.
- Use idempotency keys where possible.
- Return structured tool errors to the model.
- Set strict tool-call timeouts.
- Record tool name and result status without leaking sensitive state.

The Home Assistant token must be provided through deployment secrets and must have only the necessary access. Never store it in GitHub or bake it into an image.

## 16. Authentication and authorization

- Use a dedicated transport bearer token for Voice Satellite-to-server connections.
- Compare static tokens in constant time.
- Support future token rotation without protocol changes.
- Keep OpenAI, Home Assistant, transport, and audio-signing credentials separate.
- Authenticate audio URLs with short-lived per-object signatures.
- Restrict endpoint access at the reverse proxy/firewall where practical.
- Rate-limit failed authentication and session creation.

## 17. Container security

- Multi-stage build.
- Minimal pinned base image.
- Non-root runtime user.
- Read-only root filesystem where supported.
- Writable temporary filesystem only where required.
- Drop Linux capabilities.
- `no-new-privileges` enabled.
- Resource limits configured.
- No host network unless a demonstrated requirement exists.
- No Docker socket mount.
- Health check using `/health`.
- Graceful stop timeout long enough to close active sessions.

## 18. Secrets and deployment configuration

Required deployment secrets may include:

```text
OPENAI_API_KEY
HOME_ASSISTANT_TOKEN
EXTERNAL_TRANSPORT_TOKEN
AUDIO_URL_SIGNING_KEY
```

Non-secret configuration may include:

```text
OPENAI_REALTIME_MODEL=gpt-realtime-mini
HOME_ASSISTANT_URL=https://...
PUBLIC_BASE_URL=https://...
MAX_CONCURRENT_SESSIONS=2
LOG_LEVEL=INFO
```

Secrets must be injected by the deployment platform or secret files outside Git. They must never appear in Compose files committed to the repository, image layers, logs, or diagnostic endpoints.

## 19. Network deployment

- Host on an existing Docker server.
- Publish through the existing reverse proxy using HTTPS/WSS.
- Limit network access to Home Assistant and expected management clients.
- Ensure the public base URL used in signed audio links resolves from the tablet.
- Configure proxy buffering appropriately for streaming audio.
- Configure WebSocket upgrade and long-lived connection timeouts.
- Do not expose the transport directly to the public internet unless required and separately reviewed.

Initial resource allocation:

```text
CPU: 1 core
Memory: 512 MB to 1 GB
Restart policy: unless-stopped
```

Measure before changing limits.

## 20. Resilience and shutdown

Handle explicitly:

- Voice Satellite disconnect
- Home Assistant restart
- Pipecat processor failure
- OpenAI disconnect/rate limit
- Home Assistant tool timeout
- Output stream not fetched
- Output consumer disconnect
- Duplicate session ID
- Session cancellation
- Reverse-proxy timeout
- Container shutdown

Graceful shutdown sequence:

1. Mark readiness false.
2. Stop accepting sessions.
3. Notify active clients when possible.
4. Cancel model responses.
5. Close output streams.
6. Cancel and await session tasks.
7. Exit after a bounded grace period.

## 21. Observability

Use structured logs with:

- Session ID
- Satellite name/entity ID
- State transition
- Protocol version
- Input/output durations
- Time to ready
- Time to first transcript
- Time to first response audio
- Provider request/result status
- Tool name and result status
- Stable error code

Never log by default:

- Raw PCM
- Full transcripts containing private speech, unless explicitly opted in
- Authorization headers
- Signed audio URLs
- OpenAI or Home Assistant credentials
- Sensitive entity values

Suggested metrics:

```text
active_sessions
sessions_total
session_errors_total
time_to_ready_seconds
time_to_first_transcript_seconds
time_to_first_audio_seconds
session_duration_seconds
input_audio_seconds
output_audio_seconds
openai_input_tokens
openai_output_tokens
tool_calls_total
tool_call_duration_seconds
interruptions_total
```

## 22. Test plan

### 22.1 Unit tests

- Protocol parsing and validation
- Session state transitions
- Authentication
- Constant-time token checks
- Audio frame validation/accounting
- Queue limits and backpressure
- Signed URL generation, expiration, and invalidation
- Error normalization
- Tool allowlists and argument validation
- Secret/log redaction

### 22.2 Integration tests

Use fake OpenAI Realtime and Home Assistant servers:

- Complete successful turn
- Partial/final transcription
- Streaming output
- Tool execution
- Interruption
- OpenAI timeout and disconnect
- HA tool timeout and rejection
- Session cancellation
- Concurrent sessions
- Graceful process shutdown

### 22.3 Contract tests

Use protocol fixtures compatible with the Voice Satellite fork:

- Valid v1 session
- Unsupported version
- Invalid first message
- Authentication failure
- Binary frame before readiness
- Slow readiness
- Mid-session disconnect
- Normal completion
- Error completion

Either duplicate immutable fixtures with version checks or publish a small schema artifact. Do not create a runtime dependency between repositories.

### 22.4 Hardware acceptance tests

On the actual tablet and deployment network:

- Wake plus immediate command is not clipped.
- Pre-roll arrives before live audio.
- Dashboard remains smooth.
- Screen-off wake succeeds.
- Native output starts promptly.
- Assistant output does not self-trigger repeatedly.
- Barge-in stops output promptly.
- Sessions clean up after network loss.
- Wake detection always re-arms.

## 23. GitHub Actions plan

### 23.1 Pull requests

Run with read-only permissions and no production secrets:

- Python formatting
- Linting
- Static type checking
- Unit tests
- Integration tests using fakes
- Protocol contract tests
- Container build without push
- Dependency/security scan
- Secret scan

Pin all actions to full commit SHAs. Use Dependabot to propose updates to actions, Python dependencies, and Docker base images.

### 23.2 Main branch and releases

After required tests:

- Build the image once.
- Generate an SBOM.
- Run an image vulnerability scan.
- Publish immutable version and commit-SHA tags to GHCR.
- Attach provenance/signature where supported.
- Do not publish `latest` as the only deployment reference.

Recommended tags:

```text
v1.0.0
sha-<commit>
```

Production deployment should use a version tag or digest.

### 23.3 Workflow security

- Least-privilege job permissions.
- No production deployment secrets in PR workflows.
- No `pull_request_target` execution of contributor code.
- Protected release environment if automated deployment is later added.
- Explicit job timeouts.
- Concurrency controls for image publication.
- Retain failed test artifacts without including secrets or raw audio.

## 24. Operations

Document:

- Installation and upgrade procedure
- Credential provisioning and rotation procedure
- Health and readiness interpretation
- Expected ports and reverse-proxy settings
- Log and metric locations
- Rollback procedure
- OpenAI usage/cost checks
- Protocol compatibility matrix
- Common failure recovery
- How to disable external transport and return satellites to HA Assist

Backups should cover configuration templates and operational documentation, not ephemeral sessions or audio.

## 25. Delivery phases

### Phase 1: Service and protocol skeleton

- FastAPI application
- Health/readiness endpoints
- Authentication
- Protocol models
- Session state machine
- Fake echo implementation
- Contract test harness

### Phase 2: Native input integration

- Binary PCM ingestion
- Pipecat input adapter
- Queue limits and backpressure
- Pre-roll ordering tests
- Cancellation and disconnect handling

### Phase 3: OpenAI Realtime

- Pipecat OpenAI Realtime pipeline
- Explicit `gpt-realtime-mini` selection
- VAD and transcription
- Text event mapping
- Usage metrics and context limits

### Phase 4: Streaming output

- Evaluate output formats on Android
- Implement signed streaming endpoint
- Emit `assistant.audio`
- Handle completion, cancellation, and consumer disconnect

### Phase 5: Interruptions

- Continue input during playback
- Detect genuine barge-in
- Cancel provider output
- Invalidate output stream
- Emit interruption event
- Tune echo cancellation behavior

### Phase 6: Home Assistant tools

- Read-only state tools
- Allowlisted controls
- Confirmation policy
- Idempotency and timeout handling
- Audit-safe logging

### Phase 7: Production hardening

- Hardened container
- Reverse-proxy deployment
- Metrics and dashboards
- GitHub CI and GHCR publication
- Security and failure-mode review
- Rollback and operations documentation

## 26. Acceptance criteria

- Implements the documented External Transport Protocol v1.
- Accepts and preserves 16 kHz PCM16 pre-roll and live audio ordering.
- Uses an explicitly configured Realtime model.
- Streams response audio to Kiosk Satellite with acceptable measured startup latency.
- Supports interruption without leaving orphaned output or model sessions.
- Restricts Home Assistant tools through enforceable allowlists.
- Releases all resources after normal completion and every tested failure path.
- Does not expose secrets or persist raw audio by default.
- Publishes tested, immutable container images through GitHub Actions.
- Passes unit, integration, contract, container, and target-tablet acceptance tests.
