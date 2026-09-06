# Persistent Turn Session Architecture

## Status

Design for the development-stage External Transport v1 expansion. This replaces
the server's current one-input/one-response lifecycle before implementation.
The canonical protocol and the Voice Satellite client must change in the same
release; no server-only extension is permitted.

## Goals

- One authenticated session can contain ordered audio and text turns.
- Text and audio are first-class inputs to the same provider-neutral turn API.
- The server, never the client, owns interruption policy.
- A provider connection and event stream remain alive across turns.
- Every observable input/output event is correlated to a `turn_id` and a
  server-assigned `response_id`.
- Late events or audio from an interrupted response cannot be attached to a
  newer turn.
- All queues, audio streams, tasks, and provider calls are bounded.

## Protocol draft

The development v1 protocol gains the following controls:

```json
{"type":"turn.start","turn_id":"client-unique","input":"audio"}
{"type":"input.text","turn_id":"client-unique","text":"Turn on the lights"}
{"type":"turn.end","turn_id":"client-unique"}
{"type":"response.cancel","response_id":"server-response-id","reason":"user_stopped_playback"}
```

Binary PCM belongs only to the active audio turn. A text turn contains exactly
one `input.text`, limited to 4,000 UTF-8 bytes. Existing `input.end` remains a
development-only alias for ending the implicit initial audio turn. `session.cancel`
remains terminal.

All turn-associated server events include `turn_id`; response events also include
`response_id`. `user.transcript.final` contains `source: "client_text"` or
`source: "provider_audio"`.

## Session actor

`ConversationSession` is a single-owner async actor. The WebSocket reader only
validates frames and submits typed commands; it never directly calls a provider
or sends provider events. The actor owns:

- session and turn state;
- one persistent provider conversation;
- the active response lease and its signed audio stream;
- a bounded inbound command queue and a bounded provider-event queue;
- all turn, output-expiry, and shutdown tasks.

The actor serializes state transitions, while provider I/O and WebSocket writing
run as supervised child tasks. It is the sole authority allowed to create or
revoke response audio capabilities.

```text
WebSocket reader -> typed command queue -> ConversationSession actor
Provider event stream -----------------> bounded event queue -> actor
actor -> ordered server-event queue -> WebSocket writer
```

The writer never performs provider work. A failed writer/disconnect submits a
terminal command and all child tasks are cancelled with bounded waits.

## Turn state

A turn is immutable metadata plus mutable actor-owned state:

```text
OPEN_AUDIO | OPEN_TEXT | ENDED | RESPONDING | INTERRUPTED | COMPLETED | FAILED
```

Only one input turn is open at once. There may be at most one active response.
A completed response does not finish the conversation session; the session
returns to `READY_FOR_TURN` until terminal cancellation, idle expiry, or maximum
session duration.

A provider adapter receives `submit_audio(turn_id, pcm)`, `submit_text(turn_id,
text)`, `end_turn(turn_id)`, and `interrupt(response_id)`. It emits normalized
provider events with its provider response handle. The actor maps that handle to
the response lease created for the turn. Provider-specific Pipecat/OpenAI frame
classes never cross this boundary.

## Interruption policy

### Text

A non-empty text submission while a response is active is definitive user
intent. Before dispatching the text, the actor:

1. transitions the old response to `INTERRUPTING`;
2. asks the provider adapter to interrupt that response;
3. revokes the old response's audio stream;
4. emits `assistant.interrupted` for the old `turn_id`/`response_id`;
5. emits the submitted `user.transcript.final` with `source: "client_text"`;
6. dispatches the text turn.

### Audio

Audio received while a response is active is routed to a new audio turn without
immediate cancellation. The provider adapter/VAD emits a normalized
`user.speech_started(turn_id)` only after genuine speech is detected. The actor
then applies the same interruption sequence. Input before that signal remains
bounded and is never exposed as a client interruption decision.

### Explicit playback stop

`response.cancel` names the active `response_id`, interrupts it, and returns the
session to `READY_FOR_TURN`. It does not close the conversation.

## Output leases

Each response has at most one `OutputLease`:

```text
(response_id, turn_id, stream_id, generation, state)
```

The generation is incremented before interruption. The actor drops every later
provider audio/text event whose response handle/generation does not match the
active lease. Revocation closes active HTTP consumers and makes new signed URL
opens fail. Audio writes use bounded waits; an unread stream is detached without
blocking the provider or retaining audio.

## Provider requirements

The generic provider contract is persistent, not one-shot:

```python
async def start() -> None
async def submit_audio(turn_id: str, pcm: bytes) -> None
async def submit_text(turn_id: str, text: str) -> None
async def end_turn(turn_id: str) -> None
async def interrupt(response_id: str) -> None
async def close() -> None
async def events() -> AsyncIterator[ProviderEvent]
```

`events()` ends only on provider/session closure, never at a response boundary.
The Pipecat adapter owns mapping of `InterruptionFrame`, VAD speech frames,
OpenAI response handles, text, native transcription, and PCM frames into this
contract.

## Acceptance tests

1. Two sequential audio turns use one provider session and produce distinct
   turn/response IDs.
2. A text turn during streaming audio interrupts the old response, revokes its
   URL, echoes a client-text transcript, and produces a new response.
3. Audio during playback does not interrupt until fake VAD emits genuine speech.
4. A late old-response audio event is dropped after interruption.
5. `response.cancel` is non-terminal; `session.cancel` is terminal.
6. Disconnect, provider failure, unread audio, queue pressure, and every timer
   leave no tasks, streams, registry entries, or live provider connection.
7. Contract tests run against the Voice Satellite client implementation; live
   OpenAI validation verifies text input, native audio transcript, interruption,
   and audio URL revocation.
