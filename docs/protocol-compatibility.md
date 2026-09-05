# Protocol compatibility

| Protocol | Server status | Notes |
| --- | --- | --- |
| External Transport v1 (development draft) | Implemented server-side | Persistent audio/text turns, native/client transcripts, response correlation, server-owned interruption, non-terminal response cancellation, and signed streaming audio. |

## Turn controls

```json
{"type":"turn.start","turn_id":"turn-1","input":"audio"}
{"type":"turn.start","turn_id":"turn-2","input":"text"}
{"type":"input.text","turn_id":"turn-2","text":"hello"}
{"type":"turn.end","turn_id":"turn-2"}
{"type":"response.cancel","response_id":"1"}
```

Binary PCM belongs to the currently open audio turn. Turn IDs must be non-empty and unique within a session. Text turns accept exactly one non-empty input of at most 4,000 UTF-8 bytes. `input.end` remains an alias for ending an implicit audio turn during development. `session.cancel` is terminal; `response.cancel` leaves the provider conversation alive.

Response events include `turn_id` and `response_id`. Client text is echoed as `user.transcript.final` with `source: client_text`; provider-native audio transcription uses `source: provider_audio`.

The server owns interruption policy. Text starts interrupt active output immediately. Audio does not interrupt merely because bytes arrive; the persistent provider's genuine-speech event triggers cancellation. Every interrupted output capability is revoked, and stale provider events are fenced until the next turn is dispatched.

Unknown controls, unsupported protocol/audio formats, duplicate turn IDs, mismatched modalities, and invalid turn transitions fail closed. Input, event, and output queues use bounded waits. Pipecat is imported while constructing an OpenAI-configured application, before accepting sessions, to avoid first-session import latency.
