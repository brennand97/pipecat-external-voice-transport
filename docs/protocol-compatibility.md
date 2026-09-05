# Protocol compatibility

| Protocol | Server status | Notes |
| --- | --- | --- |
| External Transport v1 | Partial / Phase 1 | Authenticated session establishment, PCM validation, `input.end`, `session.cancel`, lifecycle and error events. |

Phase 1 intentionally reports all optional capabilities as `false`. Pipecat, transcription, interruption, conversation continuation, signed streaming audio URLs, and Home Assistant tools will be enabled only when each is implemented and tested.

The server rejects unknown control types and unsupported protocol or audio formats. It preserves binary-frame order while accounting for bounded input; later adapters must retain that ordering for Kiosk Satellite pre-roll.
