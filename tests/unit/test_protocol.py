import pytest

from voice_transport.protocol import (
    ProtocolViolation,
    parse_session_start,
    validate_control,
)


def valid_start() -> dict:
    return {
        "type": "session.start",
        "protocol_version": 1,
        "session_id": "session-1",
        "satellite": {"entity_id": "assist_satellite.kitchen", "name": "Kitchen"},
        "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1},
        "conversation": {"id": None, "wake_word": "Okay Nabu"},
    }


def test_parses_valid_v1_start() -> None:
    start = parse_session_start(valid_start())
    assert start.session_id == "session-1"
    assert start.satellite_entity_id == "assist_satellite.kitchen"


@pytest.mark.parametrize(
    "field,value", [("protocol_version", 2), ("type", "input.end")]
)
def test_rejects_invalid_start(field: str, value: object) -> None:
    message = valid_start()
    message[field] = value
    with pytest.raises(ProtocolViolation):
        parse_session_start(message)


def test_rejects_non_pcm16_audio() -> None:
    message = valid_start()
    message["audio"]["sample_rate"] = 24000
    with pytest.raises(ProtocolViolation, match="16 kHz"):
        parse_session_start(message)


def test_accepts_only_defined_controls() -> None:
    assert validate_control({"type": "input.end"}) == "input.end"
    with pytest.raises(ProtocolViolation, match="Unsupported"):
        validate_control({"type": "input.pause"})
