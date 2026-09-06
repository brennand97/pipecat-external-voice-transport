import pytest

from voice_transport.protocol import (
    ProtocolViolation,
    parse_input_text,
    parse_session_start,
    parse_turn_start,
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


def test_generic_text_client_has_no_satellite_or_device_context() -> None:
    message = valid_start()
    message.pop("satellite")
    message["client"] = {"id": "conversation.reginold", "kind": "ha_conversation"}
    message["conversation"] = {
        "id": "ha-conversation-id",
        "device_id": None,
        "input_modalities": ["text"],
        "output_modalities": ["text"],
    }

    start = parse_session_start(message)

    assert start.satellite_entity_id is None
    assert start.client_kind == "ha_conversation"
    assert start.input_modalities == frozenset({"text"})
    assert start.output_modalities == frozenset({"text"})


def test_optional_initial_prompt_overrides_a_single_session() -> None:
    message = valid_start()
    message["conversation"]["initial_prompt"] = "Answer in a pirate voice."
    message["conversation"]["initial_voice"] = "ballad"

    start = parse_session_start(message)
    assert start.initial_prompt == "Answer in a pirate voice."
    assert start.initial_voice == "ballad"


@pytest.mark.parametrize(
    "field,value", [("protocol_version", 2), ("type", "input.end")]
)
def test_rejects_invalid_start(field: str, value: object) -> None:
    message = valid_start()
    message[field] = value
    with pytest.raises(ProtocolViolation):
        parse_session_start(message)


def test_rejects_invalid_initial_prompt() -> None:
    message = valid_start()
    message["conversation"]["initial_prompt"] = " "
    with pytest.raises(ProtocolViolation, match="initial_prompt"):
        parse_session_start(message)

    message["conversation"]["initial_prompt"] = "x" * 16_001
    with pytest.raises(ProtocolViolation, match="16,000"):
        parse_session_start(message)

    message = valid_start()
    message["conversation"]["initial_voice"] = " "
    with pytest.raises(ProtocolViolation, match="initial_voice"):
        parse_session_start(message)


def test_rejects_non_pcm16_audio() -> None:
    message = valid_start()
    message["audio"]["sample_rate"] = 24000
    with pytest.raises(ProtocolViolation, match="16 kHz"):
        parse_session_start(message)


def test_parses_explicit_audio_and_text_turns() -> None:
    turn = parse_turn_start({"type": "turn.start", "turn_id": "one", "input": "audio"})
    assert turn.input_type == "audio"
    assert parse_input_text(
        {"type": "input.text", "turn_id": "two", "text": "hello"}
    ) == ("two", "hello")
    with pytest.raises(ProtocolViolation, match="4,000"):
        parse_input_text({"type": "input.text", "turn_id": "two", "text": "x" * 4_001})


def test_accepts_only_defined_controls() -> None:
    assert validate_control({"type": "input.end"}) == "input.end"
    assert validate_control({"type": "turn.end", "turn_id": "one"}) == "turn.end"
    assert (
        validate_control({"type": "response.cancel", "response_id": "1"})
        == "response.cancel"
    )
    with pytest.raises(ProtocolViolation, match="Unsupported"):
        validate_control({"type": "input.pause"})
