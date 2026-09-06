"""Provider-neutral External Transport Protocol v1 validation and messages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 1
PCM_ENCODING = "pcm_s16le"
PCM_SAMPLE_RATE = 16_000
PCM_CHANNELS = 1


class ProtocolViolation(ValueError):
    """A client-visible protocol error with a stable, non-sensitive code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class SessionStart:
    session_id: str
    satellite_entity_id: str | None
    satellite_name: str | None
    client_id: str = ""
    client_kind: str = "satellite"
    conversation_id: str | None = None
    wake_word: str | None = None
    initial_prompt: str | None = None
    initial_voice: str | None = None
    tool_profile: str | None = None
    requested_tools: tuple[str, ...] | None = None
    device_id: str | None = None
    input_modalities: frozenset[str] = frozenset({"audio", "text"})
    output_modalities: frozenset[str] = frozenset({"audio", "text"})


@dataclass(frozen=True, slots=True)
class TurnStart:
    turn_id: str
    input_type: str


def parse_turn_start(message: dict[str, Any]) -> TurnStart:
    if message.get("type") != "turn.start":
        raise ProtocolViolation("invalid_message", "Expected turn.start.")
    turn_id = _required_string(message, "turn_id")
    input_type = message.get("input")
    if input_type not in {"audio", "text"}:
        raise ProtocolViolation(
            "invalid_message", "turn.start input must be audio or text."
        )
    return TurnStart(turn_id, input_type)


def parse_input_text(message: dict[str, Any]) -> tuple[str, str]:
    if message.get("type") != "input.text":
        raise ProtocolViolation("invalid_message", "Expected input.text.")
    turn_id = _required_string(message, "turn_id")
    text = _required_string(message, "text")
    if len(text.encode()) > 4_000:
        raise ProtocolViolation(
            "input_text_too_large", "Text input exceeds 4,000 bytes."
        )
    return turn_id, text


def parse_json_message(payload: str) -> dict[str, Any]:
    import json

    try:
        message = json.loads(payload)
    except json.JSONDecodeError as err:
        raise ProtocolViolation(
            "invalid_json", "Control frames must contain JSON."
        ) from err
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise ProtocolViolation(
            "invalid_message", "A message object with a type is required."
        )
    return message


def parse_session_start(message: dict[str, Any]) -> SessionStart:
    if message.get("type") != "session.start":
        raise ProtocolViolation(
            "invalid_first_message", "The first message must be session.start."
        )
    if message.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolViolation(
            "unsupported_protocol_version", "Only protocol version 1 is supported."
        )
    session_id = _required_string(message, "session_id")
    satellite = message.get("satellite")
    client = message.get("client")
    if not isinstance(satellite, dict) and not isinstance(client, dict):
        raise ProtocolViolation(
            "invalid_message", "session.start requires satellite or client."
        )
    if isinstance(satellite, dict):
        satellite_entity_id = _required_string(satellite, "entity_id")
        satellite_name = _required_string(satellite, "name")
        client_id, client_kind = satellite_entity_id, "satellite"
    else:
        satellite_entity_id = satellite_name = None
        client_id = _required_string(_object(message, "client"), "id")
        client_kind = _required_string(_object(message, "client"), "kind")
    audio = _object(message, "audio")
    conversation = _object(message, "conversation")
    input_modalities = _modalities(
        conversation.get("input_modalities"), {"audio", "text"}
    )
    output_modalities = _modalities(
        conversation.get("output_modalities"), {"audio", "text"}
    )
    if "audio" in input_modalities and (
        audio.get("encoding") != PCM_ENCODING
        or audio.get("sample_rate") != PCM_SAMPLE_RATE
        or audio.get("channels") != PCM_CHANNELS
    ):
        raise ProtocolViolation(
            "unsupported_audio_format",
            "Audio must be 16 kHz mono pcm_s16le.",
        )
    conversation_id = conversation.get("id")
    wake_word = conversation.get("wake_word")
    initial_prompt = conversation.get("initial_prompt")
    initial_voice = conversation.get("initial_voice")
    tool_profile = conversation.get("profile")
    requested_tools = conversation.get("requested_tools")
    device_id = conversation.get("device_id")
    if conversation_id is not None and not isinstance(conversation_id, str):
        raise ProtocolViolation(
            "invalid_message", "conversation.id must be a string or null."
        )
    if wake_word is not None and not isinstance(wake_word, str):
        raise ProtocolViolation(
            "invalid_message", "conversation.wake_word must be a string or null."
        )
    if device_id is not None and (
        not isinstance(device_id, str) or not device_id.strip()
    ):
        raise ProtocolViolation(
            "invalid_message",
            "conversation.device_id must be a non-empty string or null.",
        )
    if tool_profile is not None and (
        not isinstance(tool_profile, str)
        or not tool_profile.strip()
        or len(tool_profile) > 128
    ):
        raise ProtocolViolation(
            "invalid_message",
            "conversation.profile must be a short non-empty string or null.",
        )
    if requested_tools is not None:
        if (
            not isinstance(requested_tools, list)
            or not requested_tools
            or len(requested_tools) > 128
            or not all(
                isinstance(name, str) and name and "*" not in name
                for name in requested_tools
            )
            or len(set(requested_tools)) != len(requested_tools)
        ):
            raise ProtocolViolation(
                "invalid_message",
                "conversation.requested_tools must be unique exact names.",
            )
    if initial_prompt is not None:
        if not isinstance(initial_prompt, str) or not initial_prompt.strip():
            raise ProtocolViolation(
                "invalid_message",
                "conversation.initial_prompt must be a non-empty string or null.",
            )
        if len(initial_prompt.encode()) > 16_000:
            raise ProtocolViolation(
                "initial_prompt_too_large",
                "conversation.initial_prompt exceeds 16,000 bytes.",
            )
    if initial_voice is not None:
        if not isinstance(initial_voice, str) or not initial_voice.strip():
            raise ProtocolViolation(
                "invalid_message",
                "conversation.initial_voice must be a non-empty string or null.",
            )
        if len(initial_voice.encode()) > 128:
            raise ProtocolViolation(
                "initial_voice_too_large",
                "conversation.initial_voice exceeds 128 bytes.",
            )
    return SessionStart(
        session_id=session_id,
        satellite_entity_id=satellite_entity_id,
        satellite_name=satellite_name,
        client_id=client_id,
        client_kind=client_kind,
        conversation_id=conversation_id,
        wake_word=wake_word,
        initial_prompt=initial_prompt,
        initial_voice=initial_voice,
        tool_profile=tool_profile,
        requested_tools=tuple(requested_tools) if requested_tools is not None else None,
        device_id=device_id,
        input_modalities=frozenset(input_modalities),
        output_modalities=frozenset(output_modalities),
    )


def validate_control(message: dict[str, Any]) -> str:
    message_type = message["type"]
    if message_type not in {
        "input.end",
        "turn.end",
        "response.cancel",
        "session.cancel",
    }:
        raise ProtocolViolation(
            "unsupported_message_type", f"Unsupported message type: {message_type}."
        )
    if message_type in {"turn.end", "response.cancel"}:
        _required_string(
            message, "turn_id" if message_type == "turn.end" else "response_id"
        )
    if (
        message_type in {"response.cancel", "session.cancel"}
        and "reason" in message
        and not isinstance(message["reason"], str)
    ):
        raise ProtocolViolation(
            "invalid_message", "session.cancel reason must be a string."
        )
    return message_type


def ready_message(session_id: str) -> dict[str, Any]:
    return {
        "type": "session.ready",
        "session_id": session_id,
        "capabilities": {
            "transcription": True,
            "text_input": True,
            "streaming_audio_url": True,
            "interruptions": True,
            "conversation_continuation": True,
        },
    }


def error_message(
    error: ProtocolViolation, session_id: str | None = None
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "type": "error",
        "code": error.code,
        "message": error.message,
    }
    if session_id:
        message["session_id"] = session_id
    return message


def _modalities(value: object, default: set[str]) -> set[str]:
    if value is None:
        return default
    if (
        not isinstance(value, list)
        or not value
        or len(value) > 2
        or not all(
            isinstance(item, str) and item in {"audio", "text"} for item in value
        )
    ):
        raise ProtocolViolation(
            "invalid_message", "conversation modalities must be audio and/or text."
        )
    return set(value)


def _required_string(message: dict[str, Any], key: str) -> str:
    value = message.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProtocolViolation("invalid_message", f"{key} must be a non-empty string.")
    return value


def _object(message: dict[str, Any], key: str) -> dict[str, Any]:
    value = message.get(key)
    if not isinstance(value, dict):
        raise ProtocolViolation("invalid_message", f"{key} must be an object.")
    return value
