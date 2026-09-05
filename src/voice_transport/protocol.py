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
    satellite_entity_id: str
    satellite_name: str
    conversation_id: str | None
    wake_word: str | None


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
    satellite = _object(message, "satellite")
    audio = _object(message, "audio")
    conversation = _object(message, "conversation")
    if (
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
    if conversation_id is not None and not isinstance(conversation_id, str):
        raise ProtocolViolation(
            "invalid_message", "conversation.id must be a string or null."
        )
    if wake_word is not None and not isinstance(wake_word, str):
        raise ProtocolViolation(
            "invalid_message", "conversation.wake_word must be a string or null."
        )
    return SessionStart(
        session_id=session_id,
        satellite_entity_id=_required_string(satellite, "entity_id"),
        satellite_name=_required_string(satellite, "name"),
        conversation_id=conversation_id,
        wake_word=wake_word,
    )


def validate_control(message: dict[str, Any]) -> str:
    message_type = message["type"]
    if message_type not in {"input.end", "session.cancel"}:
        raise ProtocolViolation(
            "unsupported_message_type", f"Unsupported message type: {message_type}."
        )
    if (
        message_type == "session.cancel"
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
            "transcription": False,
            "streaming_audio_url": False,
            "interruptions": False,
            "conversation_continuation": False,
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
