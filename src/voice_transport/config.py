"""Runtime configuration loaded exclusively from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigurationError(ValueError):
    """Raised when required or unsafe service configuration is supplied."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Non-secret limits and the required transport bearer token."""

    transport_token: str
    max_concurrent_sessions: int = 2
    max_audio_frame_bytes: int = 32_000
    max_input_bytes: int = 9_600_000
    max_buffered_audio_frames: int = 64
    session_start_timeout_seconds: float = 5.0
    input_idle_timeout_seconds: float = 20.0
    max_session_seconds: float = 300.0

    @classmethod
    def from_environment(cls) -> Settings:
        token = os.environ.get("EXTERNAL_TRANSPORT_TOKEN", "")
        if not token:
            raise ConfigurationError("EXTERNAL_TRANSPORT_TOKEN must be set")
        try:
            max_sessions = int(os.environ.get("MAX_CONCURRENT_SESSIONS", "2"))
            max_frame = int(os.environ.get("MAX_AUDIO_FRAME_BYTES", "32000"))
            max_input = int(os.environ.get("MAX_INPUT_BYTES", "9600000"))
            max_buffered_frames = int(os.environ.get("MAX_BUFFERED_AUDIO_FRAMES", "64"))
            session_start_timeout = float(
                os.environ.get("SESSION_START_TIMEOUT_SECONDS", "5")
            )
            input_idle_timeout = float(
                os.environ.get("INPUT_IDLE_TIMEOUT_SECONDS", "20")
            )
            max_session_seconds = float(os.environ.get("MAX_SESSION_SECONDS", "300"))
        except ValueError as err:
            raise ConfigurationError("transport limits must be numeric") from err
        if (
            max_sessions < 1
            or max_frame < 2
            or max_input < max_frame
            or max_buffered_frames < 1
            or session_start_timeout <= 0
            or input_idle_timeout <= 0
            or max_session_seconds <= 0
        ):
            raise ConfigurationError("transport limits are outside safe bounds")
        return cls(
            token,
            max_sessions,
            max_frame,
            max_input,
            max_buffered_frames,
            session_start_timeout,
            input_idle_timeout,
            max_session_seconds,
        )
