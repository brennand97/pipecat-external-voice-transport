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
    realtime_provider: str = "fake"
    openai_api_key: str = ""
    openai_realtime_model: str = "gpt-realtime-mini"
    public_base_url: str = ""
    audio_url_signing_key: str = ""
    audio_url_token_ttl_seconds: int = 60
    max_buffered_output_chunks: int = 64
    audio_stream_write_timeout_seconds: float = 1.0
    trusted_tool_config_path: str = ""

    @classmethod
    def from_environment(cls) -> Settings:
        token = os.environ.get("EXTERNAL_TRANSPORT_TOKEN", "")
        if not token:
            raise ConfigurationError("EXTERNAL_TRANSPORT_TOKEN must be set")
        realtime_provider = os.environ.get("REALTIME_PROVIDER", "fake")
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        openai_realtime_model = os.environ.get(
            "OPENAI_REALTIME_MODEL", "gpt-realtime-mini"
        )
        if realtime_provider not in {"fake", "openai_realtime"}:
            raise ConfigurationError("REALTIME_PROVIDER is not supported")
        if realtime_provider == "openai_realtime" and not openai_api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY must be set for the openai_realtime provider"
            )
        if not openai_realtime_model:
            raise ConfigurationError("OPENAI_REALTIME_MODEL must not be empty")
        public_base_url = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
        audio_url_signing_key = os.environ.get("AUDIO_URL_SIGNING_KEY", "")
        trusted_tool_config_path = os.environ.get("TRUSTED_TOOL_CONFIG_PATH", "")
        if realtime_provider == "openai_realtime" and (
            not public_base_url or not audio_url_signing_key
        ):
            raise ConfigurationError(
                "PUBLIC_BASE_URL and AUDIO_URL_SIGNING_KEY must be set "
                "for openai_realtime"
            )
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
            audio_token_ttl = int(os.environ.get("AUDIO_URL_TOKEN_TTL_SECONDS", "60"))
            max_buffered_output_chunks = int(
                os.environ.get("MAX_BUFFERED_OUTPUT_CHUNKS", "64")
            )
            audio_stream_write_timeout = float(
                os.environ.get("AUDIO_STREAM_WRITE_TIMEOUT_SECONDS", "1")
            )
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
            or audio_token_ttl <= 0
            or max_buffered_output_chunks < 1
            or audio_stream_write_timeout <= 0
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
            realtime_provider,
            openai_api_key,
            openai_realtime_model,
            public_base_url,
            audio_url_signing_key,
            audio_token_ttl,
            max_buffered_output_chunks,
            audio_stream_write_timeout,
            trusted_tool_config_path,
        )
