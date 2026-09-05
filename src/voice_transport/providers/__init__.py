"""Realtime provider implementations and factory."""

from __future__ import annotations

from voice_transport.agent.session import AgentSession
from voice_transport.config import Settings
from voice_transport.tools.config import create_tool_registry

from .base import RealtimeProvider, RealtimeProviderConfig
from .fake import FakeRealtimeProvider

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a concise, helpful voice assistant. "
    "Speak naturally and keep answers brief."
)


def create_agent_session(settings: Settings) -> AgentSession:
    """Build the configured provider session without exposing it to transport code."""
    config = RealtimeProviderConfig(
        system_instruction=DEFAULT_SYSTEM_INSTRUCTION,
        tool_registry=create_tool_registry(settings.trusted_tool_config_path),
    )
    provider: RealtimeProvider
    if settings.realtime_provider == "fake":
        provider = FakeRealtimeProvider()
    elif settings.realtime_provider == "openai_realtime":
        from .openai_realtime import OpenAIRealtimeProvider

        provider = OpenAIRealtimeProvider(
            settings.openai_api_key,
            settings.openai_realtime_model,
        )
    else:  # Settings validation prevents this; retain a defensive boundary.
        raise ValueError(f"unsupported realtime provider: {settings.realtime_provider}")
    return provider.create_session(config)
