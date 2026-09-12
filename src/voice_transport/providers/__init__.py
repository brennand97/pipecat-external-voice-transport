"""Realtime provider implementations and factory."""

from __future__ import annotations

import asyncio

from voice_transport.agent.session import AgentSession
from voice_transport.audio_enhancement import AudioInputEnhancementConfig
from voice_transport.config import Settings
from voice_transport.session_audit import SessionAuditLog
from voice_transport.tools.config import create_tool_registry

from .base import RealtimeProvider, RealtimeProviderConfig
from .fake import FakeRealtimeProvider

DEFAULT_SYSTEM_INSTRUCTION = (
    "Your name is Reginold. You are a concise, helpful voice assistant. "
    "Speak naturally and keep answers brief. When a user asks to inspect or control "
    "their connected home, use the available tools. Before every required tool result "
    "is available, emit no assistant text or audio. Never say that you are checking, "
    "looking up, pulling, or using a tool. After successful results, answer directly "
    "in one short sentence. Never claim that you read state or performed an action "
    "unless the corresponding tool call succeeded; if it fails, briefly explain that."
)


def prepare_provider(settings: Settings) -> None:
    """Load the configured provider module during application construction."""
    if settings.realtime_provider == "openai_realtime":
        from . import openai_realtime  # noqa: F401


def create_agent_session(
    settings: Settings,
    *,
    audit: SessionAuditLog | None = None,
    session_id: str = "",
    initial_prompt: str | None = None,
    initial_voice: str | None = None,
    tool_profile: str | None = None,
    requested_tools: tuple[str, ...] | None = None,
    input_modalities: frozenset[str] = frozenset({"audio", "text"}),
    output_modalities: frozenset[str] = frozenset({"audio", "text"}),
    home_assistant_device_id: str | None = None,
    session_end_event: asyncio.Event | None = None,
) -> AgentSession:
    """Build the configured provider session without exposing it to transport code."""
    tool_registry = create_tool_registry(
        settings.trusted_tool_config_path,
        audit=audit,
        session_id=session_id,
        profile_name=tool_profile,
        requested_tools=requested_tools,
        context_values=(
            {"home_assistant_device_id": home_assistant_device_id}
            if home_assistant_device_id is not None
            else None
        ),
        session_end_event=session_end_event,
    )
    config = RealtimeProviderConfig(
        system_instruction=initial_prompt or DEFAULT_SYSTEM_INSTRUCTION,
        tool_registry=tool_registry,
        audio_input=(
            tool_registry.audio_input
            if tool_registry is not None
            else AudioInputEnhancementConfig()
        ),
        gtcrn_model_path=settings.gtcrn_model_path,
        output_voice=initial_voice,
        input_modalities=input_modalities,
        output_modalities=output_modalities,
        input_transcription_language=settings.openai_input_transcription_language,
    )
    provider: RealtimeProvider
    if settings.realtime_provider == "fake":
        provider = FakeRealtimeProvider()
    elif settings.realtime_provider == "openai_realtime":
        from .openai_realtime import OpenAIRealtimeProvider

        provider = OpenAIRealtimeProvider(
            settings.openai_api_key,
            settings.openai_realtime_model,
            settings.openai_realtime_voice,
        )
    else:  # Settings validation prevents this; retain a defensive boundary.
        raise ValueError(f"unsupported realtime provider: {settings.realtime_provider}")
    return provider.create_session(config)
