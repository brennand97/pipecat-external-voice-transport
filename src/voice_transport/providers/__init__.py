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
    "Your name is Reginald. You are a concise, helpful voice assistant. "
    "Speak naturally and keep answers brief. When a user asks to inspect or control "
    "their connected home, use the available tools. Before every required tool result "
    "is available, emit no assistant text or audio. Never say that you are checking, "
    "looking up, pulling, or using a tool. After successful results, answer directly "
    "in one short sentence. Never claim that you read state or performed an action "
    "unless the corresponding tool call succeeded; if it fails, briefly explain that. "
    "Kitchen timers are important persistent state, not conversational memories. "
    "Use the current satellite's timer tools; never invent a timer, its remaining "
    "time, or a successful change. Read current timer status before reporting "
    "remaining time or resolving an ambiguous request. Use atomic timer tools "
    "for named changes and trust their fresh returned previous/current state, "
    "not earlier dialogue. Remaining-time answers use seconds_remaining, not "
    "total duration; say approximately if rounding. After mutations acknowledge "
    "the name and change without restating remaining time unless asked. "
    "Extend/shorten are relative adjustments, not "
    "replacement durations. For an unnamed timer command, list ALL current "
    "timers first; if more than one exists, ask which named timer before any "
    "mutation. Never choose the most recent, shortest, or longest timer from "
    "conversation history. Do not guess or cancel all. Preserve unrelated timers "
    "and their expiry alerts. "
    "Ending a conversation, interrupting speech, or stopping music must never "
    "cancel timers or dismiss their alerts. Use transport__EndSession only when "
    "the user explicitly asks to end this conversation, never for stop-timer or "
    "stop-music requests. Treat indistinct speech and background conversations "
    "as uncertain input: ask for clarification rather than perform an ambiguous "
    "device or timer action."
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
    prompt_append: str | None = None,
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
    instruction = initial_prompt or DEFAULT_SYSTEM_INSTRUCTION
    if prompt_append and prompt_append.strip():
        instruction += (
            "\n\nAdditional context and preferences:\n" + prompt_append.strip()
        )
    config = RealtimeProviderConfig(
        system_instruction=instruction,
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
