"""Provider-neutral contract for realtime audio-to-audio backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from voice_transport.agent.session import AgentSession


@dataclass(frozen=True, slots=True)
class RealtimeProviderConfig:
    """Provider-independent inputs for one isolated agent session."""

    system_instruction: str
    input_sample_rate: int = 16_000
    input_channels: int = 1


class RealtimeProvider(Protocol):
    """Creates isolated agent sessions for a realtime audio backend."""

    def create_session(self, config: RealtimeProviderConfig) -> AgentSession: ...
