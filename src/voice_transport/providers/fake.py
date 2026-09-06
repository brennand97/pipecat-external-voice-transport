"""Deterministic provider for protocol and orchestration tests."""

from __future__ import annotations

from voice_transport.agent.fake import FakeAgentSession
from voice_transport.agent.session import AgentSession

from .base import RealtimeProviderConfig


class FakeRealtimeProvider:
    """Creates deterministic sessions without a cloud-provider dependency."""

    def create_session(self, config: RealtimeProviderConfig) -> AgentSession:
        del config
        return FakeAgentSession()
