from voice_transport.agent.fake import FakeAgentSession
from voice_transport.config import Settings
from voice_transport.providers import create_agent_session
from voice_transport.providers.openai_realtime import OpenAIRealtimeAgentSession


def test_fake_provider_creates_isolated_agent_session() -> None:
    assert isinstance(create_agent_session(Settings("token")), FakeAgentSession)


def test_openai_provider_creates_isolated_agent_session() -> None:
    settings = Settings(
        transport_token="token",
        realtime_provider="openai_realtime",
        openai_api_key="test-key-not-used",
    )
    assert isinstance(create_agent_session(settings), OpenAIRealtimeAgentSession)
