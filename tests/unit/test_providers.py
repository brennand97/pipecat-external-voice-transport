from voice_transport.agent.fake import FakeAgentSession
from voice_transport.config import Settings
from voice_transport.providers import DEFAULT_SYSTEM_INSTRUCTION, create_agent_session
from voice_transport.providers.openai_realtime import OpenAIRealtimeAgentSession


def test_default_instruction_names_the_assistant_and_requires_silent_home_tools() -> (
    None
):
    assert "Your name is Reginald." in DEFAULT_SYSTEM_INSTRUCTION
    assert "emit no assistant text or audio" in DEFAULT_SYSTEM_INSTRUCTION


def test_fake_provider_creates_isolated_agent_session() -> None:
    assert isinstance(create_agent_session(Settings("token")), FakeAgentSession)


def test_openai_provider_creates_isolated_agent_session() -> None:
    settings = Settings(
        transport_token="token",
        realtime_provider="openai_realtime",
        openai_api_key="test-key-not-used",
        openai_realtime_voice="cedar",
    )
    session = create_agent_session(
        settings,
        initial_prompt="Reply in one short sentence.",
        initial_voice="ballad",
    )
    assert isinstance(session, OpenAIRealtimeAgentSession)
    assert session._voice == "ballad"
    assert session._config.system_instruction == "Reply in one short sentence."


def test_prompt_append_keeps_defaults_and_is_session_local():
    settings = Settings(
        transport_token="token",
        realtime_provider="openai_realtime",
        openai_api_key="test-key-not-used",
    )
    session = create_agent_session(settings, prompt_append="  Personal context  ")
    assert session._config.system_instruction == (
        DEFAULT_SYSTEM_INSTRUCTION
        + "\n\nAdditional context and preferences:\nPersonal context"
    )
    other = create_agent_session(settings)
    assert other._config.system_instruction == DEFAULT_SYSTEM_INSTRUCTION
    overridden = create_agent_session(
        settings, initial_prompt="Existing override", prompt_append="Personal"
    )
    assert overridden._config.system_instruction == (
        "Existing override\n\nAdditional context and preferences:\nPersonal"
    )
