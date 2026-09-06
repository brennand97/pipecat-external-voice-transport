import pytest

pytest.importorskip("pipecat")

from voice_transport.providers.base import RealtimeProviderConfig
from voice_transport.providers.openai_realtime import _session_properties


def test_text_only_session_does_not_advertise_or_configure_audio() -> None:
    properties = _session_properties(
        RealtimeProviderConfig(
            system_instruction="Brief.",
            input_modalities=frozenset({"text"}),
            output_modalities=frozenset({"text"}),
        ),
        "gpt-realtime-mini",
        "marin",
        [],
    )

    assert properties.output_modalities == ["text"]
    assert properties.audio is None


def test_satellite_session_configures_input_transcription_and_output_voice() -> None:
    properties = _session_properties(
        RealtimeProviderConfig(system_instruction="Brief."),
        "gpt-realtime-mini",
        "marin",
        [],
    )

    assert properties.output_modalities == ["audio", "text"]
    assert properties.audio.input.transcription is not None
    assert properties.audio.output.voice == "marin"
