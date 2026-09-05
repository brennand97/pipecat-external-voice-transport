import pytest

from voice_transport.protocol import ProtocolViolation, SessionStart
from voice_transport.session import Session, SessionState


@pytest.fixture
def session() -> Session:
    return Session(
        SessionStart("id", "assist_satellite.kitchen", "Kitchen", None, None), 8
    )


def test_audio_accounting_and_lifecycle(session: Session) -> None:
    session.mark_ready()
    session.add_audio(b"\x00\x00\x01\x00", max_frame_bytes=4)
    assert session.state is SessionState.LISTENING
    assert session.input_bytes == 4
    session.end_input()
    session.finish()
    assert session.state is SessionState.FINISHED


@pytest.mark.parametrize("frame", [b"", b"\x00"])
def test_rejects_invalid_pcm_frames(session: Session, frame: bytes) -> None:
    session.mark_ready()
    with pytest.raises(ProtocolViolation, match="even"):
        session.add_audio(frame, max_frame_bytes=4)


def test_enforces_input_limit(session: Session) -> None:
    session.mark_ready()
    with pytest.raises(ProtocolViolation, match="limit"):
        session.add_audio(b"\x00" * 10, max_frame_bytes=10)
