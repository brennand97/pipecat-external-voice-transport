from fastapi.testclient import TestClient

from voice_transport.app import create_app
from voice_transport.config import Settings


def start() -> dict:
    return {
        "type": "session.start",
        "protocol_version": 1,
        "session_id": "test-session",
        "satellite": {"entity_id": "assist_satellite.kitchen", "name": "Kitchen"},
        "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1},
        "conversation": {"id": None, "wake_word": None},
    }


def test_health_and_ready() -> None:
    client = TestClient(create_app(Settings("token")))
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}


def test_transport_lifecycle() -> None:
    client = TestClient(create_app(Settings("token")))
    with client.websocket_connect(
        "/transport/v1", headers={"authorization": "Bearer token"}
    ) as websocket:
        websocket.send_json(start())
        assert websocket.receive_json()["type"] == "session.ready"
        websocket.send_bytes(b"\x00\x00")
        websocket.send_json({"type": "input.end"})
        assert websocket.receive_json()["type"] == "assistant.response_started"
        assert websocket.receive_json()["type"] == "assistant.text.final"
        assert websocket.receive_json()["type"] == "assistant.response_finished"
        assert websocket.receive_json()["type"] == "session.finished"
