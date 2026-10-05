"""Black-box transport acceptance against the real test container.

Run using the shared testbed venv. The run manifest/credentials never leave disk.
"""

import json
import sys
import urllib.request
from contextlib import closing
from pathlib import Path

import websocket


def main(run_file):
    run = json.loads(Path(run_file).read_text())
    credentials = json.loads((Path(run["directory"]) / "credentials.json").read_text())
    token = credentials["transport_token"]
    base = run["transport_url"]
    session = {
        "type": "session.start",
        "protocol_version": 1,
        "session_id": "blackbox-test",
        "satellite": {
            "entity_id": "assist_satellite.e2e_browser",
            "name": "E2E Browser",
        },
        "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1},
        "conversation": {"id": None, "wake_word": None},
    }
    try:
        connection = websocket.create_connection(
            base.replace("http", "ws", 1) + "/transport/v1",
            timeout=5,
            header={"Authorization": "Bearer invalid"},
        )
    except websocket.WebSocketBadStatusException as exc:
        assert exc.status_code == 403
    else:
        connection.close()
        raise AssertionError("Transport accepted invalid token")
    with closing(
        websocket.create_connection(
            base.replace("http", "ws", 1) + "/transport/v1",
            timeout=5,
            header={"Authorization": "Bearer " + token},
        )
    ) as connection:

        def send(data):
            return connection.send(json.dumps(data))

        send(session)
        assert json.loads(connection.recv())["type"] == "session.ready"
        for index, modality in enumerate(["text", "audio"]):
            turn = f"turn-{index}"
            send({"type": "turn.start", "turn_id": turn, "input": modality})
            if modality == "text":
                send({"type": "input.text", "turn_id": turn, "text": "hello"})
            else:
                connection.send_binary(b"\x01\x00" * 1600)
            send({"type": "turn.end", "turn_id": turn})
            events = []
            for _ in range(20):
                event = json.loads(connection.recv())
                events.append(event)
                if event["type"] == "assistant.response_finished":
                    break
            assert events[-1]["type"] == "assistant.response_finished"
            assert all(e.get("turn_id") == turn for e in events)
            audio = next(e for e in events if e["type"] == "assistant.audio")
            url = audio.get("url") or audio.get("audio", {}).get("url")
            assert url, f"Audio URL absent; keys={list(audio)}"
            # Read only through this run's fixed loopback transport ingress.
            from urllib.parse import urlsplit

            parsed = urlsplit(url)
            with urllib.request.urlopen(
                base + parsed.path + "?" + parsed.query, timeout=5
            ) as response:
                wav = response.read()
            assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
            assert len(wav) > 28800
        # Browser capture holds the audio turn open until provider response.
        send({"type": "turn.start", "turn_id": "open-audio", "input": "audio"})
        connection.send_binary(b"\x01\x00" * 16000)
        connection.send_binary(b"\x01\x00" * 16000)
        events = []
        for _ in range(20):
            event = json.loads(connection.recv())
            events.append(event)
            if event["type"] == "assistant.response_finished":
                break
        assert events[-1]["type"] == "assistant.response_finished"
        assert any(e["type"] == "user.transcript.final" for e in events)
        send({"type": "turn.end", "turn_id": "open-audio"})
        send({"type": "session.cancel"})
        assert json.loads(connection.recv())["type"] == "session.finished"
    if run.get("tool_status") == "failed":
        raise AssertionError(
            "Test HA MCP setup failed: " + run.get("tool_error", "unknown")
        )
    if run.get("tool_status") == "ready":
        from ha_testbed.api import HA

        ha = HA(run["url"], credentials["tokens"]["access_token"])
        ha.request(
            "/api/services/input_boolean/turn_off",
            {"entity_id": "input_boolean.testbed_light"},
        )
        session["session_id"] = "blackbox-tool"
        with closing(
            websocket.create_connection(
                base.replace("http", "ws", 1) + "/transport/v1",
                timeout=15,
                header={"Authorization": "Bearer " + token},
            )
        ) as connection:

            def send(data):
                return connection.send(json.dumps(data))

            send(session)
            assert json.loads(connection.recv())["type"] == "session.ready"
            send({"type": "turn.start", "turn_id": "tool", "input": "text"})
            send({"type": "input.text", "turn_id": "tool", "text": "test tool"})
            send({"type": "turn.end", "turn_id": "tool"})
            events = []
            for _ in range(20):
                event = json.loads(connection.recv())
                events.append(event)
                if event["type"] == "assistant.response_finished":
                    break
            assert events[-1]["type"] == "assistant.response_finished"
            tool_events = [e for e in events if "tool_call" in e["type"]]
            assert any(
                e["type"] == "assistant.tool_call_finished" and not e.get("is_error")
                for e in events
            ), str(
                [
                    {k: e.get(k) for k in ["type", "tool_name", "is_error", "result"]}
                    for e in tool_events
                ]
            )
            assert ha.request("/api/states/light.testbed_light")["state"] == "on"
            send({"type": "session.cancel"})
            assert json.loads(connection.recv())["type"] == "session.finished"
        print("PASS tools: real registry -> test HA MCP -> synthetic light")
    req = urllib.request.Request(
        base + "/e2e/stats", headers={"Authorization": "Bearer " + token}
    )
    with urllib.request.urlopen(req) as response:
        stats = json.load(response)
    assert stats["input_bytes"] >= 3200 and stats["turns"] >= 2 and stats["closed"] >= 1
    print(
        "PASS transport: invalid-token rejection, persistent text/audio turns, "
        "valid streamed WAV, cleanup"
    )


if __name__ == "__main__":
    main(sys.argv[1])
