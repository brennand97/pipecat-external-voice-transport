import json
from datetime import UTC, datetime

from voice_transport.session_audit import SessionAuditLog


async def test_debug_content_log_retains_transcript_and_redacts_credentials(
    tmp_path,
) -> None:
    audit = SessionAuditLog(tmp_path, mode="debug_content", retention_days=7)

    await audit.record(
        "session-1",
        "tool.completed",
        transcript="turn on the kitchen light",
        arguments={"token": "secret", "name": "Kitchen"},
        result={"url": "https://example/audio?token=signed", "value": "done"},
    )

    entries = [
        json.loads(line)
        for line in next(tmp_path.glob("sessions-*.jsonl")).read_text().splitlines()
    ]
    assert entries == [
        {
            "arguments": {"name": "Kitchen", "token": "[REDACTED]"},
            "event": "tool.completed",
            "result": {"url": "https://example/audio?[REDACTED]", "value": "done"},
            "session_id": "session-1",
            "transcript": "turn on the kitchen light",
            "timestamp": entries[0]["timestamp"],
        }
    ]


async def test_metadata_log_omits_content_but_keeps_correlated_lifecycle(
    tmp_path,
) -> None:
    audit = SessionAuditLog(tmp_path, mode="metadata", retention_days=7)

    await audit.record(
        "session-1",
        "user.transcript.final",
        turn_id="turn-1",
        transcript="private speech",
        arguments={"name": "Kitchen"},
    )

    entry = json.loads(next(tmp_path.glob("sessions-*.jsonl")).read_text())
    assert entry["session_id"] == "session-1"
    assert entry["turn_id"] == "turn-1"
    assert entry["event"] == "user.transcript.final"
    assert "transcript" not in entry
    assert "arguments" not in entry


async def test_prune_removes_only_audit_files_older_than_retention(tmp_path) -> None:
    old = tmp_path / "sessions-2026-01-01.jsonl"
    old.write_text("old\n")
    current = tmp_path / "sessions-2026-01-10.jsonl"
    current.write_text("current\n")
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("keep\n")
    audit = SessionAuditLog(tmp_path, mode="metadata", retention_days=7)

    await audit.prune(now=datetime(2026, 1, 10, tzinfo=UTC))

    assert not old.exists()
    assert current.exists()
    assert unrelated.exists()
