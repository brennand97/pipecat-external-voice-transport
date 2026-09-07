from voice_transport.dev_portal import detail_page


def test_detail_page_has_local_search_and_event_type_filters() -> None:
    response = detail_page(
        "session-1",
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "session_id": "session-1",
                "event": "user.transcript.final",
                "transcript": "Turn on the lights",
            },
            {
                "timestamp": "2026-01-01T00:00:01Z",
                "session_id": "session-1",
                "event": "tool.call_started",
                "tool_name": "intent__HassTurnOn",
            },
        ],
    )

    page = response.body.decode()
    assert "id='timeline-search'" in page
    assert "id='event-types'" in page
    assert "user.transcript.final" in page
    assert "tool.call_started" in page
    assert "data-event='user.transcript.final'" in page
    assert "row.textContent.toLowerCase().includes(query)" in page
