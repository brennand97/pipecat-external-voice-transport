from voice_transport.tool_events import public_arguments, public_result


def test_tool_event_payloads_redact_sensitive_values_and_signed_urls() -> None:
    value, truncated = public_arguments(
        {
            "token": "secret",
            "nested": {"api_key": "secret"},
            "url": "https://example.test/audio?token=secret",
        }
    )

    assert not truncated
    assert value == {
        "token": "[redacted]",
        "nested": {"api_key": "[redacted]"},
        "url": "https://example.test/audio",
    }


def test_oversized_tool_event_payloads_are_valid_bounded_json_shapes() -> None:
    arguments, arguments_truncated = public_arguments({"text": "x" * 5_000})
    result, result_truncated = public_result([{"type": "text", "text": "x" * 9_000}])

    assert arguments_truncated
    assert arguments == {"_truncated": True}
    assert result_truncated
    assert result == [{"type": "text", "text": "[tool result truncated]"}]
