import pytest

from voice_transport.config import ConfigurationError, Settings


def test_debug_content_audit_mode_requires_a_log_path(monkeypatch) -> None:
    monkeypatch.setenv("EXTERNAL_TRANSPORT_TOKEN", "test")
    monkeypatch.setenv("SESSION_AUDIT_MODE", "debug_content")
    monkeypatch.delenv("SESSION_AUDIT_LOG_PATH", raising=False)

    with pytest.raises(ConfigurationError, match="SESSION_AUDIT_LOG_PATH"):
        Settings.from_environment()


def test_debug_content_audit_mode_loads_seven_day_retention(monkeypatch) -> None:
    monkeypatch.setenv("EXTERNAL_TRANSPORT_TOKEN", "test")
    monkeypatch.setenv("SESSION_AUDIT_MODE", "debug_content")
    monkeypatch.setenv("SESSION_AUDIT_LOG_PATH", "/var/log/voice-transport")
    monkeypatch.setenv("SESSION_AUDIT_RETENTION_DAYS", "7")

    settings = Settings.from_environment()

    assert settings.session_audit_mode == "debug_content"
    assert settings.session_audit_log_path == "/var/log/voice-transport"
    assert settings.session_audit_retention_days == 7
