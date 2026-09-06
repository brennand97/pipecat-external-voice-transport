import json

import pytest

from voice_transport.tools.config import ToolConfigurationError, create_tool_registry
from voice_transport.tools.mcp import MCPToolProvider
from voice_transport.tools.script import ScriptToolProvider


def test_tool_config_creates_only_explicit_trusted_providers(tmp_path) -> None:
    path = tmp_path / "tools.json"
    path.write_text(
        json.dumps(
            {
                "mcp_servers": [
                    {
                        "name": "home-assistant",
                        "transport": "streamable_http",
                        "url": "https://ha.example/mcp",
                        "allowed_tools": ["get_state"],
                    }
                ],
                "script_tools": [
                    {
                        "name": "calendar",
                        "command": ["/usr/local/bin/calendar-tool"],
                    }
                ],
            }
        )
    )

    registry = create_tool_registry(str(path))

    assert registry is not None
    assert isinstance(registry.providers[0], MCPToolProvider)
    assert registry.providers[0].config.allowed_tools == frozenset({"get_state"})
    assert isinstance(registry.providers[1], ScriptToolProvider)
    assert registry.providers[1].config.command == ("/usr/local/bin/calendar-tool",)


def test_tool_config_resolves_network_bearer_token_only_from_environment(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOMEASSISTANT_MCP_TOKEN", "test-token")
    path = tmp_path / "tools.json"
    path.write_text(
        json.dumps(
            {
                "mcp_servers": [
                    {
                        "name": "home-assistant",
                        "transport": "streamable_http",
                        "url": "https://ha.example/api/mcp",
                        "bearer_token_env": "HOMEASSISTANT_MCP_TOKEN",
                        "allowed_tools": ["get_state"],
                    }
                ]
            }
        )
    )

    registry = create_tool_registry(str(path))

    assert registry is not None
    assert registry.providers[0].config.bearer_token == "test-token"


def test_tool_config_rejects_missing_network_bearer_environment(tmp_path) -> None:
    path = tmp_path / "tools.json"
    path.write_text(
        json.dumps(
            {
                "mcp_servers": [
                    {
                        "name": "home-assistant",
                        "transport": "streamable_http",
                        "url": "https://ha.example/api/mcp",
                        "bearer_token_env": "MISSING_HOMEASSISTANT_MCP_TOKEN",
                        "allowed_tools": ["get_state"],
                    }
                ]
            }
        )
    )

    with pytest.raises(ToolConfigurationError, match="environment variable"):
        create_tool_registry(str(path))


def test_tool_config_fails_closed_for_unallowlisted_mcp(tmp_path) -> None:
    path = tmp_path / "tools.json"
    path.write_text(
        json.dumps(
            {
                "mcp_servers": [
                    {
                        "name": "unsafe",
                        "transport": "sse",
                        "url": "https://example.invalid/sse",
                    }
                ]
            }
        )
    )

    with pytest.raises(ToolConfigurationError, match="allowed_tools"):
        create_tool_registry(str(path))
