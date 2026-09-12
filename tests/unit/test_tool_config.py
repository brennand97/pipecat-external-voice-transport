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


def test_named_profile_parses_server_owned_audio_input(tmp_path) -> None:
    path = tmp_path / "tools.json"
    path.write_text(
        json.dumps(
            {
                "profiles": {
                    "far-field": {
                        "providers": [],
                        "allowed_tools": ["transport__Calculate"],
                        "audio_input": {
                            "enhancer": "gtcrn",
                            "provider_noise_reduction": "far_field",
                        },
                    }
                }
            }
        )
    )
    registry = create_tool_registry(str(path), profile_name="far-field")
    assert registry is not None
    assert registry.audio_input.enhancer == "gtcrn"
    assert registry.audio_input.provider_noise_reduction == "far_field"


def test_named_profile_selects_declared_provider_and_exact_client_subset(
    tmp_path,
) -> None:
    path = tmp_path / "tools.json"
    path.write_text(
        json.dumps(
            {
                "mcp_servers": [
                    {
                        "name": "home",
                        "transport": "sse",
                        "url": "https://ha.example/mcp",
                        "allowed_tools": ["intent__HassTurnOn"],
                    },
                    {
                        "name": "other",
                        "transport": "sse",
                        "url": "https://other.example/mcp",
                        "allowed_tools": ["other"],
                    },
                ],
                "profiles": {
                    "home-read-only": {
                        "providers": ["home"],
                        "allowed_tools": ["intent__Hass*"],
                    }
                },
                "default_profile": "home-read-only",
            }
        )
    )

    registry = create_tool_registry(
        str(path),
        profile_name="home-read-only",
        requested_tools=("intent__HassTurnOn",),
    )

    # The requested MCP provider plus the always-available local calculator.
    assert len(registry.providers) == 2
    assert registry.requested_names == frozenset({"intent__HassTurnOn"})
    assert registry.allowed_patterns[0].value == "intent__Hass*"


def test_context_required_tool_is_disabled_without_device_context(tmp_path) -> None:
    path = tmp_path / "tools.json"
    path.write_text(
        json.dumps(
            {
                "mcp_servers": [
                    {
                        "name": "home",
                        "transport": "sse",
                        "url": "https://ha.example/mcp",
                        "allowed_tools": ["voice_satellite__StartTimer"],
                    }
                ],
                "profiles": {
                    "home": {
                        "providers": ["home"],
                        "allowed_tools": ["voice_satellite__StartTimer"],
                        "context_injections": {
                            "voice_satellite__StartTimer": {
                                "device_id": "home_assistant_device_id"
                            }
                        },
                    }
                },
            }
        )
    )

    generic = create_tool_registry(str(path), profile_name="home")
    physical = create_tool_registry(
        str(path),
        profile_name="home",
        context_values={"home_assistant_device_id": "device-1"},
    )

    assert generic.disabled_tool_names == frozenset({"voice_satellite__StartTimer"})
    assert physical.context_injections == {
        "voice_satellite__StartTimer": {"device_id": "device-1"}
    }


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
