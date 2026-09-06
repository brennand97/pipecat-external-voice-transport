"""Load administrator-trusted tool declarations from a mounted JSON file."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..session_audit import SessionAuditLog
from ..session_plan import SessionPlanError, ToolNamePattern
from .mcp import MCPServerConfig, MCPToolProvider
from .registry import ToolRegistry
from .script import ScriptToolConfig, ScriptToolProvider


class ToolConfigurationError(ValueError):
    """Raised when the trusted deployment tool configuration is unsafe."""


def create_tool_registry(
    config_path: str,
    *,
    audit: SessionAuditLog | None = None,
    session_id: str = "",
    profile_name: str | None = None,
    requested_tools: tuple[str, ...] | None = None,
) -> ToolRegistry | None:
    """Create a fresh session-scoped registry from a trusted JSON file.

    The file selects fixed MCP endpoints or executable argv vectors. It never
    accepts shell expressions, model-selected commands, or literal environment
    values. ``env_names`` copies only named values already supplied by the
    deployment environment into a stdio MCP child process.
    """
    if not config_path:
        return None
    try:
        document = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise ToolConfigurationError("cannot load trusted tool configuration") from err
    if not isinstance(document, dict):
        raise ToolConfigurationError("tool configuration must be an object")
    _reject_unknown_keys(
        document, {"mcp_servers", "script_tools", "profiles", "default_profile"}
    )
    mcp_servers = _list(document, "mcp_servers")
    script_tools = _list(document, "script_tools")
    profiles = _profiles(document, mcp_servers, script_tools)
    selected_name = (
        profile_name or _optional_string(document, "default_profile") or "default"
    )
    selected = profiles.get(selected_name)
    if selected is None:
        raise ToolConfigurationError("unknown tool profile")
    if requested_tools is not None:
        if not requested_tools or len(set(requested_tools)) != len(requested_tools):
            raise ToolConfigurationError(
                "requested_tools must be a unique non-empty list"
            )
        if any("*" in name for name in requested_tools):
            raise ToolConfigurationError("requested_tools must contain exact names")
    providers = [
        MCPToolProvider(_mcp_config(item))
        for item in mcp_servers
        if _string(_object(item, "mcp server"), "name") in selected[0]
    ] + [
        ScriptToolProvider(_script_config(item))
        for item in script_tools
        if _string(_object(item, "script tool"), "name") in selected[0]
    ]
    return ToolRegistry(
        tuple(providers),
        audit=audit,
        session_id=session_id,
        allowed_patterns=selected[1],
        requested_names=frozenset(requested_tools)
        if requested_tools is not None
        else None,
    )


def _profiles(
    document: dict[str, Any], mcp_servers: list[object], script_tools: list[object]
) -> dict[str, tuple[frozenset[str], tuple[ToolNamePattern, ...]]]:
    """Parse profiles, or synthesize a restrictive legacy default profile."""
    raw = document.get("profiles")
    if raw is None:
        names = frozenset(
            [_string(_object(item, "mcp server"), "name") for item in mcp_servers]
            + [_string(_object(item, "script tool"), "name") for item in script_tools]
        )
        allowed = [
            name
            for item in mcp_servers
            for name in _string_list(
                _object(item, "mcp server").get("allowed_tools", []),
                "allowed_tools",
            )
        ] + [_string(_object(item, "script tool"), "name") for item in script_tools]
        return {"default": (names, _patterns(allowed, "allowed_tools"))}
    if not isinstance(raw, dict) or not raw:
        raise ToolConfigurationError("profiles must be a non-empty object")
    available = {
        *(_string(_object(item, "mcp server"), "name") for item in mcp_servers),
        *(_string(_object(item, "script tool"), "name") for item in script_tools),
    }
    profiles: dict[str, tuple[frozenset[str], tuple[ToolNamePattern, ...]]] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name:
            raise ToolConfigurationError("profile names must be non-empty strings")
        item = _object(value, "profile")
        _reject_unknown_keys(item, {"providers", "allowed_tools"})
        providers = frozenset(_string_list(item.get("providers"), "providers"))
        if not providers or not providers <= available:
            raise ToolConfigurationError("profile references an unknown provider")
        profiles[name] = (
            providers,
            _patterns(
                _string_list(item.get("allowed_tools"), "allowed_tools"),
                "allowed_tools",
            ),
        )
    return profiles


def _patterns(value: list[str], field: str) -> tuple[ToolNamePattern, ...]:
    try:
        patterns = tuple(ToolNamePattern.parse(item) for item in value)
    except SessionPlanError as err:
        raise ToolConfigurationError(
            f"{field} contains an invalid tool pattern"
        ) from err
    if not patterns or len({item.value for item in patterns}) != len(patterns):
        raise ToolConfigurationError(f"{field} must be a unique non-empty list")
    return patterns


def _mcp_config(value: object) -> MCPServerConfig:
    item = _object(value, "mcp server")
    _reject_unknown_keys(
        item,
        {
            "name",
            "transport",
            "url",
            "command",
            "args",
            "env_names",
            "bearer_token_env",
            "allowed_tools",
            "request_timeout_seconds",
            "max_concurrent_calls",
        },
    )
    name = _string(item, "name")
    transport = _string(item, "transport")
    if transport not in {"stdio", "sse", "streamable_http"}:
        raise ToolConfigurationError("MCP transport is not supported")
    url = _optional_string(item, "url")
    command = _optional_string(item, "command")
    args = tuple(_string_list(item.get("args", []), "args"))
    allowed_tool_values = _string_list(item.get("allowed_tools", []), "allowed_tools")
    _patterns(allowed_tool_values, "allowed_tools")
    allowed_tools = frozenset(allowed_tool_values)
    if transport == "stdio":
        if not command or url:
            raise ToolConfigurationError(
                "stdio MCP configuration requires command only"
            )
    elif not url or command:
        raise ToolConfigurationError("network MCP configuration requires url only")
    env = _environment(_string_list(item.get("env_names", []), "env_names"))
    bearer_token_env = _optional_string(item, "bearer_token_env")
    if bearer_token_env and transport == "stdio":
        raise ToolConfigurationError(
            "bearer_token_env is only supported for network MCP configuration"
        )
    bearer_token = (
        _environment([bearer_token_env]).get(bearer_token_env)
        if bearer_token_env
        else None
    )
    return MCPServerConfig(
        name=name,
        transport=transport,  # type: ignore[arg-type]
        url=url,
        command=command,
        args=args,
        env=env or None,
        bearer_token=bearer_token,
        allowed_tools=allowed_tools,
        request_timeout_seconds=_positive_float(item, "request_timeout_seconds", 15.0),
        max_concurrent_calls=_positive_int(item, "max_concurrent_calls", 2),
    )


def _script_config(value: object) -> ScriptToolConfig:
    item = _object(value, "script tool")
    _reject_unknown_keys(
        item, {"name", "command", "timeout_seconds", "max_concurrent_calls"}
    )
    command = tuple(_string_list(item.get("command"), "command"))
    if not command:
        raise ToolConfigurationError("script command must not be empty")
    return ScriptToolConfig(
        name=_string(item, "name"),
        command=command,
        timeout_seconds=_positive_float(item, "timeout_seconds", 15.0),
        max_concurrent_calls=_positive_int(item, "max_concurrent_calls", 2),
    )


def _environment(names: list[str]) -> dict[str, str]:
    missing = [name for name in names if name not in os.environ]
    if missing:
        raise ToolConfigurationError("a configured tool environment variable is absent")
    return {name: os.environ[name] for name in names}


def _list(document: dict[str, Any], key: str) -> list[object]:
    value = document.get(key, [])
    if not isinstance(value, list):
        raise ToolConfigurationError(f"{key} must be an array")
    return value


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ToolConfigurationError(f"{name} must be an object")
    return value


def _string(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise ToolConfigurationError(f"{key} must be a non-empty string")
    return value


def _optional_string(item: dict[str, Any], key: str) -> str | None:
    value = item.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ToolConfigurationError(f"{key} must be a non-empty string when set")
    return value


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ToolConfigurationError(f"{name} must be an array of non-empty strings")
    return value


def _positive_float(item: dict[str, Any], key: str, default: float) -> float:
    value = item.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ToolConfigurationError(f"{key} must be positive")
    return float(value)


def _positive_int(item: dict[str, Any], key: str, default: int) -> int:
    value = item.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ToolConfigurationError(f"{key} must be a positive integer")
    return value


def _reject_unknown_keys(item: dict[str, Any], allowed: set[str]) -> None:
    if item.keys() - allowed:
        raise ToolConfigurationError("tool configuration contains unsupported fields")
