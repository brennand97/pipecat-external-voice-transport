"""Load administrator-trusted tool declarations from a mounted JSON file."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .mcp import MCPServerConfig, MCPToolProvider
from .registry import ToolRegistry
from .script import ScriptToolConfig, ScriptToolProvider


class ToolConfigurationError(ValueError):
    """Raised when the trusted deployment tool configuration is unsafe."""


def create_tool_registry(config_path: str) -> ToolRegistry | None:
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
    _reject_unknown_keys(document, {"mcp_servers", "script_tools"})
    mcp_servers = _list(document, "mcp_servers")
    script_tools = _list(document, "script_tools")
    providers = [MCPToolProvider(_mcp_config(item)) for item in mcp_servers] + [
        ScriptToolProvider(_script_config(item)) for item in script_tools
    ]
    return ToolRegistry(tuple(providers))


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
    allowed_tools = frozenset(
        _string_list(item.get("allowed_tools", []), "allowed_tools")
    )
    if not allowed_tools:
        raise ToolConfigurationError("MCP allowed_tools must not be empty")
    if transport == "stdio":
        if not command or url:
            raise ToolConfigurationError(
                "stdio MCP configuration requires command only"
            )
    elif not url or command:
        raise ToolConfigurationError("network MCP configuration requires url only")
    env = _environment(_string_list(item.get("env_names", []), "env_names"))
    return MCPServerConfig(
        name=name,
        transport=transport,  # type: ignore[arg-type]
        url=url,
        command=command,
        args=args,
        env=env or None,
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
