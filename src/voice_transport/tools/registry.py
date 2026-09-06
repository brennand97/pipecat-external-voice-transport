"""Concurrent discovery and dispatch across trusted async tool providers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ..session_audit import SessionAuditLog
from .base import AsyncToolProvider, ToolDefinition, ToolResult


@dataclass(slots=True)
class ToolRegistry:
    """A session-scoped registry; duplicate model-facing names fail closed."""

    providers: tuple[AsyncToolProvider, ...]
    audit: SessionAuditLog | None = None
    session_id: str = ""
    _tools: dict[str, AsyncToolProvider] = field(default_factory=dict)
    _definitions: list[ToolDefinition] = field(default_factory=list)
    _ready: bool = False

    async def discover(self) -> list[ToolDefinition]:
        if self._ready:
            return list(self._definitions)
        # MCP streamable HTTP contexts are task-affine: discovery opens the
        # context and later session cleanup closes it. Avoid ``gather()``,
        # which would enter each context in a child task.
        discovered = [await provider.list_tools() for provider in self.providers]
        for provider, tools in zip(self.providers, discovered, strict=True):
            for tool in tools:
                if tool.name in self._tools:
                    raise ValueError(f"duplicate tool name: {tool.name}")
                self._tools[tool.name] = provider
                self._definitions.append(tool)
        self._ready = True
        await self._record(
            "tools.discovered", tools=[tool.name for tool in self._definitions]
        )
        return list(self._definitions)

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if not self._ready:
            await self.discover()
        provider = self._tools.get(name)
        if provider is None:
            return ToolResult(
                content=[{"type": "text", "text": f"Unknown tool: {name}"}],
                is_error=True,
            )
        definition = next(tool for tool in self._definitions if tool.name == name)
        validation_error = _validate_arguments(definition.input_schema, arguments)
        if validation_error is not None:
            await self._record(
                "tool.validation_rejected",
                tool_name=name,
                arguments=arguments,
                error=validation_error,
            )
            return ToolResult(
                content=[
                    {"type": "text", "text": f"Invalid arguments: {validation_error}"}
                ],
                is_error=True,
            )
        await self._record("tool.call_started", tool_name=name, arguments=arguments)
        try:
            result = await provider.call_tool(name, arguments)
        except Exception as err:
            await self._record(
                "tool.call_failed", tool_name=name, arguments=arguments, error=str(err)
            )
            raise
        await self._record(
            "tool.call_finished",
            tool_name=name,
            arguments=arguments,
            result=result.content,
            is_error=result.is_error,
        )
        return result

    async def close(self) -> None:
        # Streamable HTTP MCP contexts are task-affine. ``gather`` creates a
        # child task for each close and triggers AnyIO cancel-scope failures,
        # so preserve the caller task while retaining a per-provider bound.
        for provider in self.providers:
            try:
                async with asyncio.timeout(3):
                    await provider.close()
            except TimeoutError:
                pass

    async def _record(self, event: str, **fields: Any) -> None:
        if self.audit is not None and self.session_id:
            await self.audit.record(self.session_id, event, **fields)


def _validate_arguments(
    schema: dict[str, Any], arguments: dict[str, Any]
) -> str | None:
    """Validate the object-schema subset exposed to the model before I/O.

    MCP servers remain the final authority. This local boundary prevents an LLM
    from sending clearly malformed calls over the network and returns a useful,
    retryable tool result instead of converting a model mistake into a session
    failure.
    """
    if not isinstance(arguments, dict):
        return "arguments must be an object."
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        return "tool schema is not a valid object schema."
    for key in required:
        if isinstance(key, str) and key not in arguments:
            return f"missing required property '{key}'."
    if schema.get("additionalProperties") is False:
        for key in arguments:
            if key not in properties:
                return f"unknown property '{key}'."
    for key, value in arguments.items():
        property_schema = properties.get(key)
        if not isinstance(property_schema, dict):
            continue
        error = _validate_value(key, property_schema, value)
        if error is not None:
            return error
    return None


def _validate_value(name: str, schema: dict[str, Any], value: Any) -> str | None:
    expected = schema.get("type")
    valid_types = {
        "string": lambda value: isinstance(value, str),
        "boolean": lambda value: isinstance(value, bool),
        "number": lambda value: (
            isinstance(value, (int, float)) and not isinstance(value, bool)
        ),
        "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "object": lambda value: isinstance(value, dict),
        "array": lambda value: isinstance(value, list),
        "null": lambda value: value is None,
    }
    if (
        isinstance(expected, str)
        and expected in valid_types
        and not valid_types[expected](value)
    ):
        return f"property '{name}' must be a {expected}."
    allowed = schema.get("enum")
    if isinstance(allowed, list) and value not in allowed:
        return f"property '{name}' must be one of the allowed values."
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            return f"property '{name}' must be >= {minimum}."
        if isinstance(maximum, (int, float)) and value > maximum:
            return f"property '{name}' must be <= {maximum}."
    return None
