"""The sole model-facing bridge from Pipecat function calls to async tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from voice_transport.agent.session import AgentEvent

from .registry import ToolRegistry


class PipecatToolBridge:
    """Expose registry tools as Pipecat async function schemas and handlers."""

    def __init__(
        self,
        registry: ToolRegistry,
        emit_event: Callable[[AgentEvent], Awaitable[None]] | None = None,
    ) -> None:
        self._registry = registry
        self._emit_event = emit_event

    async def function_schemas(self) -> list[Any]:
        """Discover tools then create Pipecat handlers that await each result."""
        from pipecat.adapters.schemas.function_schema import FunctionSchema

        schemas = []
        for tool in await self._registry.discover():
            schema = tool.input_schema
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            if not isinstance(properties, dict) or not isinstance(required, list):
                raise ValueError(f"tool {tool.name} has an invalid object schema")

            async def handler(params, tool_name: str = tool.name) -> None:
                arguments = dict(params.arguments)
                if self._emit_event is not None:
                    await self._emit_event(
                        AgentEvent(
                            "assistant.tool_call_started",
                            tool_name=tool_name,
                            tool_arguments=arguments,
                        )
                    )
                try:
                    result = await self._registry.call(tool_name, arguments)
                    payload = {"content": result.content, "is_error": result.is_error}
                    await params.result_callback(payload)
                    if self._emit_event is not None:
                        await self._emit_event(
                            AgentEvent(
                                "assistant.tool_call_finished",
                                tool_name=tool_name,
                                tool_arguments=arguments,
                                tool_result=result.content,
                                is_error=result.is_error,
                            )
                        )
                except Exception:  # noqa: BLE001 - normalize tool boundary failures
                    payload = {
                        "content": [{"type": "text", "text": "Tool execution failed."}],
                        "is_error": True,
                    }
                    await params.result_callback(payload)
                    if self._emit_event is not None:
                        await self._emit_event(
                            AgentEvent(
                                "assistant.tool_call_finished",
                                tool_name=tool_name,
                                tool_arguments=arguments,
                                tool_result=payload["content"],
                                is_error=True,
                            )
                        )

            schemas.append(
                FunctionSchema(
                    name=tool.name,
                    description=tool.description,
                    properties=properties,
                    required=[item for item in required if isinstance(item, str)],
                    handler=handler,
                )
            )
        return schemas

    async def close(self) -> None:
        await self._registry.close()
