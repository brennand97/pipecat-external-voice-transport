"""The sole model-facing bridge from Pipecat function calls to async tools."""

from __future__ import annotations

from typing import Any

from .registry import ToolRegistry


class PipecatToolBridge:
    """Expose registry tools as Pipecat async function schemas and handlers."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

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
                try:
                    result = await self._registry.call(
                        tool_name, dict(params.arguments)
                    )
                    await params.result_callback(
                        {"content": result.content, "is_error": result.is_error}
                    )
                except Exception:  # noqa: BLE001 - normalize tool boundary failures
                    await params.result_callback(
                        {
                            "content": [
                                {"type": "text", "text": "Tool execution failed."}
                            ],
                            "is_error": True,
                        }
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
