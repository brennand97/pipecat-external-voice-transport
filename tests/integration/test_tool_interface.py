"""Local, network-free integration coverage for the model-to-tool boundary."""

from dataclasses import dataclass, field
from typing import Any

from voice_transport.tools.base import ToolDefinition, ToolResult
from voice_transport.tools.pipecat_bridge import PipecatToolBridge
from voice_transport.tools.registry import ToolRegistry


class LocalHomeAssistantTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def list_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                "light__HassLightSet",
                "Set a light brightness",
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
                    },
                    "required": ["name", "brightness"],
                    "additionalProperties": False,
                },
            )
        ]

    async def call_tool(self, _name: str, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append(arguments)
        return ToolResult([{"type": "text", "text": "Light updated."}])

    async def close(self) -> None:
        return None


@dataclass
class ModelCall:
    arguments: dict[str, Any]
    results: list[dict[str, Any]] = field(default_factory=list)

    async def result_callback(self, result: dict[str, Any]) -> None:
        self.results.append(result)


async def test_local_tool_rejects_invalid_schema_input_without_call() -> None:
    provider = LocalHomeAssistantTool()
    bridge = PipecatToolBridge(ToolRegistry((provider,)))
    schema = (await bridge.function_schemas())[0]

    invalid_call = ModelCall({"name": "Kitchen", "brightness": 101})
    await schema.handler(invalid_call)

    assert invalid_call.results == [
        {
            "content": [
                {
                    "type": "text",
                    "text": "Invalid arguments: property 'brightness' must be <= 100.",
                }
            ],
            "is_error": True,
        }
    ]
    assert provider.calls == []


async def test_local_model_tool_interface_forwards_valid_schema_input() -> None:
    provider = LocalHomeAssistantTool()
    bridge = PipecatToolBridge(ToolRegistry((provider,)))
    schema = (await bridge.function_schemas())[0]

    valid_call = ModelCall({"name": "Kitchen", "brightness": 50})
    await schema.handler(valid_call)

    assert valid_call.results == [
        {"content": [{"type": "text", "text": "Light updated."}], "is_error": False}
    ]
    assert provider.calls == [{"name": "Kitchen", "brightness": 50}]
