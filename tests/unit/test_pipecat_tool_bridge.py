from dataclasses import dataclass, field
from typing import Any

from voice_transport.tools.base import ToolDefinition, ToolResult
from voice_transport.tools.pipecat_bridge import PipecatToolBridge
from voice_transport.tools.registry import ToolRegistry


class EchoProvider:
    async def list_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                "echo",
                "Echo input",
                {"type": "object", "properties": {"text": {"type": "string"}}},
            )
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult([{"type": "text", "text": arguments["text"]}])

    async def close(self) -> None:
        return None


@dataclass
class Params:
    arguments: dict[str, Any]
    results: list[dict[str, Any]] = field(default_factory=list)

    async def result_callback(self, result: dict[str, Any]) -> None:
        self.results.append(result)


async def test_pipecat_bridge_emits_tool_lifecycle_with_arguments_and_result() -> None:
    events = []

    async def emit(event) -> None:
        events.append(event)

    bridge = PipecatToolBridge(ToolRegistry((EchoProvider(),)), emit)
    schema = (await bridge.function_schemas())[0]
    await schema.handler(Params({"text": "hello"}))

    assert [(event.type, event.tool_name) for event in events] == [
        ("assistant.tool_call_started", "echo"),
        ("assistant.tool_call_finished", "echo"),
    ]
    assert events[0].tool_arguments == {"text": "hello"}
    assert events[1].tool_result == [{"type": "text", "text": "hello"}]


async def test_pipecat_bridge_resolves_async_tool_result_callback() -> None:
    bridge = PipecatToolBridge(ToolRegistry((EchoProvider(),)))
    schema = (await bridge.function_schemas())[0]
    params = Params({"text": "hello"})
    await schema.handler(params)
    assert params.results == [
        {"content": [{"type": "text", "text": "hello"}], "is_error": False}
    ]
