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


async def test_pipecat_bridge_resolves_async_tool_result_callback() -> None:
    bridge = PipecatToolBridge(ToolRegistry((EchoProvider(),)))
    schema = (await bridge.function_schemas())[0]
    params = Params({"text": "hello"})
    await schema.handler(params)
    assert params.results == [
        {"content": [{"type": "text", "text": "hello"}], "is_error": False}
    ]
