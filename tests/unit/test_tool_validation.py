import json
from typing import Any

from voice_transport.session_audit import SessionAuditLog
from voice_transport.tools.base import ToolDefinition, ToolResult
from voice_transport.tools.registry import ToolRegistry


class RecordingProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def list_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                "set_light",
                "Set a light",
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            )
        ]

    async def call_tool(self, _name: str, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append(arguments)
        return ToolResult([{"type": "text", "text": "ok"}])

    async def close(self) -> None:
        return None


async def test_registry_rejects_model_arguments_that_violate_tool_schema() -> None:
    provider = RecordingProvider()
    registry = ToolRegistry((provider,))

    result = await registry.call("set_light", {"brightness": "bright"})

    assert result.is_error is True
    assert result.content == [
        {"type": "text", "text": "Invalid arguments: missing required property 'name'."}
    ]
    assert provider.calls == []


async def test_registry_rejects_unknown_and_out_of_range_model_arguments() -> None:
    provider = RecordingProvider()
    registry = ToolRegistry((provider,))

    unknown = await registry.call("set_light", {"name": "Kitchen", "colour": "red"})
    out_of_range = await registry.call(
        "set_light", {"name": "Kitchen", "brightness": 101}
    )

    assert unknown.is_error is True
    assert "unknown property 'colour'" in unknown.content[0]["text"]
    assert out_of_range.is_error is True
    assert "must be <= 100" in out_of_range.content[0]["text"]
    assert provider.calls == []


async def test_debug_audit_records_tool_arguments_and_results(tmp_path) -> None:
    provider = RecordingProvider()
    registry = ToolRegistry(
        (provider,),
        audit=SessionAuditLog(tmp_path, mode="debug_content", retention_days=7),
        session_id="session-1",
    )

    await registry.call("set_light", {"name": "Kitchen", "brightness": 50})

    events = [
        json.loads(line)
        for line in next(tmp_path.glob("sessions-*.jsonl")).read_text().splitlines()
    ]
    completed = next(
        event for event in events if event["event"] == "tool.call_finished"
    )
    assert completed["arguments"] == {"brightness": 50, "name": "Kitchen"}
    assert completed["result"] == [{"text": "ok", "type": "text"}]


async def test_registry_forwards_valid_model_arguments() -> None:
    provider = RecordingProvider()
    registry = ToolRegistry((provider,))

    result = await registry.call("set_light", {"name": "Kitchen", "brightness": 50})

    assert result.is_error is False
    assert provider.calls == [{"name": "Kitchen", "brightness": 50}]
