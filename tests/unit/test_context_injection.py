from dataclasses import dataclass

from voice_transport.tools.base import ToolDefinition, ToolResult
from voice_transport.tools.registry import ToolRegistry


@dataclass
class Provider:
    arguments: dict | None = None

    async def list_tools(self):
        return [
            ToolDefinition(
                "voice_satellite__StartTimer",
                "Start a device timer.",
                {
                    "type": "object",
                    "properties": {
                        "seconds": {"type": "integer"},
                        "device_id": {"type": "string"},
                    },
                    "required": ["seconds", "device_id"],
                    "additionalProperties": False,
                },
            )
        ]

    async def call_tool(self, _name, arguments):
        self.arguments = arguments
        return ToolResult([])

    async def close(self):
        pass


async def test_context_argument_is_hidden_from_schema_and_injected_at_call() -> None:
    provider = Provider()
    registry = ToolRegistry(
        (provider,),
        context_injections={"voice_satellite__StartTimer": {"device_id": "device-1"}},
    )

    [tool] = await registry.discover()
    assert "device_id" not in tool.input_schema["properties"]
    assert "device_id" not in tool.input_schema["required"]

    result = await registry.call("voice_satellite__StartTimer", {"seconds": 30})
    assert not result.is_error
    assert provider.arguments == {"seconds": 30, "device_id": "device-1"}


async def test_start_timer_gets_a_server_default_name_when_omitted() -> None:
    class TimerProvider(Provider):
        async def list_tools(self):
            return [
                ToolDefinition(
                    "StartTimer",
                    "Start a device timer.",
                    {
                        "type": "object",
                        "properties": {
                            "device_id": {"type": "string"},
                            "minutes": {"type": "integer"},
                            "name": {"type": "string"},
                        },
                        "required": ["device_id"],
                    },
                )
            ]

    provider = TimerProvider()
    registry = ToolRegistry(
        (provider,), context_injections={"StartTimer": {"device_id": "device-1"}}
    )
    await registry.discover()

    result = await registry.call("StartTimer", {"minutes": 1})

    assert not result.is_error
    assert provider.arguments == {
        "minutes": 1,
        "device_id": "device-1",
        "name": "Timer",
    }


async def test_model_cannot_supply_hidden_context_argument() -> None:
    provider = Provider()
    registry = ToolRegistry(
        (provider,),
        context_injections={"voice_satellite__StartTimer": {"device_id": "device-1"}},
    )
    await registry.discover()

    result = await registry.call(
        "voice_satellite__StartTimer", {"seconds": 30, "device_id": "other"}
    )
    assert result.is_error
    assert provider.arguments is None
