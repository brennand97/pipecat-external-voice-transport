from types import SimpleNamespace

import pytest

from voice_transport.tools.base import ToolDefinition
from voice_transport.tools.mcp import MCPServerConfig, MCPToolProvider


def test_mcp_provider_is_lazy_until_a_tool_operation() -> None:
    provider = MCPToolProvider(
        MCPServerConfig(
            name="home-assistant",
            transport="streamable_http",
            url="https://ha.example/mcp",
        )
    )
    assert provider._session is None


async def test_mcp_provider_reads_current_sdk_snake_case_input_schema() -> None:
    provider = MCPToolProvider(
        MCPServerConfig(
            name="home-assistant",
            transport="streamable_http",
            url="https://ha.example/api/mcp",
            allowed_tools=frozenset({"get_state"}),
        )
    )

    class Session:
        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="get_state",
                        description="Read state",
                        input_schema={"type": "object", "properties": {}},
                    )
                ]
            )

    provider._session = Session()

    assert await provider.list_tools() == [
        ToolDefinition("get_state", "Read state", {"type": "object", "properties": {}})
    ]


async def test_mcp_provider_fails_closed_for_an_invalid_remote_tool_schema() -> None:
    provider = MCPToolProvider(
        MCPServerConfig(
            name="home-assistant",
            transport="streamable_http",
            url="https://ha.example/api/mcp",
            allowed_tools=frozenset({"broken"}),
        )
    )

    class Session:
        async def list_tools(self):
            return SimpleNamespace(
                tools=[SimpleNamespace(name="broken", description="Broken")]
            )

    provider._session = Session()

    with pytest.raises(ValueError, match="invalid input schema"):
        await provider.list_tools()
