from types import SimpleNamespace

import pytest

from voice_transport.tools.base import ToolDefinition, ToolResult
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


async def test_mcp_tool_call_stays_in_the_calling_task() -> None:
    provider = MCPToolProvider(
        MCPServerConfig(
            name="home-assistant",
            transport="streamable_http",
            url="https://ha.example/api/mcp",
            allowed_tools=frozenset({"get_state"}),
        )
    )

    class Session:
        task = None

        async def call_tool(self, _name, _arguments):
            import asyncio

            self.task = asyncio.current_task()
            return SimpleNamespace(content=[], isError=False)

    session = Session()
    provider._session = session
    import asyncio

    caller = asyncio.current_task()
    assert await provider.call_tool("get_state", {}) == ToolResult([], is_error=False)
    assert session.task is caller


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
