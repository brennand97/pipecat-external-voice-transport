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
