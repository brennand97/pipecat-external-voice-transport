import asyncio

from voice_transport.tools.registry import ToolRegistry
from voice_transport.tools.session_control import (
    END_SESSION_TOOL,
    SessionControlToolProvider,
)


async def test_end_session_tool_signals_only_its_own_event() -> None:
    event = asyncio.Event()
    provider = SessionControlToolProvider(event)

    tools = await provider.list_tools()
    result = await provider.call_tool(END_SESSION_TOOL, {})

    assert [tool.name for tool in tools] == [END_SESSION_TOOL]
    assert not result.is_error
    assert event.is_set()


async def test_end_session_tool_bypasses_external_policy_filters() -> None:
    registry = ToolRegistry(
        (SessionControlToolProvider(asyncio.Event()),),
        allowed_patterns=(),
        requested_names=frozenset({"other__Tool"}),
        server_tool_names=frozenset({END_SESSION_TOOL}),
    )

    await registry.discover()

    assert registry.tool_names == (END_SESSION_TOOL,)
