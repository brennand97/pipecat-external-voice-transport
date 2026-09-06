import sys

from voice_transport.tools.script import ScriptToolConfig, ScriptToolProvider

_SCRIPT = """
import json
import sys
request = json.load(sys.stdin)
if request['method'] == 'tools.list':
    response = {
        'tools': [
            {
                'name': 'echo',
                'description': 'Echo text',
                'input_schema': {'type': 'object'},
            }
        ]
    }
else:
    response = {'content': [{'type': 'text', 'text': request['arguments']['text']}]}
print(json.dumps(response))
"""


def provider() -> ScriptToolProvider:
    return ScriptToolProvider(ScriptToolConfig("test", (sys.executable, "-c", _SCRIPT)))


async def test_script_tools_discover_and_invoke_asynchronously() -> None:
    tool_provider = provider()
    assert (await tool_provider.list_tools())[0].name == "echo"
    result = await tool_provider.call_tool("echo", {"text": "hello"})
    assert result.content == [{"type": "text", "text": "hello"}]


async def test_script_tool_discovery_is_cached() -> None:
    tool_provider = provider()
    first = await tool_provider.list_tools()
    second = await tool_provider.list_tools()
    assert first == second
