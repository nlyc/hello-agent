"""MCPTool 测试：自动展开、名称前缀、类型转换、直接调用与 Trace。"""

from __future__ import annotations

import pytest

from agentflow.core.exceptions import ToolError
from agentflow.protocols.mcp import (
    InMemoryTransport,
    MCPClient,
    MCPRemoteTool,
    MCPServer,
    MCPTool,
)
from agentflow.tools.registry import ToolRegistry
from agentflow.trace import Tracer


def make_server() -> MCPServer:
    server = MCPServer("math-server")

    def add(a: int, b: int) -> int:
        """两数相加"""
        return a + b

    def concat(items: list, enabled: bool = False) -> str:
        """拼接列表"""
        return ("Y" if enabled else "N") + ",".join(str(item) for item in items)

    server.add_tool(add)
    server.add_tool(concat)
    return server


def make_mcp_tool(tracer=None) -> MCPTool:
    server = make_server()
    return MCPTool(
        "math server",
        transport=InMemoryTransport(server.handle),
        tracer=tracer,
        agent_name="math-agent",
    )


def test_expand_prefixes_and_parameters():
    tool = make_mcp_tool()
    wrappers = tool.expand()
    by_name = {wrapper.name: wrapper for wrapper in wrappers}
    assert set(by_name) == {"math_server_add", "math_server_concat"}

    add = by_name["math_server_add"]
    assert [parameter.name for parameter in add.get_parameters()] == ["a", "b"]
    assert add.get_parameters()[0].type == "integer"
    assert add.get_parameters()[0].required is True
    assert by_name["math_server_concat"].get_parameters()[1].required is False


def test_remote_tool_coerces_types():
    wrappers = {wrapper.name: wrapper for wrapper in make_mcp_tool().expand()}
    assert wrappers["math_server_add"].run({"a": "2", "b": "3"}) == "5"
    assert wrappers["math_server_concat"].run(
        {"items": '["a", "b"]', "enabled": "true"}
    ) == "Ya,b"


def test_remote_tool_invalid_type_raises_tool_error():
    wrappers = {wrapper.name: wrapper for wrapper in make_mcp_tool().expand()}
    with pytest.raises(ToolError):
        wrappers["math_server_add"].run({"a": "不是数字", "b": "1"})
    with pytest.raises(ToolError):
        wrappers["math_server_concat"].run({"items": "不是数组"})


def test_direct_actions_and_register():
    tool = make_mcp_tool()
    listed = tool.run({"action": "list"})
    assert "- add:" in listed
    assert "- concat:" in listed
    assert tool.run(
        {"action": "call", "tool": "add", "arguments": {"a": 4, "b": 5}}
    ) == "9"
    assert tool.run(
        {"action": "call", "tool": "add", "arguments": '{"a": 1, "b": 2}'}
    ) == "3"
    assert tool.run({"action": "call"}).startswith("错误:")
    assert tool.run({"action": "unknown"}).startswith("错误:")

    registry = ToolRegistry()
    expanded = tool.expand_and_register(registry)
    assert len(expanded) == 2
    assert "math_server_add" in registry


def test_mcp_call_trace():
    tracer = Tracer()
    events = []
    tracer.subscribe(events.append)
    wrappers = {wrapper.name: wrapper for wrapper in make_mcp_tool(tracer).expand()}
    assert wrappers["math_server_add"].run({"a": 1, "b": 2}) == "3"
    calls = [event for event in events if event.type == "mcp.call"]
    assert calls
    assert calls[0].data["server_tool"] == "add"
    assert calls[0].data["output"] == "3"
    assert calls[0].agent_name == "math-agent"


def test_close_marks_transport_closed():
    tool = make_mcp_tool()
    tool.connect()
    tool.close()
    assert tool.client._transport.closed is True
    with pytest.raises(Exception):
        MCPClient(transport=tool.client._transport).list_tools()


def test_remote_tool_uses_configured_call_timeout():
    class RecordingClient:
        def __init__(self) -> None:
            self.timeout = None

        def call_tool(self, name, arguments=None, *, timeout=None):
            self.timeout = timeout
            return "ok"

    client = RecordingClient()
    wrapper = MCPRemoteTool(
        client=client,
        remote_name="slow",
        public_name="slow",
        description="slow tool",
        input_schema={},
        timeout=42.0,
    )
    assert wrapper.run({}) == "ok"
    assert client.timeout == 42.0
