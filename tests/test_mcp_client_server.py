"""MCP Client / Server 测试：内存传输、stdio 子进程与错误映射。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agentflow.core.exceptions import MCPError
from agentflow.protocols.mcp import (
    InMemoryTransport,
    MCPClient,
    MCPServer,
    JsonRpcError,
)

ROOT = Path(__file__).resolve().parents[1]


def make_server() -> MCPServer:
    server = MCPServer("test-server")

    def add(a: int, b: int) -> int:
        """两数相加"""
        return a + b

    def fail() -> str:
        """总是失败"""
        raise RuntimeError("boom")

    server.add_tool(add)
    server.add_tool(fail)
    return server


def test_inmemory_handshake_list_and_call():
    server = make_server()
    client = MCPClient(transport=InMemoryTransport(server.handle))

    info = client.connect()
    assert info["serverInfo"]["name"] == "test-server"
    assert "tools" in info["capabilities"]

    tools = {item["name"]: item for item in client.list_tools()}
    assert set(tools) == {"add", "fail"}
    assert tools["add"]["description"] == "两数相加"
    assert tools["add"]["inputSchema"]["properties"]["a"]["type"] == "integer"
    assert tools["add"]["inputSchema"]["required"] == ["a", "b"]

    assert client.call_tool("add", {"a": 2, "b": 3}) == "5"
    with pytest.raises(MCPError, match="boom"):
        client.call_tool("fail")
    client.close()


def test_server_protocol_errors():
    server = make_server()
    with pytest.raises(JsonRpcError) as info:
        server.handle("tools/call", {"name": "missing"})
    assert info.value.code == -32602

    with pytest.raises(JsonRpcError) as info:
        server.handle("unknown/method", {})
    assert info.value.code == -32601

    with pytest.raises(JsonRpcError):
        server.handle("tools/call", {})


def test_duplicate_tool_rejected():
    server = MCPServer()

    def echo(text: str) -> str:
        return text

    server.add_tool(echo)
    with pytest.raises(ValueError):
        server.add_tool(echo)


def test_inmemory_resources_and_prompts():
    server = make_server()
    server.add_resource(
        "travel://guide/beijing",
        name="北京攻略",
        description="静态攻略",
        content="故宫周一闭馆。",
    )
    server.add_prompt(
        "plan_trip",
        description="生成规划提示词",
        arguments=[{"name": "city", "required": True}],
        template="请为{city}规划两日行程。",
    )
    client = MCPClient(transport=InMemoryTransport(server.handle))
    assert client.connect()["capabilities"]["resources"] == {}

    resources = client.list_resources()
    assert resources[0]["uri"] == "travel://guide/beijing"
    assert client.read_resource("travel://guide/beijing") == "故宫周一闭馆。"

    prompts = client.list_prompts()
    assert prompts[0]["name"] == "plan_trip"
    prompt = client.get_prompt("plan_trip", {"city": "北京"})
    assert "北京" in prompt["messages"][0]["content"]["text"]

    with pytest.raises(Exception):
        client.get_prompt("plan_trip", {})


def test_stdio_roundtrip(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(
        "from agentflow.protocols.mcp import MCPServer\n"
        "server = MCPServer('stdio-test')\n"
        "def echo(text: str) -> str:\n"
        "    '''回显文本'''\n"
        "    return text\n"
        "server.add_tool(echo)\n"
        "server.run_stdio()\n",
        encoding="utf-8",
    )
    client = MCPClient(
        [sys.executable, str(script)],
        timeout=10,
        env={"PYTHONPATH": str(ROOT)},
    )
    try:
        info = client.connect()
        assert info["serverInfo"]["name"] == "stdio-test"
        assert [item["name"] for item in client.list_tools()] == ["echo"]
        assert client.call_tool("echo", {"text": "你好 MCP"}) == "你好 MCP"
    finally:
        client.close()


def test_stdio_missing_command():
    with pytest.raises(MCPError):
        MCPClient([])
