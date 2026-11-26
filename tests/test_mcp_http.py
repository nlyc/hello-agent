"""MCP HTTP/SSE 传输测试：JSON-RPC 往返、鉴权与 SSE 握手。"""

from __future__ import annotations

from urllib.request import Request, urlopen

import pytest

from agentflow.core.exceptions import MCPError
from agentflow.protocols.mcp import (
    MCPClient,
    MCPHttpServer,
    MCPHttpTransport,
    MCPServer,
)


def make_http_server(*, token: str = "") -> MCPHttpServer:
    server = MCPServer("http-test")

    def add(a: int, b: int) -> int:
        """两数相加"""
        return a + b

    server.add_tool(add)
    server.add_resource("travel://guide", content="故宫周一闭馆。")
    server.add_prompt(
        "greet",
        arguments=[{"name": "name", "required": True}],
        template="你好 {name}",
    )
    return MCPHttpServer(server, token=token, port=0)


def test_http_roundtrip_tools_resources_prompts():
    http = make_http_server()
    url = http.start()
    try:
        client = MCPClient(transport=MCPHttpTransport(url, timeout=10))
        info = client.connect()
        assert info["serverInfo"]["name"] == "http-test"
        assert [tool["name"] for tool in client.list_tools()] == ["add"]
        assert client.call_tool("add", {"a": 2, "b": 3}) == "5"
        assert client.read_resource("travel://guide") == "故宫周一闭馆。"
        prompt = client.get_prompt("greet", {"name": "小明"})
        assert "小明" in prompt["messages"][0]["content"]["text"]
    finally:
        http.stop()


def test_http_token_auth_and_sse():
    http = make_http_server(token="secret")
    url = http.start()
    try:
        with pytest.raises(MCPError, match="401"):
            MCPClient(transport=MCPHttpTransport(url, timeout=10)).connect()

        authorized = MCPClient(
            transport=MCPHttpTransport(url, token="secret", timeout=10)
        )
        assert authorized.connect()["serverInfo"]["name"] == "http-test"

        sse_url = url.rsplit("/mcp", 1)[0] + "/mcp/sse"
        request = Request(
            sse_url, headers={"Authorization": "Bearer secret"}
        )
        with urlopen(request, timeout=5) as response:
            first = response.readline().decode("utf-8")
            second = response.readline().decode("utf-8")
        assert "event: ready" in first
        assert "data:" in second
    finally:
        http.stop()
