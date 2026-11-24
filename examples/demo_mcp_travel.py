"""MCP Demo：发现 → 展开 → 注册 → Agent 调用（离线，不联网）。

    py -3.14 -m examples.demo_mcp_travel

流程：
1. 以 stdio 启动 examples.mcp_server_travel；
2. initialize + tools/list；
3. MCPTool.expand() 生成 travel_* 包装工具并注册进 ToolRegistry；
4. ReActAgent 用脚本化 LLM 走一次「查天气 → 汇总」；
5. 打印 Trace，最后关闭子进程。
"""

from __future__ import annotations

import sys

from agentflow.agents import ReActAgent
from agentflow.llm.base import BaseLLM
from agentflow.protocols.mcp import MCPTool
from agentflow.tools import ToolRegistry
from agentflow.trace import Tracer
from agentflow.trace.event import TraceEvent


class ScriptedDemoLLM(BaseLLM):
    """按队列返回固定回复，用于离线演示 Agent 循环。"""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def chat(self, messages, *, temperature=None, json_mode=False, tools=None) -> str:
        self.calls += 1
        if not self._responses:
            return "Thought: 没有更多脚本\nAction: Finish[演示结束]"
        return self._responses.pop(0)


def _on_trace(event: TraceEvent) -> None:
    """只打印关键事件，避免刷屏。"""
    if event.type in ("tool", "mcp.call", "end"):
        data = dict(event.data or {})
        if event.type == "tool":
            data = {"tool": data.get("tool"), "params": data.get("params")}
        elif event.type == "mcp.call":
            data = {
                "server_tool": data.get("server_tool"),
                "arguments": data.get("arguments"),
                "duration_ms": data.get("duration_ms"),
            }
        print(f"[trace] {event.type}: {data}")


def main() -> int:
    tracer = Tracer()
    tracer.subscribe(_on_trace)

    mcp = MCPTool(
        "travel",
        command=[sys.executable, "-m", "examples.mcp_server_travel"],
        timeout=30.0,
        connect_timeout=10.0,
        tracer=tracer,
        agent_name="mcp-travel",
    )
    try:
        info = mcp.connect()
        server = info.get("serverInfo", {})
        print(f"[mcp] connected: {server.get('name')} v{server.get('version')}")

        tools = mcp.list_tools()
        print(f"[mcp] discovered {len(tools)} tools:")
        for item in tools:
            print(f"  - {item['name']}: {item.get('description', '')}")

        registry = ToolRegistry()
        wrappers = mcp.expand_and_register(registry)
        print(f"[mcp] registered {len(wrappers)} tools: {[w.name for w in wrappers]}")

        direct = registry.execute("travel_get_weather", {"city": "北京"})
        print(f"[direct] travel_get_weather(北京) -> {direct}")

        llm = ScriptedDemoLLM(
            [
                'Thought: 需要查询北京天气\nAction: travel_get_weather(city="北京")',
                "Thought: 已拿到天气\nAction: Finish[北京：晴，18℃，适合出行。]",
            ]
        )
        agent = ReActAgent(
            name="mcp-travel-agent",
            llm=llm,
            tools=registry,
            tracer=tracer,
        )
        answer = agent.run("北京天气怎么样？")
        print(f"[agent] {answer}")
        return 0
    finally:
        mcp.close()


if __name__ == "__main__":
    raise SystemExit(main())
