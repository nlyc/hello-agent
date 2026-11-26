"""A2ATool 测试：Agent as Tool、技能调用与 Trace。"""

from __future__ import annotations

import pytest

from agentflow.core.exceptions import ToolError
from agentflow.protocols.a2a import A2AServer, A2ATool
from agentflow.tools.registry import ToolRegistry
from agentflow.trace import Tracer


def make_server() -> A2AServer:
    server = A2AServer("规划助手", "生成并审查行程")

    @server.skill("plan_itinerary", "生成两日行程")
    def plan_itinerary(payload):
        city = payload.get("input", "北京")
        return f"{city}两日：D1 故宫（周二）→ 颐和园东宫门；D2 香山或室内备选。"

    @server.skill("review_plan", "审查行程")
    def review_plan(payload):
        plan = payload.get("input", "")
        return f"审查通过：避开周一闭馆。原计划：{plan}"

    return server


def test_a2a_tool_name_description_and_call():
    server = make_server()
    url = server.start()
    try:
        tool = A2ATool(url, tracer=None)
        assert tool.name.startswith("a2a_")
        assert "plan_itinerary" in tool.description

        result = tool.run({"skill": "plan_itinerary", "input": "北京"})
        assert "故宫" in result
        assert "北京两日" in result

        result = tool.run(
            {"skill": "plan_itinerary", "input": {"input": "上海"}}
        )
        assert "上海两日" in result
    finally:
        server.stop()


def test_a2a_tool_errors_are_tool_errors():
    server = make_server()
    url = server.start()
    try:
        tool = A2ATool(url)
        with pytest.raises(ToolError):
            tool.run({})
        with pytest.raises(ToolError):
            tool.run({"skill": "missing", "input": "北京"})
    finally:
        server.stop()


def test_a2a_tool_trace_and_registry():
    tracer = Tracer()
    events = []
    tracer.subscribe(events.append)
    server = make_server()
    url = server.start()
    try:
        tool = A2ATool(url, tracer=tracer, agent_name="coordinator")
        registry = ToolRegistry()
        registry.register_tool(tool)
        assert tool.name in registry
        assert "故宫" in registry.execute(
            tool.name, {"skill": "plan_itinerary", "input": "北京"}
        )
        calls = [event for event in events if event.type == "a2a.call"]
        assert calls
        assert calls[0].data["skill"] == "plan_itinerary"
        assert calls[0].data["agent"] == "规划助手"
        assert calls[0].agent_name == "coordinator"
        assert calls[0].data["duration_ms"] >= 0
    finally:
        server.stop()


def test_a2a_tool_review_chain():
    server = make_server()
    url = server.start()
    try:
        planner = A2ATool(url, name="planner")
        reviewer = A2ATool(url, name="reviewer")
        plan = planner.run({"skill": "plan_itinerary", "input": "北京"})
        review = reviewer.run({"skill": "review_plan", "input": plan})
        assert "审查通过" in review
        assert "故宫" in review
    finally:
        server.stop()
