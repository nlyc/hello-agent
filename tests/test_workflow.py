"""StateGraph 引擎（自研状态图）测试：线性图、条件边、流式、Mermaid、安全阀。"""

import pytest

from agentflow.core.exceptions import WorkflowError
from agentflow.workflow import END, START, StateGraph


def test_linear_invoke_accumulates_state():
    g = StateGraph()
    g.add_node("a", lambda s: {"x": s.get("x", 0) + 1})
    g.add_node("b", lambda s: {"y": 2})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)

    result = g.compile().invoke({"x": 0})
    assert result == {"x": 1, "y": 2}


def test_stream_yields_nodes_in_order():
    g = StateGraph()
    seen = []
    g.add_node("a", lambda s: seen.append("a") or {"v": 1})
    g.add_node("b", lambda s: seen.append("b") or {"v": 2})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)

    nodes = [n for n, _ in g.compile().stream({"v": 0})]
    assert nodes == ["a", "b"]


def test_conditional_edges_support_loop():
    g = StateGraph()

    def bump(s):
        return {"n": s.get("n", 0) + 1}

    def route(s):
        return "again" if s.get("n", 0) < 3 else "done"

    g.add_node("a", bump)
    g.add_node("b", bump)
    g.add_edge(START, "a")
    g.add_conditional_edges("a", route, {"again": "a", "done": "b"})
    g.add_edge("b", END)

    assert g.compile().invoke({"n": 0})["n"] == 4


def test_max_steps_safety_valve():
    g = StateGraph()

    def loop(s):
        return {"n": s.get("n", 0) + 1}

    g.add_node("loop", loop)
    g.add_edge(START, "loop")
    g.add_conditional_edges("loop", lambda s: "loop", {"loop": "loop"})

    with pytest.raises(WorkflowError):
        g.compile(max_steps=5).invoke({"n": 0})


def test_unknown_route_key_raises():
    g = StateGraph()
    g.add_node("a", lambda s: {})
    g.add_edge(START, "a")
    g.add_conditional_edges("a", lambda s: "missing", {"present": "a"})

    with pytest.raises(WorkflowError):
        g.compile().invoke({})


def test_unknown_node_raises():
    g = StateGraph()
    g.add_edge(START, "nope")

    with pytest.raises(WorkflowError):
        g.compile().invoke({})


def test_to_mermaid():
    g = StateGraph()
    g.add_node("a", lambda s: {})
    g.add_node("b", lambda s: {})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)

    mermaid = g.compile().to_mermaid()
    assert mermaid.startswith("flowchart TD")
    for token in ("START", "END", "a", "b"):
        assert token in mermaid
