"""SearchTool（多源降级链）测试。"""

import pytest

from agentflow.core.exceptions import ToolError
from agentflow.tools.builtin import SearchTool


def test_uses_first_backend():
    backends = [("a", lambda q: f"A:{q}"), ("b", lambda q: f"B:{q}")]
    tool = SearchTool(backends=backends)
    assert tool.run({"input": "天气"}) == "A:天气"


def test_degrades_on_failure():
    def boom(q):
        raise RuntimeError("挂了")

    backends = [("a", boom), ("b", lambda q: f"B:{q}")]
    tool = SearchTool(backends=backends)
    assert tool.run({"input": "x"}) == "B:x"


def test_all_fail_raises():
    def boom(q):
        raise RuntimeError("挂了")

    tool = SearchTool(backends=[("a", boom)])
    with pytest.raises(ToolError):
        tool.run({"input": "x"})


def test_empty_query_raises():
    tool = SearchTool(backends=[("a", lambda q: "ok")])
    with pytest.raises(ToolError):
        tool.run({"input": "  "})


def test_no_backends_raises(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    tool = SearchTool()
    with pytest.raises(ToolError):
        tool.run({"input": "x"})
