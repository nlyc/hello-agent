"""Agent 基类单元测试。"""

import pytest

from agentflow.core import Agent
from agentflow.llm.base import BaseLLM


class DummyLLM(BaseLLM):
    def chat(self, messages, *, temperature=None, json_mode=False, tools=None):
        return "回答"


class EchoAgent(Agent):
    def _execute(self, input_text, **kwargs):
        return f"收到: {input_text}"


def test_agent_is_abstract():
    with pytest.raises(TypeError):
        Agent(name="x", llm=DummyLLM())


def test_run_records_user_and_assistant():
    a = EchoAgent(name="echo", llm=DummyLLM())
    assert a.run("hello") == "收到: hello"
    assert [m.role for m in a.get_history()] == ["user", "assistant"]


def test_get_history_returns_copy():
    a = EchoAgent(name="echo", llm=DummyLLM())
    a.run("x")
    hist = a.get_history()
    hist.clear()
    assert len(a.get_history()) == 2  # 内部未被外部清空


def test_clear_history():
    a = EchoAgent(name="echo", llm=DummyLLM())
    a.run("x")
    a.clear_history()
    assert a.get_history() == []
