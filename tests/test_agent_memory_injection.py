"""三范式自动注入记忆测试：构建、注入、写回、开关与降级。"""

from __future__ import annotations

from typing import Any

import pytest

from agentflow.agents import PlanAndSolveAgent, ReActAgent, ReflectionAgent
from agentflow.core.agent import Agent
from agentflow.core.config import Config, MemoryConfig
from agentflow.memory import BaseMemory, MemoryContext
from agentflow.trace import Tracer
from tests.fakes import ScriptedLLM

MEMORY_TEXT = "【记忆上下文】\n- [长期记忆] 用户偏好自然风光"
MEMORY_ITEM = "用户偏好自然风光"


class FakeMemoryManager:
    """返回固定记忆文本并记录写回调用的测试替身。"""

    def __init__(self, text: str = MEMORY_TEXT, items: list[dict] | None = None):
        self.text = text
        self.items = items if items is not None else [{"content": MEMORY_ITEM}]
        self.turns: list[tuple[str, str]] = []
        self.build_calls: list[str] = []

    def build_context(self, query: str = "") -> MemoryContext:
        self.build_calls.append(query)
        return MemoryContext(
            text=self.text,
            items=list(self.items),
            working_items=[],
            long_term_hits=[],
        )

    def record_turn(self, user_input: str, assistant_output: str) -> str:
        self.turns.append((user_input, assistant_output))
        return "memory-id"


class BrokenBuildMemory(FakeMemoryManager):
    """构建上下文必定失败的测试替身。"""

    def build_context(self, query: str = "") -> MemoryContext:
        raise RuntimeError("build down")


class BrokenWriteMemory(FakeMemoryManager):
    """写回必定失败的测试替身。"""

    def record_turn(self, user_input: str, assistant_output: str) -> str:
        raise RuntimeError("write down")


class RecordingMemory(BaseMemory):
    """旧形态 BaseMemory：只实现 remember / recall / clear。"""

    def __init__(self, recalled: str = "旧记忆：用户偏好自然风光"):
        self.recalled = recalled
        self.queries: list[str] = []
        self.remembered: list[tuple[str, str]] = []

    def remember(self, key: str, value: str) -> None:
        self.remembered.append((key, value))

    def recall(self, query: str) -> str:
        self.queries.append(query)
        return self.recalled

    def clear(self) -> None:
        self.remembered.clear()


class RaisingAgent(Agent):
    """_execute 必定失败的 Agent，用于验证失败不写记忆。"""

    def _execute(self, input_text: str, **kwargs) -> str:
        raise RuntimeError("execute failed")


def _react(llm, memory=None, **kwargs) -> ReActAgent:
    """构建返回 Finish 的 ReAct Agent。"""
    return ReActAgent(name="react", llm=llm, memory=memory, **kwargs)


def _finish_llm() -> ScriptedLLM:
    """返回一次 Finish 的脚本化 LLM。"""
    return ScriptedLLM(["Thought: 已知信息足够\nAction: Finish[北京推荐山水路线]"])


def test_react_injects_memory_into_system_prompt_and_message():
    memory = FakeMemoryManager()
    llm = _finish_llm()
    agent = _react(llm, memory=memory)

    assert agent.run("北京怎么玩") == "北京推荐山水路线"

    messages = llm.calls[0]["messages"]
    assert MEMORY_TEXT in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert MEMORY_ITEM in messages[1]["content"]
    assert messages[2] == {"role": "user", "content": "北京怎么玩"}
    assert memory.build_calls == ["北京怎么玩"]
    assert memory.turns == [("北京怎么玩", "北京推荐山水路线")]


def test_reflection_injects_memory_into_all_three_stages():
    memory = FakeMemoryManager()
    llm = ScriptedLLM(["初稿", "无需改进"])
    agent = ReflectionAgent(name="reflection", llm=llm, memory=memory, max_iterations=1)

    assert agent.run("设计北京三日游") == "初稿"

    initial_prompt = llm.calls[0]["messages"][0]["content"]
    reflect_prompt = llm.calls[1]["messages"][0]["content"]
    assert MEMORY_TEXT in initial_prompt
    assert MEMORY_TEXT in reflect_prompt
    assert "设计北京三日游" in initial_prompt
    assert memory.turns == [("设计北京三日游", "初稿")]


def test_reflection_refine_stage_also_sees_memory():
    memory = FakeMemoryManager()
    llm = ScriptedLLM(["初稿", "建议补充交通信息", "终稿"])
    agent = ReflectionAgent(name="reflection", llm=llm, memory=memory, max_iterations=1)

    assert agent.run("设计北京三日游") == "终稿"
    assert len(llm.calls) == 3
    refine_prompt = llm.calls[2]["messages"][0]["content"]
    assert MEMORY_TEXT in refine_prompt
    assert "设计北京三日游" in refine_prompt


def test_plan_solve_injects_memory_into_planner_and_executor():
    memory = FakeMemoryManager()
    llm = ScriptedLLM(
        [
            '```python\n["查询北京自然风光景点"]\n```',
            "推荐颐和园与香山",
        ]
    )
    agent = PlanAndSolveAgent(name="plan", llm=llm, memory=memory)

    assert agent.run("规划北京自然风光路线") == "推荐颐和园与香山"

    planner_prompt = llm.calls[0]["messages"][0]["content"]
    executor_prompt = llm.calls[1]["messages"][0]["content"]
    assert MEMORY_TEXT in planner_prompt
    assert MEMORY_TEXT in executor_prompt
    assert memory.turns == [("规划北京自然风光路线", "推荐颐和园与香山")]


def test_record_turn_is_called_once_on_success():
    memory = FakeMemoryManager()
    agent = _react(_finish_llm(), memory=memory)
    agent.run("问题")
    assert len(memory.turns) == 1


def test_record_turn_is_skipped_when_execute_raises():
    memory = FakeMemoryManager()
    agent = RaisingAgent(name="boom", llm=ScriptedLLM(), memory=memory)
    with pytest.raises(RuntimeError):
        agent.run("问题")
    assert memory.turns == []
    assert memory.build_calls == ["问题"]


def test_legacy_base_memory_recall_path():
    memory = RecordingMemory()
    llm = _finish_llm()
    agent = _react(llm, memory=memory)

    assert agent.run("北京怎么玩") == "北京推荐山水路线"
    messages = llm.calls[0]["messages"]
    assert "旧记忆：用户偏好自然风光" in messages[0]["content"]
    assert len(messages) == 2  # 旧形态没有结构化条目，不额外插消息
    assert memory.queries == ["北京怎么玩"]
    assert memory.remembered == []  # 旧形态不自动写回，保持 v1.0 行为


def test_memory_enabled_false_matches_v1_baseline():
    baseline_llm = _finish_llm()
    _react(baseline_llm, memory=None).run("北京怎么玩")

    memory = FakeMemoryManager()
    disabled_llm = _finish_llm()
    agent = _react(disabled_llm, memory=memory, memory_enabled=False)
    agent.run("北京怎么玩")

    assert disabled_llm.calls[0]["messages"] == baseline_llm.calls[0]["messages"]
    assert memory.build_calls == []
    assert memory.turns == []


def test_config_memory_disabled_matches_v1_baseline():
    config = Config(memory=MemoryConfig(enabled=False))
    baseline_llm = _finish_llm()
    _react(baseline_llm, memory=None, config=Config()).run("北京怎么玩")

    memory = FakeMemoryManager()
    llm = _finish_llm()
    agent = _react(llm, memory=memory, config=config)
    agent.run("北京怎么玩")

    assert agent.memory_enabled is False
    assert llm.calls[0]["messages"] == baseline_llm.calls[0]["messages"]
    assert memory.build_calls == []
    assert memory.turns == []


def test_build_failure_degrades_to_empty_context():
    tracer = Tracer()
    events = []
    tracer.subscribe(events.append)
    memory = BrokenBuildMemory()
    llm = _finish_llm()
    agent = _react(llm, memory=memory, tracer=tracer)

    assert agent.run("北京怎么玩") == "北京推荐山水路线"

    messages = llm.calls[0]["messages"]
    assert len(messages) == 2  # 无记忆注入
    assert "【记忆上下文】" not in messages[0]["content"]
    error_events = [event for event in events if event.type == "memory.error"]
    assert error_events and error_events[0].data["phase"] == "build"


def test_write_failure_does_not_break_run():
    tracer = Tracer()
    events = []
    tracer.subscribe(events.append)
    memory = BrokenWriteMemory()
    agent = _react(_finish_llm(), memory=memory, tracer=tracer)

    assert agent.run("北京怎么玩") == "北京推荐山水路线"

    assert any(event.type == "memory.build" for event in events)
    error_events = [event for event in events if event.type == "memory.error"]
    assert error_events and error_events[0].data["phase"] == "write"
    assert not any(event.type == "memory.write" for event in events)


def test_config_from_env_memory_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEMORY_ENABLED", "false")
    assert Config.from_env().memory.enabled is False
    monkeypatch.setenv("MEMORY_ENABLED", "true")
    assert Config.from_env().memory.enabled is True


class _KwargsEchoAgent(Agent):
    """记录收到的 memory_context，验证基类传参。"""

    def _execute(self, input_text: str, memory_context: Any = None, **kwargs) -> str:
        self.seen_context = memory_context
        return "ok"


def test_memory_context_is_passed_to_execute():
    memory = FakeMemoryManager()
    agent = _KwargsEchoAgent(name="echo", llm=ScriptedLLM(), memory=memory)
    assert agent.run("问题") == "ok"
    assert agent.seen_context is not None
    assert agent.seen_context.text == MEMORY_TEXT
