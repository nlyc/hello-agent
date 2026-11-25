"""Agent × ContextBuilder 集成测试：注入、去重、失败降级与三范式。"""

from __future__ import annotations

import pytest

from agentflow.agents import PlanAndSolveAgent, ReActAgent, ReflectionAgent
from agentflow.context import (
    ContextBuilder,
    ContextConfig,
    ContextPacket,
    StaticContextSource,
)
from agentflow.memory.context import MemoryContext
from agentflow.trace import Tracer
from tests.fakes import ScriptedLLM

MEMORY_TEXT = "【记忆上下文】\n- [长期记忆] 旧记忆"
CONTEXT_ITEM = "用户偏好自然风光"


class FakeMemory:
    """记录记忆构建与写回的测试替身。"""

    def __init__(self) -> None:
        self.build_calls: list[str] = []
        self.turns: list[tuple[str, str]] = []

    def build_context(self, query: str = "") -> MemoryContext:
        self.build_calls.append(query)
        return MemoryContext(text=MEMORY_TEXT, items=[{"content": "旧记忆"}])

    def record_turn(self, user_input: str, assistant_output: str) -> str:
        self.turns.append((user_input, assistant_output))
        return "memory-id"


def make_builder(item: str = CONTEXT_ITEM, *, role: str = "你是旅行助手") -> ContextBuilder:
    source = StaticContextSource(
        [
            ContextPacket(
                content=item,
                relevance_score=0.9,
                source="memory",
                section="context",
                trusted=True,
            )
        ]
    )
    return ContextBuilder(
        [source],
        config=ContextConfig(max_tokens=1000),
        role_prompt=role,
        output_rules="给出可执行行程",
    )


def finish_llm() -> ScriptedLLM:
    return ScriptedLLM(["Thought: 信息足够\nAction: Finish[北京山水路线]"])


def test_no_builder_keeps_legacy_memory_path():
    memory = FakeMemory()
    llm = finish_llm()
    agent = ReActAgent(name="react", llm=llm, memory=memory)

    assert agent.run("北京怎么玩") == "北京山水路线"
    messages = llm.calls[0]["messages"]
    assert MEMORY_TEXT in messages[0]["content"]
    assert messages[1]["content"].startswith("相关记忆：")
    assert memory.build_calls == ["北京怎么玩"]
    assert memory.turns == [("北京怎么玩", "北京山水路线")]


def test_builder_injects_prompt_text_and_emits_trace():
    tracer = Tracer()
    events = []
    tracer.subscribe(events.append)
    llm = finish_llm()
    agent = ReActAgent(
        name="react",
        llm=llm,
        context_builder=make_builder(),
        tracer=tracer,
    )

    assert agent.run("北京怎么玩") == "北京山水路线"

    messages = llm.calls[0]["messages"]
    system = messages[0]["content"]
    assert "# Role & Policies" in system
    assert "你是旅行助手" in system
    assert CONTEXT_ITEM in system
    assert "# Task" not in system
    assert "北京怎么玩" not in system
    assert messages[-1] == {"role": "user", "content": "北京怎么玩"}
    assert len(messages) == 2  # 不再额外插入「相关记忆」消息

    context_events = [event for event in events if event.type == "context.build"]
    assert context_events
    data = context_events[0].data
    assert data["provider"] == "ContextBuilder"
    assert data["tokens"] > 0
    assert "role" in data["sections"]
    assert data["packets_selected"] >= 2


def test_builder_does_not_duplicate_items_message():
    llm = finish_llm()
    agent = ReActAgent(name="react", llm=llm, context_builder=make_builder())
    agent.run("北京怎么玩")
    contents = [message["content"] for message in llm.calls[0]["messages"]]
    assert not any("相关记忆" in content for content in contents)
    assert sum(content.count(CONTEXT_ITEM) for content in contents) == 1


def test_builder_failure_degrades_to_empty_context():
    class BrokenBuilder:
        def build(self, task, *, history=None, **kwargs):
            raise RuntimeError("builder down")

    tracer = Tracer()
    events = []
    tracer.subscribe(events.append)
    llm = finish_llm()
    agent = ReActAgent(
        name="react",
        llm=llm,
        context_builder=BrokenBuilder(),
        tracer=tracer,
    )

    assert agent.run("北京怎么玩") == "北京山水路线"
    messages = llm.calls[0]["messages"]
    assert "# Role & Policies" not in messages[0]["content"]
    errors = [event for event in events if event.type == "context.error"]
    assert errors and "builder down" in errors[0].data["error"]


def test_builder_does_not_call_legacy_memory_build_but_records_turn():
    memory = FakeMemory()
    llm = finish_llm()
    agent = ReActAgent(
        name="react",
        llm=llm,
        memory=memory,
        context_builder=make_builder(),
    )
    assert agent.run("北京怎么玩") == "北京山水路线"
    assert memory.build_calls == []
    assert memory.turns == [("北京怎么玩", "北京山水路线")]


def test_reflection_uses_prompt_text_without_task_duplication():
    llm = ScriptedLLM(["初稿", "无需改进"])
    agent = ReflectionAgent(
        name="reflection",
        llm=llm,
        max_iterations=1,
        context_builder=make_builder(),
    )
    assert agent.run("设计北京三日游") == "初稿"

    initial_prompt = llm.calls[0]["messages"][0]["content"]
    reflect_prompt = llm.calls[1]["messages"][0]["content"]
    assert "你是旅行助手" in initial_prompt
    assert CONTEXT_ITEM in initial_prompt
    assert initial_prompt.count("设计北京三日游") == 1
    assert reflect_prompt.count("设计北京三日游") == 1


def test_plan_solve_uses_prompt_text_in_planner_and_executor():
    llm = ScriptedLLM(
        [
            '```python\n["查询北京自然风光景点"]\n```',
            "推荐颐和园与香山",
        ]
    )
    agent = PlanAndSolveAgent(
        name="plan", llm=llm, context_builder=make_builder()
    )
    assert agent.run("规划北京自然风光路线") == "推荐颐和园与香山"

    planner_prompt = llm.calls[0]["messages"][0]["content"]
    executor_prompt = llm.calls[1]["messages"][0]["content"]
    assert "你是旅行助手" in planner_prompt
    assert CONTEXT_ITEM in planner_prompt
    assert planner_prompt.count("规划北京自然风光路线") == 1
    assert executor_prompt.count("规划北京自然风光路线") == 1


def test_context_config_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CONTEXT_MAX_TOKENS", "123")
    monkeypatch.setenv("CONTEXT_RELEVANCE_WEIGHT", "0.5")
    monkeypatch.setenv("CONTEXT_RECENCY_WEIGHT", "0.5")
    monkeypatch.setenv("CONTEXT_RECENT_HISTORY", "2")
    config = ContextConfig.from_env()
    assert config.max_tokens == 123
    assert config.relevance_weight == 0.5
    assert config.recency_weight == 0.5
    assert config.recent_history == 2


def test_failed_run_does_not_keep_user_message():
    from agentflow.core.agent import Agent

    class BoomAgent(Agent):
        def _execute(self, input_text, **kwargs):
            raise RuntimeError("boom")

    agent = BoomAgent(name="boom", llm=ScriptedLLM())
    with pytest.raises(RuntimeError):
        agent.run("问题")
    assert agent.get_history() == []
