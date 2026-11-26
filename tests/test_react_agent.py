"""ReActAgent（文本协议循环）测试。"""

import pytest

from agentflow.agents import ReActAgent
from agentflow.core.exceptions import AgentError
from agentflow.tools import ToolParameter, ToolRegistry
from tests.fakes import ScriptedLLM


def _weather_registry():
    calls = []
    registry = ToolRegistry()

    def get_weather(city):
        calls.append(("get_weather", city))
        return f"{city}今天晴"

    registry.register_function(
        "get_weather",
        "查询城市天气",
        get_weather,
        [ToolParameter("city", "string", "城市名", True)],
    )
    return registry, calls


def test_react_tool_then_finish():
    registry, calls = _weather_registry()
    llm = ScriptedLLM(
        [
            'Thought: 先查天气\nAction: get_weather(city="北京")',
            "Thought: 可以回答了\nAction: Finish[北京今天晴，适合出行。]",
        ]
    )
    agent = ReActAgent("travel", llm, tools=registry)
    result = agent.run("北京天气怎么样？")
    assert result == "北京今天晴，适合出行。"
    assert calls == [("get_weather", "北京")]
    assert len(llm.calls) == 2


def test_react_deduplicates_similar_query_calls():
    calls = []
    registry = ToolRegistry()

    def search_notes(query):
        calls.append(query)
        return "笔记结果：周一闭馆"

    registry.register_function(
        "search_notes",
        "检索旅行笔记",
        search_notes,
        [ToolParameter("query", "string", "检索词", True)],
    )
    llm = ScriptedLLM(
        [
            'Thought: 查笔记\nAction: search_notes(query="故宫闭馆")',
            'Thought: 换个说法再查\nAction: search_notes(query="故宫 周一 闭馆")',
            "Thought: 够了\nAction: Finish[周一闭馆]",
        ]
    )
    agent = ReActAgent("travel", llm, tools=registry)
    assert agent.run("故宫周几闭馆") == "周一闭馆"
    assert calls == ["故宫闭馆"]
    second_observation = llm.calls[2]["messages"][-1]["content"]
    assert "重复查询已跳过" in second_observation


def test_react_finish_takes_priority_over_tool_lookup():    # 答案里含工具名 get_weather，但 Finish 检查在前，不应触发工具查找
    llm = ScriptedLLM(["Thought: 总结\nAction: Finish[用 get_weather 工具查到晴天]"])
    agent = ReActAgent("travel", llm, tools=ToolRegistry())  # 空注册表，工具不存在
    assert agent.run("x") == "用 get_weather 工具查到晴天"


def test_react_finish_multiline_answer_preserved():
    llm = ScriptedLLM(["Thought: 总结\nAction: Finish[第一点：晴\n第二点：适合出行]"])
    agent = ReActAgent("t", llm)
    assert agent.run("x") == "第一点：晴\n第二点：适合出行"


def test_react_unknown_tool_feeds_observation():
    llm = ScriptedLLM(
        [
            'Thought: 试一下\nAction: no_such_tool(x="1")',
            "Thought: 换个方式\nAction: Finish[结束]",
        ]
    )
    agent = ReActAgent("t", llm, tools=ToolRegistry())
    assert agent.run("x") == "结束"
    # 第一轮的 Observation 里应包含「未定义的工具」
    obs = llm.calls[1]["messages"][-1]["content"]
    assert "未定义的工具" in obs


def test_react_max_steps_raises():
    llm = ScriptedLLM(["garbage"], default="garbage")
    agent = ReActAgent("t", llm)
    with pytest.raises(AgentError):
        agent.run("x")
