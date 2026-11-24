"""FunctionCallAgent 与原生 tool_calls 解析测试。"""

from __future__ import annotations

import pytest

from agentflow.agents import FunctionCallAgent
from agentflow.core.config import AgentConfig, Config
from agentflow.core.exceptions import AgentError
from agentflow.llm.base import BaseLLM, LLMResponse, ToolCallRequest
from agentflow.llm.openai_compatible import OpenAICompatibleLLM
from agentflow.tools import FunctionTool, ToolParameter, ToolRegistry
from agentflow.trace import Tracer


class FakeToolLLM(BaseLLM):
    """按队列返回 LLMResponse；记录每次调用。"""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, *, temperature=None, json_mode=False, tools=None) -> str:
        return ""

    def chat_with_tools(self, messages, *, tools, temperature=None) -> LLMResponse:
        self.calls.append({"messages": list(messages), "tools": list(tools)})
        if not self.responses:
            return LLMResponse(content="结束")
        return self.responses.pop(0)


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(
        FunctionTool(
            "get_weather",
            "查询城市天气",
            lambda city: f"{city}：晴，18℃",
            [ToolParameter("city", "string", "城市", True)],
        )
    )
    return registry


def test_parallel_tool_calls_and_final_answer():
    tracer = Tracer()
    events = []
    tracer.subscribe(events.append)
    llm = FakeToolLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCallRequest("c1", "get_weather", {"city": "北京"}),
                    ToolCallRequest("c2", "get_weather", {"city": "上海"}),
                ]
            ),
            LLMResponse(content="北京晴，上海晴。"),
        ]
    )
    agent = FunctionCallAgent(name="fc", llm=llm, tools=make_registry(), tracer=tracer)

    assert agent.run("查北京和上海天气") == "北京晴，上海晴。"
    tool_events = [event for event in events if event.type == "tool"]
    assert [event.data["tool"] for event in tool_events] == ["get_weather", "get_weather"]
    assert tool_events[0].data["params"] == {"city": "北京"}

    second_messages = llm.calls[1]["messages"]
    tool_messages = [message for message in second_messages if message["role"] == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[0]["tool_call_id"] == "c1"
    assert "18℃" in tool_messages[1]["content"]


def test_unknown_tool_becomes_observation():
    llm = FakeToolLLM(
        [
            LLMResponse(
                tool_calls=[ToolCallRequest("c1", "missing", {})]
            ),
            LLMResponse(content="工具不可用，直接回答。"),
        ]
    )
    agent = FunctionCallAgent(name="fc", llm=llm, tools=make_registry())
    assert agent.run("问题") == "工具不可用，直接回答。"
    tool_message = [
        message for message in llm.calls[1]["messages"] if message["role"] == "tool"
    ][0]
    assert tool_message["content"].startswith("错误:")


def test_empty_tools_and_max_steps():
    with pytest.raises(AgentError):
        FunctionCallAgent(
            name="fc", llm=FakeToolLLM([]), tools=ToolRegistry()
        ).run("问题")

    llm = FakeToolLLM(
        [
            LLMResponse(
                tool_calls=[ToolCallRequest(f"c{i}", "get_weather", {"city": "北京"})]
            )
            for i in range(3)
        ]
    )
    agent = FunctionCallAgent(
        name="fc",
        llm=llm,
        tools=make_registry(),
        config=Config(agent=AgentConfig(max_steps=2)),
    )
    with pytest.raises(AgentError):
        agent.run("问题")


def test_schemas_are_passed_to_model():
    llm = FakeToolLLM([LLMResponse(content="直接回答")])
    agent = FunctionCallAgent(name="fc", llm=llm, tools=make_registry())
    assert agent.run("你好") == "直接回答"
    schema = llm.calls[0]["tools"][0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "get_weather"
    assert schema["function"]["parameters"]["required"] == ["city"]


class _FakeFunction:
    name = "get_weather"
    arguments = '{"city": "北京"}'


class _FakeToolCall:
    id = "call_1"
    function = _FakeFunction()


class _FakeMessage:
    content = ""
    tool_calls = [_FakeToolCall()]


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeResponse()


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


def test_openai_chat_with_tools_parses_tool_calls():
    client = _FakeClient()
    llm = OpenAICompatibleLLM(api_key="test", client=client)
    response = llm.chat_with_tools(
        [{"role": "user", "content": "北京天气"}],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )
    assert response.content == ""
    assert response.tool_calls == [
        ToolCallRequest("call_1", "get_weather", {"city": "北京"})
    ]
    assert client.chat.completions.kwargs["tool_choice"] == "auto"
    assert client.chat.completions.kwargs["tools"]


def test_base_llm_chat_with_tools_falls_back_to_text():
    class TextLLM(BaseLLM):
        def chat(self, messages, *, temperature=None, json_mode=False, tools=None):
            return "纯文本"

    response = TextLLM().chat_with_tools([], tools=[])
    assert response.content == "纯文本"
    assert response.tool_calls == []
