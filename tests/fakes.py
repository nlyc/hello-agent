"""测试替身：假 OpenAI 客户端与可编排 LLM，用于不联网测试 LLM 层与 Agent 循环。"""

from agentflow.llm.base import BaseLLM


class ScriptedLLM(BaseLLM):
    """按队列顺序返回预设回复的 BaseLLM，并记录每次调用的入参，供 Agent 循环测试。"""

    def __init__(self, responses=None, default=""):
        self._responses = list(responses or [])
        self.default = default
        self.calls: list[dict] = []

    def chat(self, messages, *, temperature=None, json_mode=False, tools=None):
        self.calls.append(
            {
                "messages": list(messages),
                "temperature": temperature,
                "json_mode": json_mode,
                "tools": tools,
            }
        )
        if not self._responses:
            return self.default
        return self._responses.pop(0)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    """记录每次 create 的 kwargs，按队列顺序返回内容或抛错。"""

    def __init__(self, responses=None):
        self.calls: list[dict] = []
        self._responses = list(responses or [])

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeCompletions 未编程响应")
        next_item = self._responses.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return FakeResponse(next_item)


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    """模仿 openai.OpenAI：暴露 chat.completions.create(...)。"""

    def __init__(self, responses=None):
        self.completions = FakeCompletions(responses)
        self.chat = FakeChat(self.completions)
