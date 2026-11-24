"""Model Layer 协议：可替换、可测试的核心契约。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


@dataclass
class ToolCallRequest:
    """模型请求执行的一次工具调用（原生 Function Calling）。"""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """带原生工具调用的模型响应。"""

    content: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_assistant_message(self) -> dict[str, Any]:
        """转成 OpenAI assistant 消息（含 tool_calls）。"""
        message: dict[str, Any] = {
            "role": "assistant",
            "content": self.content or "",
        }
        if self.tool_calls:
            import json

            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments, ensure_ascii=False
                        ),
                    },
                }
                for call in self.tool_calls
            ]
        return message


class BaseLLM(ABC):
    """LLM 统一接口。所有范式 / 工作流只依赖本协议，实现可替换（测试注入 fake 靠它）。"""

    @abstractmethod
    def chat(
        self,
        messages: Sequence[Any],
        *,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        tools: Optional[list[dict]] = None,
    ) -> str:
        """调用一次对话模型，返回纯文本。

        messages 每项可为 Message 或 OpenAI 风格 dict；
        json_mode 要求服务端返回 JSON 对象；tools 为原生 function calling 预留。
        """
        ...

    def chat_with_tools(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[dict],
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """带原生工具调用的对话。

        默认实现回退到 ``chat``（无工具调用），保证旧实现无需改动；
        ``OpenAICompatibleLLM`` 覆盖本方法以返回结构化 ``tool_calls``。
        """
        content = self.chat(
            messages,
            temperature=temperature,
            tools=list(tools) if tools else None,
        )
        return LLMResponse(content=content, tool_calls=[])
