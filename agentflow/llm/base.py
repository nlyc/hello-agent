"""Model Layer 协议：可替换、可测试的核心契约。"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence


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
