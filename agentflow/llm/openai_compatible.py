"""公司中转站实现：统一 OpenAI 兼容接口，换 model 即切换。"""

from typing import Any, Optional, Sequence

from openai import OpenAI

from agentflow.core.env import get_env
from agentflow.core.exceptions import LLMError
from agentflow.core.message import Message
from agentflow.llm.base import BaseLLM
from agentflow.llm.registry import ModelRegistry


def _to_openai_message(item) -> dict[str, Any]:
    if isinstance(item, Message):
        return item.to_dict()
    if isinstance(item, dict):
        return item
    raise TypeError(f"消息项须为 Message 或 dict，收到 {type(item)!r}")


class OpenAICompatibleLLM(BaseLLM):
    """基于 OpenAI 兼容网关（公司中转站）的模型客户端。

    base_url / api_key / model 优先取构造参数，其次 .env / 环境变量
    LLM_BASE_URL / LLM_API_KEY / LLM_MODEL，模型默认 deepseek-chat；
    三家云端（DeepSeek / GLM / OpenAI）只是不同的 model 名。
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        timeout: float = 60.0,
        registry: Optional[ModelRegistry] = None,
        client: Optional[Any] = None,
    ):
        self.base_url = base_url or get_env("LLM_BASE_URL")
        self.api_key = api_key or get_env("LLM_API_KEY")
        self.model = model or get_env("LLM_MODEL", "deepseek-chat")
        self.temperature = temperature
        self.timeout = timeout
        self._registry = registry
        self._client = client  # 测试注入的假客户端；None 则惰性创建真实客户端
        self._owns_client = client is None

    def _get_client(self):
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
        return self._client

    def chat(
        self,
        messages: Sequence[Any],
        *,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        tools: Optional[list[dict]] = None,
    ) -> str:
        if not self.api_key:
            raise LLMError("未配置 LLM_API_KEY")

        payload = [_to_openai_message(m) for m in messages]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "temperature": self.temperature if temperature is None else temperature,
            "stream": False,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools

        def _create(use_json_mode: bool) -> str:
            call_kwargs = dict(kwargs)
            if not use_json_mode:
                call_kwargs.pop("response_format", None)
            response = self._get_client().chat.completions.create(**call_kwargs)
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise LLMError("模型返回了空内容")
            return content.strip()

        try:
            return _create(json_mode)
        except LLMError:
            raise
        except Exception as first_error:
            # 个别网关不支持 response_format，去掉后重试一次，避免整链路崩溃。
            if json_mode:
                try:
                    return _create(False)
                except Exception:
                    pass
            raise LLMError(f"调用模型失败: {first_error}") from first_error

    def use_model(self, alias: str) -> None:
        """切换到指定模型：注册表别名解析到 model（+可选 key/base_url），否则当作裸模型名。"""
        old_base, old_key = self.base_url, self.api_key
        if self._registry and alias in self._registry:
            spec = self._registry.get(alias)
            self.model = spec.model
            if spec.api_key:
                self.api_key = spec.api_key
            if spec.base_url:
                self.base_url = spec.base_url
        else:
            self.model = alias
        if self._owns_client and (self.base_url != old_base or self.api_key != old_key):
            self._client = None  # 网关地址或密钥变了，重建客户端
