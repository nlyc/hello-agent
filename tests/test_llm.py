"""LLM 层单元测试（用假客户端，不联网）。"""

import pytest

from agentflow.core import Message
from agentflow.core.exceptions import ConfigError, LLMError
from agentflow.llm import BaseLLM, ModelRegistry, ModelSpec, OpenAICompatibleLLM
from tests.fakes import FakeClient


def make_llm(**kwargs):
    defaults = dict(
        api_key="sk-test",
        base_url="https://relay.example.com/v1",
        model="deepseek-chat",
    )
    defaults.update(kwargs)
    return OpenAICompatibleLLM(**defaults)


def test_base_llm_is_abstract():
    with pytest.raises(TypeError):
        BaseLLM()


def test_chat_returns_content():
    client = FakeClient(["你好"])
    llm = make_llm(client=client)
    assert llm.chat([Message(role="user", content="hi")]) == "你好"
    assert client.completions.calls[0]["model"] == "deepseek-chat"


def test_chat_accepts_dict_messages():
    llm = make_llm(client=FakeClient(["ok"]))
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok"


def test_chat_empty_content_raises():
    llm = make_llm(client=FakeClient(["   "]))
    with pytest.raises(LLMError):
        llm.chat([Message(role="user", content="hi")])


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    llm = OpenAICompatibleLLM(api_key="", base_url="https://x/v1")
    with pytest.raises(LLMError):
        llm.chat([Message(role="user", content="hi")])


def test_json_mode_retries_without_response_format():
    client = FakeClient([RuntimeError("不支持 response_format"), '{"a": 1}'])
    llm = make_llm(client=client)
    assert llm.chat([Message(role="user", content="hi")], json_mode=True) == '{"a": 1}'
    assert "response_format" in client.completions.calls[0]
    assert "response_format" not in client.completions.calls[1]


def test_use_model_via_registry():
    registry = ModelRegistry(
        {"glm": ModelSpec(model="glm-4-plus"), "openai": ModelSpec(model="gpt-4o")}
    )
    llm = make_llm(registry=registry)
    llm.use_model("glm")
    assert llm.model == "glm-4-plus"


def test_use_model_raw_name():
    llm = make_llm()
    llm.use_model("gpt-4o")
    assert llm.model == "gpt-4o"


def test_registry_unknown_alias_raises():
    with pytest.raises(ConfigError):
        ModelRegistry().get("nope")


def test_registry_register_and_list():
    r = ModelRegistry()
    r.register("deepseek", "deepseek-chat")
    assert "deepseek" in r
    assert r.list_aliases() == ["deepseek"]
    assert r.get("deepseek").model == "deepseek-chat"
