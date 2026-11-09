"""Config 单元测试。"""

from agentflow.core import Config
from agentflow.core.env import _PUBLIC_KEYS, env_snapshot

_ENV_KEYS = [
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "LLM_TIMEOUT",
    "AGENT_MAX_STEPS",
    "AGENT_MAX_TOOL_ITERATIONS",
    "MAX_HISTORY_LENGTH",
    "DEBUG",
]


def test_zero_config_defaults():
    c = Config()
    assert c.llm.model == "deepseek-chat"
    assert c.llm.temperature == 0.2
    assert c.agent.max_steps == 6
    assert c.debug is False


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "glm-4-plus")
    monkeypatch.setenv("LLM_BASE_URL", "https://relay.example.com/v1")
    monkeypatch.setenv("AGENT_MAX_STEPS", "10")
    monkeypatch.setenv("DEBUG", "true")
    c = Config.from_env()
    assert c.llm.model == "glm-4-plus"
    assert c.llm.base_url == "https://relay.example.com/v1"
    assert c.agent.max_steps == 10
    assert c.debug is True


def test_from_env_defaults_when_unset(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    c = Config.from_env()
    assert c.llm.model == "deepseek-chat"
    assert c.llm.base_url == ""
    assert c.agent.max_steps == 6


def test_env_snapshot_only_exposes_whitelist_and_masks_secrets(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "glm-4-plus")
    monkeypatch.setenv("LLM_API_KEY", "sk-secret-value")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret")
    monkeypatch.setenv("QDRANT_API_KEY", "qd-secret")
    # 不在白名单中的系统变量不应出现在快照里。
    monkeypatch.setenv("AGENTFLOW_NOT_WHITELISTED", "hidden")

    snapshot = env_snapshot()

    assert set(snapshot) <= _PUBLIC_KEYS
    assert snapshot["LLM_MODEL"] == "glm-4-plus"
    assert snapshot["LLM_API_KEY"] == "***"
    assert snapshot["TAVILY_API_KEY"] == "***"
    assert snapshot["QDRANT_API_KEY"] == "***"
    assert "AGENTFLOW_NOT_WHITELISTED" not in snapshot
    assert "sk-secret-value" not in snapshot.values()
    assert "tvly-secret" not in snapshot.values()
    assert "qd-secret" not in snapshot.values()
