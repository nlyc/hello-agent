"""测试全局隔离：不读取开发者本地 `.env`，保证结果可复现。

Web 控制台等入口会在导入时加载项目根目录 `.env`。如果测试依赖开发者
本地的在线模型 / Qdrant 配置，整套测试就会随环境变化。这里在每个用例前
屏蔽 `.env` 并移除其中注入的变量；用例内部仍可用 ``monkeypatch.setenv``
显式构造配置。
"""

from __future__ import annotations

import pytest

import agentflow.core.env as env


@pytest.fixture(autouse=True)
def isolate_dotenv(monkeypatch):
    """忽略 `.env` 文件，只保留显式设置的环境变量。"""
    monkeypatch.setattr(env, "find_env_file", lambda: None)
    for key in list(env._loaded_keys):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(env, "_loaded_keys", set())
    monkeypatch.setattr(env, "_loaded_path", None)
    monkeypatch.setattr(env, "_loaded_mtime", None)
    yield
    # monkeypatch 会在用例结束后恢复环境变量与模块属性。
