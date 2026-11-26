"""MemoryManager 测试：上下文合并、轮次召回、开关与失败降级。"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentflow.core.config import MemoryConfig
from agentflow.core.exceptions import MemoryError
from agentflow.memory import LongTermMemory, MemoryManager, WorkingMemory
from agentflow.retrieval import HashingEmbedder


@pytest.fixture()
def manager(tmp_path: Path):
    """构建默认离线记忆管理器。"""
    store = LongTermMemory(HashingEmbedder(), tmp_path / "memory.sqlite3")
    instance = MemoryManager(HashingEmbedder(), store)
    try:
        yield instance
    finally:
        store.close()


def test_build_context_contains_preference(manager: MemoryManager):
    manager.add("用户偏好自然风光和摄影", importance=0.9, metadata={"kind": "preference"})
    context = manager.build_context("推荐自然风光路线")
    assert context.enabled is True
    assert "自然风光" in context.text
    assert any(item["source"] in {"working", "long_term"} for item in context.items)
    assert context.to_dict()["text"] == context.text


def test_build_context_respects_token_budget(manager: MemoryManager):
    manager.add("用户偏好自然风光")
    context = manager.build_context("自然风光", max_tokens=5)
    assert len(context.text) < 60
    assert manager.build_context("自然风光", max_tokens=0).text == ""


def test_record_turn_can_be_recalled(manager: MemoryManager):
    manager.record_turn("我喜欢自然风光和摄影", "已记录，后续优先推荐山水路线。")
    context = manager.build_context("自然风光")
    assert "自然风光" in context.text
    assert manager.long_term.count() == 1


def test_manager_search_merges_sources(manager: MemoryManager):
    manager.add("用户对海鲜过敏", importance=0.8)
    entries = manager.search("海鲜过敏")
    assert entries
    assert {entry["source"] for entry in entries} <= {"working", "long_term"}
    assert entries[0]["content"] == "用户对海鲜过敏"


def test_disabled_memory_returns_disabled_context(tmp_path: Path):
    store = LongTermMemory(HashingEmbedder(), tmp_path / "memory.sqlite3")
    try:
        config = MemoryConfig(enabled=False)
        manager = MemoryManager(HashingEmbedder(), store, config=config)
        assert manager.add("用户偏好自然风光") == ""
        context = manager.build_context("自然风光")
        assert context.enabled is False
        assert context.text == ""
    finally:
        store.close()


class _BrokenSearch:
    """检索必定失败的测试替身。"""

    user_id = "default"

    def search(self, *args, **kwargs):
        """模拟底层失败。"""
        raise RuntimeError("backend down")

    def stats(self):
        """模拟统计失败。"""
        raise RuntimeError("backend down")


def test_build_context_degrades_on_substep_failure():
    manager = MemoryManager(
        HashingEmbedder(),
        _BrokenSearch(),  # type: ignore[arg-type]
        working=_BrokenSearch(),  # type: ignore[arg-type]
    )
    context = manager.build_context("自然风光")
    assert context.enabled is True
    assert context.text == ""
    assert context.working_items == []
    assert context.long_term_hits == []
    assert "error" in manager.stats()["working"]
    assert "error" in manager.stats()["long_term"]


def test_user_id_must_match_store(tmp_path: Path):
    store = LongTermMemory(HashingEmbedder(), tmp_path / "memory.sqlite3", user_id="alice")
    try:
        with pytest.raises(MemoryError):
            MemoryManager(HashingEmbedder(), store, user_id="bob")
    finally:
        store.close()


def test_stats_and_clear(manager: MemoryManager):
    manager.add("用户偏好自然风光")
    stats = manager.stats()
    assert stats["enabled"] is True
    assert stats["working"]["items"] == 1
    assert stats["long_term"]["memories"] == 1

    manager.clear()
    assert manager.long_term.count() == 0
    assert len(manager.working) == 0


def test_custom_working_memory(tmp_path: Path):
    store = LongTermMemory(HashingEmbedder(), tmp_path / "memory.sqlite3")
    working = WorkingMemory(max_items=1)
    try:
        manager = MemoryManager(HashingEmbedder(), store, working=working)
        manager.add("第一条")
        manager.add("第二条")
        assert len(manager.working) == 1
    finally:
        store.close()
