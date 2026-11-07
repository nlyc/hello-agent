"""LongTermMemory 测试：持久化、用户隔离、增删改查与指纹校验。"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentflow.core.exceptions import EmbeddingMismatchError, MemoryError
from agentflow.memory import LongTermMemory
from agentflow.retrieval import HashingEmbedder


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """返回临时记忆库路径。"""
    return tmp_path / "memory.sqlite3"


def test_cross_instance_persistence(db_path: Path):
    first = LongTermMemory(HashingEmbedder(), db_path)
    memory_id = first.add("用户偏好自然风光和摄影")
    first.close()

    second = LongTermMemory(HashingEmbedder(), db_path)
    try:
        hits = second.search("自然风光")
        assert hits
        assert hits[0].memory_id == memory_id
        assert "自然风光" in hits[0].content
    finally:
        second.close()


def test_user_isolation(db_path: Path):
    alice = LongTermMemory(HashingEmbedder(), db_path, user_id="alice")
    bob = LongTermMemory(HashingEmbedder(), db_path, user_id="bob")
    try:
        alice.add("喜欢自然风光")
        assert bob.count() == 0
        assert bob.search("自然风光") == []
    finally:
        alice.close()
        bob.close()


def test_get_update_remove(db_path: Path):
    memory = LongTermMemory(HashingEmbedder(), db_path)
    try:
        memory_id = memory.add("喜欢江南古镇", importance=0.4)
        hit = memory.get(memory_id)
        assert hit is not None and hit.memory_type == "semantic"

        assert memory.update(memory_id, content="喜欢云南古镇", importance=0.8)
        updated = memory.get(memory_id)
        assert updated is not None
        assert updated.content == "喜欢云南古镇"
        assert updated.importance == 0.8

        assert memory.remove(memory_id) is True
        assert memory.get(memory_id) is None
        assert memory.remove(memory_id) is False
    finally:
        memory.close()


def test_clear_only_current_user(db_path: Path):
    alice = LongTermMemory(HashingEmbedder(), db_path, user_id="alice")
    bob = LongTermMemory(HashingEmbedder(), db_path, user_id="bob")
    try:
        alice.add("喜欢自然风光")
        alice.add("偏好摄影")
        bob.add("喜欢美食")
        assert alice.clear() == 2
        assert alice.count() == 0
        assert bob.count() == 1
    finally:
        alice.close()
        bob.close()


def test_metadata_and_stats(db_path: Path):
    memory = LongTermMemory(HashingEmbedder(), db_path)
    try:
        memory.add(
            "用户对海鲜过敏",
            memory_type="semantic",
            metadata={"source": "profile"},
        )
        memory.add("上周去过杭州", memory_type="episodic")
        hit = memory.search("海鲜过敏")[0]
        assert hit.metadata == {"source": "profile"}
        stats = memory.stats()
        assert stats["memories"] == 2
        assert stats["types"] == {"episodic": 1, "semantic": 1}
        assert stats["dimension"] == 512
        assert stats["fingerprint"] == HashingEmbedder().fingerprint
    finally:
        memory.close()


def test_fingerprint_mismatch_is_rejected(db_path: Path):
    first = LongTermMemory(HashingEmbedder(), db_path)
    first.add("喜欢自然风光")
    first.close()

    other = LongTermMemory(HashingEmbedder(dimension=256), db_path)
    try:
        with pytest.raises(EmbeddingMismatchError):
            other.add("喜欢摄影")
        with pytest.raises(EmbeddingMismatchError):
            other.search("自然风光")
    finally:
        other.close()


def test_empty_content_and_user_validation(db_path: Path):
    memory = LongTermMemory(HashingEmbedder(), db_path)
    try:
        with pytest.raises(MemoryError):
            memory.add("   ")
    finally:
        memory.close()
    with pytest.raises(MemoryError):
        LongTermMemory(HashingEmbedder(), db_path, user_id="")


def test_search_updates_access_stats(db_path: Path):
    memory = LongTermMemory(HashingEmbedder(), db_path)
    try:
        memory_id = memory.add("喜欢自然风光")
        hit = memory.search("自然风光")[0]
        assert hit.access_count == 1
        assert memory.get(memory_id).access_count == 1
        memory.search("自然风光")
        assert memory.get(memory_id).access_count == 2
    finally:
        memory.close()
