"""WorkingMemory 测试：TTL、容量淘汰、Token 裁剪与重要性加权。"""

from __future__ import annotations

import pytest

from agentflow.core.exceptions import MemoryError
from agentflow.memory import WorkingMemory, estimate_tokens


def test_expired_items_are_dropped():
    memory = WorkingMemory(ttl_minutes=0)
    memory.add("自然风光")
    assert len(memory) == 0
    assert memory.get_context("自然风光") == ""


def test_max_items_keeps_highest_scoring():
    memory = WorkingMemory(max_items=2)
    memory.add("第一条", importance=0.1)
    memory.add("第二条", importance=0.9)
    memory.add("第三条", importance=0.9)
    contents = [item.content for item in memory.all_items()]
    assert "第一条" not in contents
    assert len(contents) == 2
    assert len(memory) == 2


def test_token_budget_trims_long_items():
    memory = WorkingMemory()
    memory.add("宜人的自然风光和摄影路线", importance=0.9)
    memory.add("故宫", importance=0.1)
    text = memory.get_context("", max_tokens=10)
    assert text
    assert estimate_tokens(text) <= 10
    assert "宜人的自然风光和摄影路线" not in text
    assert "故宫" in text


def test_importance_weight_changes_order():
    memory = WorkingMemory()
    memory.add("自然风光", importance=0.1)
    memory.add("自然风光", importance=0.9)
    items = memory.search("自然风光", top_k=2)
    assert [item.importance for item in items] == [0.9, 0.1]
    assert items[0].score > items[1].score


def test_relevance_changes_order():
    memory = WorkingMemory()
    memory.add("北京烤鸭推荐全聚德")
    memory.add("自然风光和摄影")
    items = memory.search("自然风光", top_k=2)
    assert items[0].content == "自然风光和摄影"
    assert items[0].score > items[1].score


def test_access_tracking_on_get_context():
    memory = WorkingMemory()
    item_id = memory.add("自然风光")
    memory.get_context("自然风光", top_k=1)
    item = next(item for item in memory.all_items() if item.item_id == item_id)
    assert item.access_count == 1


def test_empty_content_and_invalid_params():
    with pytest.raises(MemoryError):
        WorkingMemory().add("   ")
    with pytest.raises(MemoryError):
        WorkingMemory(max_items=0)
    with pytest.raises(MemoryError):
        WorkingMemory(ttl_minutes=-1)


def test_clear_and_stats():
    memory = WorkingMemory(max_items=5)
    memory.add("自然风光")
    stats = memory.stats()
    assert stats == {"items": 1, "max_items": 5, "ttl_minutes": 60.0}
    memory.clear()
    assert memory.get_context("") == ""
    assert memory.stats()["items"] == 0


def test_zero_token_budget_returns_empty():
    memory = WorkingMemory()
    memory.add("自然风光")
    assert memory.get_context("自然风光", max_tokens=0) == ""
    assert memory.search("自然风光", top_k=0) == []
