"""上下文打分测试：Jaccard 相关性、新近性衰减与综合分。"""

from datetime import datetime, timedelta, timezone

import pytest

from agentflow.context import (
    ContextConfig,
    combine_score,
    jaccard_similarity,
    recency_score,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_jaccard_chinese_overlap():
    assert jaccard_similarity("故宫开放时间", "故宫门票和开放时间") > 0
    assert jaccard_similarity("故宫", "上海外滩") == 0.0


def test_jaccard_english_and_empty():
    assert jaccard_similarity("beijing weather", "Beijing weather is sunny") > 0
    assert jaccard_similarity("", "任何内容") == 0.0
    assert jaccard_similarity("内容", "") == 0.0


def test_recency_zero_age_is_one():
    assert recency_score(NOW, now=NOW) == pytest.approx(1.0)


def test_recency_half_life_is_half():
    created = NOW - timedelta(hours=72)
    assert recency_score(created, now=NOW, half_life_hours=72) == pytest.approx(0.5)


def test_recency_floor_and_future():
    old = NOW - timedelta(days=365)
    assert recency_score(old, now=NOW) == pytest.approx(0.1)
    # 未来时间按年龄 0 处理，封顶 1.0。
    assert recency_score(NOW + timedelta(hours=1), now=NOW) == pytest.approx(1.0)


def test_recency_rejects_invalid_half_life():
    with pytest.raises(ValueError):
        recency_score(NOW, now=NOW, half_life_hours=0)


def test_combine_score_uses_weights():
    config = ContextConfig(relevance_weight=0.7, recency_weight=0.3)
    assert combine_score(1.0, 0.0, config) == pytest.approx(0.7)
    assert combine_score(0.0, 1.0, config) == pytest.approx(0.3)
    assert combine_score(0.5, 0.5, config) == pytest.approx(0.5)
