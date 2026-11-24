"""上下文打分：关键词相关性、新近性指数衰减与综合分。

默认相关性使用与 BM25 / HashingEmbedder 同一套分词（CJK bigram + 英文词），
保证稀疏检索与上下文选择的口径一致；生产环境可替换为向量余弦。
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from agentflow.context.types import ContextConfig, ensure_aware, utc_now
from agentflow.retrieval.embedder import tokenize


class RelevanceScorer(Protocol):
    """相关性打分协议：返回 ``[0, 1]``。"""

    def score(self, query: str, content: str) -> float:
        """计算 query 与 content 的相关性。"""
        ...


class JaccardScorer:
    """Jaccard 关键词重叠：集合交集 / 并集。"""

    def score(self, query: str, content: str) -> float:
        return jaccard_similarity(query, content)


def jaccard_similarity(query: str, content: str) -> float:
    """CJK bigram + 英文词口径的 Jaccard 相似度。"""
    query_terms = set(tokenize(str(query or "")))
    content_terms = set(tokenize(str(content or "")))
    if not query_terms or not content_terms:
        return 0.0
    union = query_terms | content_terms
    if not union:
        return 0.0
    return len(query_terms & content_terms) / len(union)


def recency_score(
    timestamp: datetime,
    *,
    now: datetime | None = None,
    half_life_hours: float = 72.0,
    floor: float = 0.1,
) -> float:
    """指数衰减的新近性分数：半衰期后降为 0.5，并设下限。

    参数：
        timestamp: 信息产生时间。
        now: 当前时间；为空时取 UTC 现在。
        half_life_hours: 半衰期（小时），必须大于 0。
        floor: 最低分，避免久远信息被完全抹掉。
    """
    if half_life_hours <= 0:
        raise ValueError("half_life_hours 必须大于 0")
    current = ensure_aware(now or utc_now())
    created = ensure_aware(timestamp)
    age_hours = max(0.0, (current - created).total_seconds() / 3600.0)
    value = 0.5 ** (age_hours / float(half_life_hours))
    return max(float(floor), min(1.0, value))


def combine_score(
    relevance: float,
    recency: float,
    config: ContextConfig,
) -> float:
    """综合分 = 相关性权重 × 相关性 + 新近性权重 × 新近性。"""
    return (
        float(config.relevance_weight) * max(0.0, min(1.0, float(relevance)))
        + float(config.recency_weight) * max(0.0, min(1.0, float(recency)))
    )
