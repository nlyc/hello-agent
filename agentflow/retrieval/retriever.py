"""混合检索：dense 向量召回 + BM25 稀疏召回 + RRF 融合。"""

from __future__ import annotations

from typing import Any

from agentflow.core.exceptions import RetrievalError
from agentflow.retrieval.bm25 import BM25Index
from agentflow.retrieval.embedder import Embedder
from agentflow.retrieval.types import RetrievalHit, VectorStore


class HybridRetriever:
    """RRF 混合检索器。

    候选池大小为 ``top_k * candidate_multiplier``；两路结果按
    ``score = Σ 1 / (rrf_k + rank)`` 融合，rank 从 1 开始。融合后按
    ``content_hash``（缺失时退回 ``chunk_id``）去重，并保留 top_k。

    ``min_score`` 作用于向量召回的**余弦相似度**（RRF 融合前）：低于该值
    的稠密命中会被丢弃，避免完全不相关的文档仅凭排名进入上下文。默认 0
    表示不启用；不同 Embedder 的分数量纲不同，建议按后端分别配置。
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25: BM25Index,
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
        min_score: float = 0.0,
    ) -> None:
        if rrf_k <= 0:
            raise RetrievalError("RRF 的 rrf_k 必须大于 0")
        if candidate_multiplier < 1:
            raise RetrievalError("candidate_multiplier 必须大于等于 1")
        if min_score < 0:
            raise RetrievalError("min_score 不能为负数")
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25
        self.rrf_k = int(rrf_k)
        self.candidate_multiplier = int(candidate_multiplier)
        # 向量召回的余弦相似度下限；0 表示关闭（保留所有正分命中）。
        self.min_score = float(min_score)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        """执行混合检索并返回融合后的 top_k 命中。"""
        if top_k <= 0:
            return []
        pool_size = top_k * self.candidate_multiplier

        query_vector = self.embedder.embed_query(query)
        dense_hits = self.vector_store.search(
            query_vector, top_k=pool_size, filters=filters
        )
        sparse_hits = self.bm25.search(query, top_k=pool_size, filters=filters)

        # 稠密检索在全零查询下会返回相似度 0 的记录，这类结果不参与融合；
        # min_score 再过滤掉相似度不足的弱相关命中。
        dense_hits = [
            hit for hit in dense_hits if hit.score > 0 and hit.score >= self.min_score
        ]
        return self._fuse([("dense", dense_hits), ("bm25", sparse_hits)], top_k)

    def _fuse(
        self,
        ranked_lists: list[tuple[str, list[RetrievalHit]]],
        top_k: int,
    ) -> list[RetrievalHit]:
        """按 RRF 融合多路有序结果并去重。"""
        records: dict[tuple[str, str], dict[str, Any]] = {}
        for source_name, hits in ranked_lists:
            seen: set[tuple[str, str]] = set()
            rank = 0
            for hit in hits:
                key = _dedup_key(hit)
                if key in seen:
                    continue
                seen.add(key)
                rank += 1
                record = records.get(key)
                if record is None:
                    record = {
                        "hit": hit,
                        "score": 0.0,
                        "sources": [],
                        "raw_scores": {},
                    }
                    records[key] = record
                record["score"] += 1.0 / (self.rrf_k + rank)
                record["sources"].append(source_name)
                record["raw_scores"][source_name] = hit.score

        fused: list[RetrievalHit] = []
        for record in records.values():
            origin: RetrievalHit = record["hit"]
            metadata = dict(origin.metadata)
            metadata["retrieval_sources"] = list(record["sources"])
            metadata["raw_scores"] = dict(record["raw_scores"])
            fused.append(
                RetrievalHit(
                    chunk_id=origin.chunk_id,
                    doc_id=origin.doc_id,
                    text=origin.text,
                    score=float(record["score"]),
                    source=origin.source,
                    citation=origin.citation,
                    metadata=metadata,
                    content_hash=origin.content_hash,
                    namespace=origin.namespace,
                    heading_path=origin.heading_path,
                    page=origin.page,
                )
            )
        fused.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return fused[:top_k]


def _dedup_key(hit: RetrievalHit) -> tuple[str, str]:
    """优先按 content_hash 去重，缺失时退回 chunk_id。"""
    if hit.content_hash:
        return ("hash", hit.content_hash)
    return ("id", hit.chunk_id)
