"""BM25 稀疏索引：中文按字 bigram、英文按词，支持 namespace / metadata 过滤。"""

from __future__ import annotations

import math
import threading
from collections import Counter
from typing import Any, Sequence

from agentflow.core.exceptions import RetrievalError
from agentflow.retrieval.embedder import tokenize
from agentflow.retrieval.types import Chunk, RetrievalHit, matches_filters


class BM25Index:
    """内存 BM25 索引。

    参数采用常用默认值 ``k1=1.5``、``b=0.75``。每次 ``search`` 会在过滤后的
    候选集上重新计算文档频率与平均长度，保证过滤条件不会破坏排序口径。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise RetrievalError("BM25 的 k1 必须大于 0")
        if not 0.0 <= b <= 1.0:
            raise RetrievalError("BM25 的 b 必须在 [0, 1] 区间")
        self.k1 = float(k1)
        self.b = float(b)
        self._chunks: dict[str, Chunk] = {}
        self._tokens: dict[str, list[str]] = {}
        self._lock = threading.RLock()

    def add(self, chunks: Sequence[Chunk]) -> None:
        """加入或更新一批 chunk；相同 ``chunk_id`` 以最后一次为准。"""
        with self._lock:
            for chunk in chunks:
                if not chunk.chunk_id:
                    raise RetrievalError("BM25 索引要求 chunk_id 非空")
                text = chunk.text
                if chunk.heading_path:
                    text = f"{text}\n{chunk.heading_path}"
                self._chunks[chunk.chunk_id] = chunk
                self._tokens[chunk.chunk_id] = tokenize(text)

    def remove(self, chunk_id: str) -> bool:
        """按 chunk_id 删除，返回是否确实删除。"""
        with self._lock:
            self._tokens.pop(chunk_id, None)
            return self._chunks.pop(chunk_id, None) is not None

    def remove_doc(self, doc_id: str, namespace: str | None = None) -> int:
        """删除指定文档（可限定命名空间）的全部 chunk，返回删除条数。"""
        with self._lock:
            targets = [
                chunk_id
                for chunk_id, chunk in self._chunks.items()
                if chunk.doc_id == doc_id
                and (namespace is None or chunk.namespace == namespace)
            ]
            for chunk_id in targets:
                self._chunks.pop(chunk_id, None)
                self._tokens.pop(chunk_id, None)
            return len(targets)

    def clear(self) -> None:
        """清空索引。"""
        with self._lock:
            self._chunks.clear()
            self._tokens.clear()

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        """检索并返回按 BM25 分数降序排列的命中。"""
        if top_k <= 0:
            return []
        query_terms = list(dict.fromkeys(tokenize(query)))
        if not query_terms:
            return []

        with self._lock:
            candidates = [
                (chunk_id, chunk, self._tokens.get(chunk_id, []))
                for chunk_id, chunk in self._chunks.items()
                if matches_filters(
                    filters,
                    namespace=chunk.namespace,
                    metadata=chunk.metadata,
                    doc_id=chunk.doc_id,
                )
            ]
        if not candidates:
            return []

        doc_freq: Counter[str] = Counter()
        for _, _, tokens in candidates:
            doc_freq.update(set(tokens))
        total_docs = len(candidates)
        avg_length = sum(len(tokens) for _, _, tokens in candidates) / total_docs
        if avg_length <= 0:
            avg_length = 1.0

        scored: list[RetrievalHit] = []
        for _, chunk, tokens in candidates:
            length = len(tokens)
            term_freq = Counter(tokens)
            score = 0.0
            for term in query_terms:
                frequency = term_freq.get(term, 0)
                if frequency <= 0:
                    continue
                df = doc_freq.get(term, 0)
                idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / avg_length
                )
                score += idf * frequency * (self.k1 + 1.0) / denominator
            if score > 0:
                scored.append(RetrievalHit.from_chunk(chunk, score))

        scored.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return scored[:top_k]

    def __len__(self) -> int:
        """返回索引中的 chunk 数量。"""
        with self._lock:
            return len(self._chunks)
