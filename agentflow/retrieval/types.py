"""检索层数据模型：文档、分块、检索命中与入库/回答结果。

本模块只做纯数据承载与轻量转换，不依赖向量库或模型，便于 M8.2 的加载器、
分块器、向量库和 RAGPipeline 共用同一套类型。
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, Sequence


def _iso(value: datetime) -> str:
    """把 datetime 序列化为 ISO 8601 字符串。"""
    return value.isoformat(timespec="seconds")


def _embedding_to_dict(value: bytes | Sequence[float] | None) -> Any:
    """把向量转换为可 JSON 序列化的形式（bytes 用 base64）。"""
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if value is None:
        return None
    return list(value)


def matches_filters(
    filters: dict[str, Any] | None,
    *,
    namespace: str = "",
    metadata: dict[str, Any] | None = None,
    doc_id: str = "",
) -> bool:
    """判断一条记录是否满足过滤条件。

    约定：
    - ``namespace`` / ``doc_id`` 是保留键，直接匹配记录字段；
    - 其余键从 ``metadata`` 中取值；
    - 期望值为列表/元组/集合时，实际值命中任意一项即通过。
    """
    if not filters:
        return True
    data = metadata or {}
    for key, expected in filters.items():
        if key == "namespace":
            actual = namespace
        elif key == "doc_id":
            actual = doc_id
        else:
            actual = data.get(key)
        if isinstance(expected, (list, tuple, set, frozenset)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def citation_for_chunk(chunk: "Chunk") -> str:
    """根据 chunk 的来源、页码和标题路径生成引用串。"""
    metadata = chunk.metadata or {}
    source = (
        metadata.get("source")
        or metadata.get("source_uri")
        or metadata.get("title")
        or chunk.doc_id
    )
    parts = [str(source)] if source else []
    if chunk.page is not None:
        parts.append(f"p.{chunk.page}")
    if chunk.heading_path:
        parts.append(chunk.heading_path)
    return " ".join(parts)


@dataclass
class Document:
    """一篇待入库文档的元数据（正文可放在 ``text`` 中）。"""

    doc_id: str
    namespace: str = "travel"
    source_type: str = "file"
    source_uri: str = ""
    title: str = ""
    content_hash: str = ""
    mime: str = ""
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""
        return {
            "doc_id": self.doc_id,
            "namespace": self.namespace,
            "source_type": self.source_type,
            "source_uri": self.source_uri,
            "title": self.title,
            "content_hash": self.content_hash,
            "mime": self.mime,
            "text": self.text,
            "metadata": dict(self.metadata),
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }


@dataclass
class Chunk:
    """文档分块。

    ``embedding`` 在内存中可以是 float 列表或 float32 BLOB；写入 SQLite 时
    由向量库统一转换为 little-endian float32 BLOB。
    """

    chunk_id: str
    doc_id: str
    namespace: str
    text: str
    heading_path: str = ""
    page: int | None = None
    token_count: int = 0
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: bytes | Sequence[float] | None = None
    embedding_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典（向量转 base64）。"""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "namespace": self.namespace,
            "text": self.text,
            "heading_path": self.heading_path,
            "page": self.page,
            "token_count": self.token_count,
            "content_hash": self.content_hash,
            "metadata": dict(self.metadata),
            "embedding": _embedding_to_dict(self.embedding),
            "embedding_fingerprint": self.embedding_fingerprint,
        }


@dataclass
class RetrievalHit:
    """一次检索命中的 chunk 及其分数、来源和引用。"""

    chunk_id: str
    doc_id: str
    text: str
    score: float = 0.0
    source: str = ""
    citation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    namespace: str = ""
    heading_path: str = ""
    page: int | None = None

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        score: float,
        *,
        source: str | None = None,
        citation: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "RetrievalHit":
        """由 chunk 构造检索命中，自动补齐来源和引用串。"""
        merged_metadata = dict(chunk.metadata)
        if metadata:
            merged_metadata.update(metadata)
        resolved_source = source
        if resolved_source is None:
            resolved_source = str(
                merged_metadata.get("source")
                or merged_metadata.get("source_uri")
                or chunk.doc_id
            )
        return cls(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            score=float(score),
            source=resolved_source,
            citation=citation if citation is not None else citation_for_chunk(chunk),
            metadata=merged_metadata,
            content_hash=chunk.content_hash,
            namespace=chunk.namespace,
            heading_path=chunk.heading_path,
            page=chunk.page,
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "citation": self.citation,
            "metadata": dict(self.metadata),
            "content_hash": self.content_hash,
            "namespace": self.namespace,
            "heading_path": self.heading_path,
            "page": self.page,
        }


@dataclass
class IngestResult:
    """文档入库统计。"""

    doc_id: str
    namespace: str
    chunk_count: int = 0
    skipped: int = 0
    source_uri: str = ""
    title: str = ""
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""
        return {
            "doc_id": self.doc_id,
            "namespace": self.namespace,
            "chunk_count": self.chunk_count,
            "skipped": self.skipped,
            "source_uri": self.source_uri,
            "title": self.title,
            "content_hash": self.content_hash,
            "metadata": dict(self.metadata),
        }


@dataclass
class RAGAnswer:
    """带引用的 RAG 回答。"""

    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    hits: list[RetrievalHit] = field(default_factory=list)
    has_answer: bool = True

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": list(self.citations),
            "hits": [hit.to_dict() for hit in self.hits],
            "has_answer": self.has_answer,
        }


class VectorStore(Protocol):
    """向量库协议：SQLite / Qdrant 等后端实现同一组方法。"""

    def upsert(self, chunks: Sequence[Chunk]) -> int:
        """写入或更新 chunk，返回处理条数。"""
        ...

    def search(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        """按余弦相似度召回。"""
        ...

    def delete(self, doc_id: str | None = None, namespace: str | None = None) -> int:
        """删除指定文档或命名空间下的 chunk，返回删除条数。"""
        ...

    def count(self, namespace: str | None = None) -> int:
        """统计 chunk 数量。"""
        ...

    def clear(self, namespace: str | None = None) -> None:
        """清空指定命名空间；为空时清空全部。"""
        ...
