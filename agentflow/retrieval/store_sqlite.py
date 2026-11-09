"""SQLite 向量库：零额外依赖、可持久化、支持 namespace 与 metadata 过滤。"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Any, Sequence

from agentflow.core.exceptions import EmbeddingMismatchError, RetrievalError
from agentflow.retrieval.types import Chunk, RetrievalHit, matches_filters

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    text TEXT NOT NULL,
    heading_path TEXT NOT NULL DEFAULT '',
    page INTEGER,
    token_count INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    embedding BLOB NOT NULL,
    embedding_fingerprint TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_chunks_namespace ON chunks(namespace);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
"""


def pack_vector(vector: Sequence[float]) -> bytes:
    """把浮点向量打包为 little-endian float32 BLOB。"""
    return struct.pack(f"<{len(vector)}f", *[float(value) for value in vector])


def unpack_vector(blob: bytes) -> list[float]:
    """把 float32 BLOB 解包为浮点列表。"""
    if len(blob) % 4 != 0:
        raise RetrievalError("向量 BLOB 长度不是 4 的整数倍")
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def _as_vector(value: bytes | Sequence[float] | None) -> list[float]:
    """把 bytes 或浮点序列统一成浮点列表。"""
    if value is None:
        raise RetrievalError("chunk 缺少 embedding，无法写入向量库")
    if isinstance(value, bytes):
        return unpack_vector(value)
    return [float(item) for item in value]


def _normalize(vector: Sequence[float]) -> list[float]:
    """L2 归一化；全零向量保持原样。"""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return [float(value) for value in vector]
    return [float(value) / norm for value in vector]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """纯 Python 点积（归一化后即余弦相似度）。"""
    return float(sum(a * b for a, b in zip(left, right)))


class SQLiteVectorStore:
    """基于 SQLite 的持久化向量库。

    向量以 little-endian float32 BLOB 存储，写入时归一化，检索时用点积
    代替余弦相似度。默认开启 WAL，所有读写共用连接并由 ``RLock`` 保护。
    """

    def __init__(self, path: str | Path = ".data/rag.sqlite3") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def upsert(self, chunks: Sequence[Chunk]) -> int:
        """写入或更新一批 chunk，返回处理条数。"""
        prepared: list[tuple[Chunk, list[float]]] = []
        for chunk in chunks:
            if not chunk.chunk_id:
                raise RetrievalError("向量库要求 chunk_id 非空")
            if not chunk.namespace:
                raise RetrievalError("向量库要求 namespace 非空")
            vector = _normalize(_as_vector(chunk.embedding))
            prepared.append((chunk, vector))
        if not prepared:
            return 0

        self._validate_fingerprints(prepared)
        rows = []
        for chunk, vector in prepared:
            rows.append(
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.namespace,
                    chunk.text,
                    chunk.heading_path,
                    chunk.page,
                    chunk.token_count,
                    chunk.content_hash,
                    json.dumps(chunk.metadata or {}, ensure_ascii=False, default=str),
                    pack_vector(vector),
                    chunk.embedding_fingerprint,
                )
            )
        sql = """
        INSERT INTO chunks (
            chunk_id, doc_id, namespace, text, heading_path, page,
            token_count, content_hash, metadata, embedding, embedding_fingerprint
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            doc_id=excluded.doc_id,
            namespace=excluded.namespace,
            text=excluded.text,
            heading_path=excluded.heading_path,
            page=excluded.page,
            token_count=excluded.token_count,
            content_hash=excluded.content_hash,
            metadata=excluded.metadata,
            embedding=excluded.embedding,
            embedding_fingerprint=excluded.embedding_fingerprint
        """
        with self._lock:
            self._conn.executemany(sql, rows)
            self._conn.commit()
        return len(rows)

    def search(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        """按余弦相似度检索，返回 top_k 命中。"""
        if top_k <= 0:
            return []
        query = _normalize([float(value) for value in vector])
        if not query:
            return []

        namespace = (filters or {}).get("namespace")
        sql = "SELECT * FROM chunks"
        params: list[Any] = []
        if isinstance(namespace, str) and namespace:
            sql += " WHERE namespace = ?"
            params.append(namespace)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        hits: list[RetrievalHit] = []
        for row in rows:
            metadata = _load_metadata(row["metadata"])
            if not matches_filters(
                filters,
                namespace=row["namespace"],
                metadata=metadata,
                doc_id=row["doc_id"],
            ):
                continue
            stored = unpack_vector(row["embedding"])
            if len(stored) != len(query):
                raise EmbeddingMismatchError(
                    f"查询向量维度 {len(query)} 与已存向量维度 {len(stored)} 不一致，"
                    "请确认 Embedder 或重建索引"
                )
            chunk = Chunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                namespace=row["namespace"],
                text=row["text"],
                heading_path=row["heading_path"],
                page=row["page"],
                token_count=row["token_count"],
                content_hash=row["content_hash"],
                metadata=metadata,
                embedding_fingerprint=row["embedding_fingerprint"],
            )
            hits.append(RetrievalHit.from_chunk(chunk, _dot(query, stored)))

        hits.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return hits[:top_k]

    def delete(self, doc_id: str | None = None, namespace: str | None = None) -> int:
        """删除指定文档/命名空间下的 chunk；两者都为空时清空全表。"""
        clauses: list[str] = []
        params: list[Any] = []
        if doc_id is not None:
            clauses.append("doc_id = ?")
            params.append(doc_id)
        if namespace is not None:
            clauses.append("namespace = ?")
            params.append(namespace)
        sql = "DELETE FROM chunks"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self._lock:
            cursor = self._conn.execute(sql, params)
            self._conn.commit()
            return int(cursor.rowcount)

    def count(self, namespace: str | None = None) -> int:
        """统计 chunk 数量。"""
        with self._lock:
            if namespace is None:
                row = self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM chunks WHERE namespace = ?", (namespace,)
                ).fetchone()
        return int(row["n"])

    def clear(self, namespace: str | None = None) -> None:
        """清空指定命名空间；为空时清空全部。"""
        self.delete(namespace=namespace)

    def fingerprint(self, namespace: str | None = None) -> str | None:
        """返回命名空间内已记录的 embedding fingerprint。"""
        sql = (
            "SELECT DISTINCT embedding_fingerprint FROM chunks "
            "WHERE embedding_fingerprint <> ''"
        )
        params: list[Any] = []
        if namespace is not None:
            sql += " AND namespace = ?"
            params.append(namespace)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        values = [row["embedding_fingerprint"] for row in rows]
        return values[0] if values else None

    def content_hashes(
        self, namespace: str | None = None, doc_id: str | None = None
    ) -> set[str]:
        """返回已入库的内容哈希集合，用于跨文档/同文档去重。"""
        clauses = ["content_hash <> ''"]
        params: list[Any] = []
        if namespace is not None:
            clauses.append("namespace = ?")
            params.append(namespace)
        if doc_id is not None:
            clauses.append("doc_id = ?")
            params.append(doc_id)
        sql = "SELECT DISTINCT content_hash FROM chunks WHERE " + " AND ".join(clauses)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return {row["content_hash"] for row in rows}

    def list_chunks(self, namespace: str | None = None) -> list[Chunk]:
        """列出命名空间内的全部 chunk（不含向量），用于重建 BM25 等内存索引。"""
        sql = "SELECT * FROM chunks"
        params: list[Any] = []
        if namespace is not None:
            sql += " WHERE namespace = ?"
            params.append(namespace)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            Chunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                namespace=row["namespace"],
                text=row["text"],
                heading_path=row["heading_path"],
                page=row["page"],
                token_count=row["token_count"],
                content_hash=row["content_hash"],
                metadata=_load_metadata(row["metadata"]),
                embedding_fingerprint=row["embedding_fingerprint"],
            )
            for row in rows
        ]

    def close(self) -> None:
        """关闭数据库连接（可重复调用）。"""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> "SQLiteVectorStore":
        """支持 ``with`` 上下文。"""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """退出上下文时关闭连接。"""
        self.close()

    def _validate_fingerprints(self, prepared: Sequence[tuple[Chunk, list[float]]]) -> None:
        """校验同一命名空间内 embedding fingerprint 保持一致。"""
        batch: dict[str, str] = {}
        for chunk, _ in prepared:
            fingerprint = chunk.embedding_fingerprint
            if not fingerprint:
                continue
            previous = batch.get(chunk.namespace)
            if previous is not None and previous != fingerprint:
                raise EmbeddingMismatchError(
                    f"同一批次中命名空间 '{chunk.namespace}' 出现不同 embedding "
                    f"fingerprint：{previous} / {fingerprint}"
                )
            batch[chunk.namespace] = fingerprint
        for namespace, fingerprint in batch.items():
            existing = self.fingerprint(namespace)
            if existing is not None and existing != fingerprint:
                raise EmbeddingMismatchError(
                    f"命名空间 '{namespace}' 已有 embedding fingerprint {existing}，"
                    f"与当前 {fingerprint} 不一致；请重建索引"
                )


def _load_metadata(raw: str) -> dict[str, Any]:
    """安全解析元数据 JSON，损坏时退化为空字典。"""
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}
