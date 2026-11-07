"""LongTermMemory：SQLite 持久化长期记忆，按 user_id 隔离并做向量检索。

每条记忆保存正文、重要性、类型、元数据与 float32 向量。写入和检索时都会
校验 embedding fingerprint，避免换用不同 Embedder 后新旧向量混用。默认使用
纯标准库的 ``sqlite3``，路径缺省为 ``.data/memory.sqlite3``。
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, Sequence

from agentflow.core.exceptions import EmbeddingMismatchError, MemoryError
from agentflow.retrieval.store_sqlite import pack_vector, unpack_vector

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'semantic',
    content TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}',
    embedding BLOB NOT NULL,
    embedding_fingerprint TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_user_type ON memories(user_id, memory_type);
"""


class _Embedder(Protocol):
    """长期记忆只需要一个能把文本转成向量的实现。"""

    fingerprint: str

    def embed_query(self, text: str) -> list[float]:
        """把单条文本转成向量。"""
        ...


@dataclass
class MemoryHit:
    """一次长期记忆检索命中的记录。"""

    memory_id: str
    user_id: str
    content: str
    score: float = 0.0
    importance: float = 0.5
    memory_type: str = "semantic"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "content": self.content,
            "score": self.score,
            "importance": self.importance,
            "memory_type": self.memory_type,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "last_accessed_at": self.last_accessed_at.isoformat(timespec="seconds"),
            "access_count": self.access_count,
        }


class LongTermMemory:
    """基于 SQLite 的长期记忆存储。

    参数：
        embedder: 提供 ``embed_query`` 与 ``fingerprint`` 的向量化实现。
        path: SQLite 文件路径；``:memory:`` 表示进程内临时库。
        user_id: 记忆归属用户，所有读写都会按该字段隔离。

    向量在写入时做 L2 归一化，检索时用点积即余弦相似度。
    """

    def __init__(
        self,
        embedder: _Embedder,
        path: str | Path = ".data/memory.sqlite3",
        user_id: str = "default",
    ) -> None:
        if embedder is None or not hasattr(embedder, "embed_query"):
            raise MemoryError("LongTermMemory 需要提供 embed_query 的 embedder")
        if not str(user_id or "").strip():
            raise MemoryError("LongTermMemory 的 user_id 不能为空")
        self.embedder = embedder
        self.user_id = str(user_id)
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            self.path, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def add(
        self,
        content: str,
        *,
        memory_type: str = "semantic",
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> str:
        """写入一条记忆，返回 memory_id。"""
        text = str(content or "").strip()
        if not text:
            raise MemoryError("长期记忆内容不能为空")
        vector = self._embed(text)
        fingerprint = str(getattr(self.embedder, "fingerprint", ""))
        self._ensure_fingerprint(fingerprint)

        now = _now_iso()
        identifier = memory_id or uuid.uuid4().hex
        row = (
            identifier,
            self.user_id,
            str(memory_type or "semantic"),
            text,
            _clamp_importance(importance),
            now,
            now,
            0,
            json.dumps(metadata or {}, ensure_ascii=False, default=str),
            pack_vector(vector),
            fingerprint,
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memories (
                    memory_id, user_id, memory_type, content, importance,
                    created_at, last_accessed_at, access_count, metadata,
                    embedding, embedding_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            self._conn.commit()
        return identifier

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        memory_type: str | None = None,
        min_score: float | None = None,
    ) -> list[MemoryHit]:
        """按余弦相似度检索当前用户的记忆，命中记录会更新访问统计。"""
        if top_k <= 0:
            return []
        vector = self._embed(str(query or ""))
        fingerprint = str(getattr(self.embedder, "fingerprint", ""))
        self._ensure_fingerprint(fingerprint)

        sql = "SELECT * FROM memories WHERE user_id = ?"
        params: list[Any] = [self.user_id]
        if memory_type is not None:
            sql += " AND memory_type = ?"
            params.append(memory_type)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        hits: list[MemoryHit] = []
        for row in rows:
            stored = unpack_vector(row["embedding"])
            if len(stored) != len(vector):
                raise EmbeddingMismatchError(
                    f"查询向量维度 {len(vector)} 与已存记忆向量维度 {len(stored)} "
                    "不一致，请确认 Embedder 或重建记忆库"
                )
            score = _dot(vector, stored)
            if min_score is not None and score < min_score:
                continue
            hits.append(_row_to_hit(row, score))

        hits.sort(key=lambda hit: (-hit.score, hit.memory_id))
        selected = hits[:top_k]
        self._touch(selected)
        return selected

    def get(self, memory_id: str) -> MemoryHit | None:
        """按 ID 读取当前用户的一条记忆，不存在时返回 ``None``。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE memory_id = ? AND user_id = ?",
                (memory_id, self.user_id),
            ).fetchone()
        return _row_to_hit(row, 0.0) if row is not None else None

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
        memory_type: str | None = None,
    ) -> bool:
        """更新一条记忆；未提供的字段保持不变。记忆不存在时返回 ``False``。"""
        current = self.get(memory_id)
        if current is None:
            return False

        assignments: list[str] = []
        params: list[Any] = []
        if content is not None:
            text = str(content).strip()
            if not text:
                raise MemoryError("长期记忆内容不能为空")
            vector = self._embed(text)
            fingerprint = str(getattr(self.embedder, "fingerprint", ""))
            self._ensure_fingerprint(fingerprint)
            assignments.extend(["content = ?", "embedding = ?", "embedding_fingerprint = ?"])
            params.extend([text, pack_vector(vector), fingerprint])
        if importance is not None:
            assignments.append("importance = ?")
            params.append(_clamp_importance(importance))
        if metadata is not None:
            assignments.append("metadata = ?")
            params.append(json.dumps(metadata, ensure_ascii=False, default=str))
        if memory_type is not None:
            assignments.append("memory_type = ?")
            params.append(str(memory_type))
        if not assignments:
            return True

        assignments.append("last_accessed_at = ?")
        params.append(_now_iso())
        params.extend([memory_id, self.user_id])
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET " + ", ".join(assignments) + " WHERE memory_id = ? AND user_id = ?",
                params,
            )
            self._conn.commit()
        return True

    def remove(self, memory_id: str) -> bool:
        """删除一条记忆；不存在时返回 ``False``。"""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE memory_id = ? AND user_id = ?",
                (memory_id, self.user_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def clear(self) -> int:
        """清空当前用户的全部长期记忆，返回删除条数。"""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE user_id = ?", (self.user_id,)
            )
            self._conn.commit()
            return int(cursor.rowcount)

    def count(self) -> int:
        """统计当前用户的记忆条数。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE user_id = ?", (self.user_id,)
            ).fetchone()
        return int(row["n"])

    def stats(self) -> dict[str, Any]:
        """返回当前用户的记忆统计信息。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT memory_type, COUNT(*) AS n FROM memories "
                "WHERE user_id = ? GROUP BY memory_type",
                (self.user_id,),
            ).fetchall()
            dimension_row = self._conn.execute(
                "SELECT embedding FROM memories WHERE user_id = ? AND embedding_fingerprint <> '' LIMIT 1",
                (self.user_id,),
            ).fetchone()
        types = {row["memory_type"]: int(row["n"]) for row in rows}
        dimension = len(unpack_vector(dimension_row["embedding"])) if dimension_row else 0
        return {
            "user_id": self.user_id,
            "path": self.path,
            "memories": sum(types.values()),
            "types": types,
            "dimension": dimension,
            "fingerprint": self._existing_fingerprint(),
        }

    def close(self) -> None:
        """关闭数据库连接（可重复调用）。"""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> "LongTermMemory":
        """支持 ``with`` 上下文。"""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """退出上下文时关闭连接。"""
        self.close()

    def _embed(self, text: str) -> list[float]:
        """向量化并做 L2 归一化，空向量视为错误。"""
        vector = [float(value) for value in self.embedder.embed_query(text)]
        if not vector:
            raise MemoryError("Embedder 返回了空向量，无法写入/检索长期记忆")
        return _normalize(vector)

    def _existing_fingerprint(self) -> str:
        """返回当前用户已记录的 embedding fingerprint。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT embedding_fingerprint FROM memories "
                "WHERE user_id = ? AND embedding_fingerprint <> '' LIMIT 1",
                (self.user_id,),
            ).fetchone()
        return str(row["embedding_fingerprint"]) if row is not None else ""

    def _ensure_fingerprint(self, fingerprint: str) -> None:
        """校验当前 Embedder 与已存记忆使用的 fingerprint 一致。"""
        existing = self._existing_fingerprint()
        if existing and fingerprint and existing != fingerprint:
            raise EmbeddingMismatchError(
                f"用户 '{self.user_id}' 的长期记忆使用 embedding fingerprint "
                f"{existing}，与当前 {fingerprint} 不一致；请重建记忆库"
            )

    def _touch(self, hits: Sequence[MemoryHit]) -> None:
        """更新命中记录的访问时间与次数。"""
        if not hits:
            return
        now = datetime.now()
        stamp = now.isoformat(timespec="seconds")
        with self._lock:
            self._conn.executemany(
                "UPDATE memories SET last_accessed_at = ?, access_count = access_count + 1 "
                "WHERE memory_id = ? AND user_id = ?",
                [(stamp, hit.memory_id, self.user_id) for hit in hits],
            )
            self._conn.commit()
        for hit in hits:
            hit.last_accessed_at = now
            hit.access_count += 1


def _row_to_hit(row: sqlite3.Row, score: float) -> MemoryHit:
    """把数据库行转换为 MemoryHit。"""
    return MemoryHit(
        memory_id=row["memory_id"],
        user_id=row["user_id"],
        content=row["content"],
        score=float(score),
        importance=float(row["importance"]),
        memory_type=row["memory_type"],
        metadata=_load_metadata(row["metadata"]),
        created_at=_parse_time(row["created_at"]),
        last_accessed_at=_parse_time(row["last_accessed_at"]),
        access_count=int(row["access_count"]),
    )


def _load_metadata(raw: str) -> dict[str, Any]:
    """安全解析元数据 JSON，损坏时退化为空字典。"""
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _parse_time(raw: str) -> datetime:
    """解析 ISO 时间字符串，失败时退化为当前时间。"""
    try:
        return datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return datetime.now()


def _now_iso() -> str:
    """当前时间的 ISO 字符串（秒精度）。"""
    return datetime.now().isoformat(timespec="seconds")


def _clamp_importance(value: float) -> float:
    """把重要性限制在 [0, 1]。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.5
    return min(1.0, max(0.0, number))


def _normalize(vector: Sequence[float]) -> list[float]:
    """L2 归一化；全零向量保持原样。"""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return [float(value) for value in vector]
    return [float(value) / norm for value in vector]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """纯 Python 点积（归一化后即余弦相似度）。"""
    return float(sum(a * b for a, b in zip(left, right)))
