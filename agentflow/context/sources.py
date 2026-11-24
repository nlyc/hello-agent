"""上下文来源适配器：把 Memory / RAG / 历史 / 静态信息转换为 ContextPacket。

每个 Source 独立失败；Builder 负责捕获异常并记录 ``source_errors``，因此
本模块只做转换，不做跨来源容错。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, Sequence

from agentflow.context.types import ContextPacket, clamp01, ensure_aware, utc_now


class ContextSource(Protocol):
    """上下文来源协议。"""

    name: str

    def gather(
        self, query: str, *, limit: int, now: datetime
    ) -> list[ContextPacket]:
        """按 query 汇集候选信息包。"""
        ...


def parse_timestamp(value: Any, default: datetime | None = None) -> datetime:
    """把 datetime / ISO 字符串 / 时间戳转换为 UTC aware；失败时用默认值。"""
    fallback = ensure_aware(default or utc_now())
    if value is None:
        return fallback
    if isinstance(value, datetime):
        return ensure_aware(value)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=fallback.tzinfo)
        except (OverflowError, OSError, ValueError):
            return fallback
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return fallback
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return ensure_aware(datetime.fromisoformat(text))
        except ValueError:
            return fallback
    return fallback


def _entry_value(entry: Any, key: str, default: Any = None) -> Any:
    """同时兼容 dict 与对象形式的记忆条目。"""
    if isinstance(entry, dict):
        return entry.get(key, default)
    return getattr(entry, key, default)


def history_packets(
    history: Any,
    *,
    limit: int,
    now: datetime | None = None,
) -> list[ContextPacket]:
    """把最近 ``limit`` 条对话历史转换为 ContextPacket。

    ``history`` 可以是 ``MessageHistory``、Message 列表或 dict 列表。
    """
    if history is None or limit <= 0:
        return []
    getter = getattr(history, "get", None)
    if callable(getter):
        messages = list(getter())
    else:
        messages = list(history)
    current = ensure_aware(now or utc_now())
    packets: list[ContextPacket] = []
    for message in messages[-limit:]:
        if isinstance(message, dict):
            content = str(message.get("content", "") or "").strip()
            role = str(message.get("role", "user") or "user")
            timestamp = message.get("timestamp", current)
        else:
            content = str(getattr(message, "content", "") or "").strip()
            role = str(getattr(message, "role", "user") or "user")
            timestamp = getattr(message, "timestamp", current)
        if not content:
            continue
        packets.append(
            ContextPacket(
                content=f"[{role}] {content}",
                timestamp=parse_timestamp(timestamp, current),
                relevance_score=0.6,
                source="history",
                section="context",
                metadata={"role": role},
                trusted=True,
            )
        )
    return packets


class HistoryContextSource:
    """把固定历史列表包装成 Source，便于独立测试或复用。"""

    name = "history"

    def __init__(self, history: Any) -> None:
        self.history = history

    def gather(
        self, query: str, *, limit: int, now: datetime
    ) -> list[ContextPacket]:
        return history_packets(self.history, limit=limit, now=now)


class StaticContextSource:
    """静态信息源：系统指令、领域背景等。"""

    def __init__(self, packets: Sequence[ContextPacket], *, name: str = "static") -> None:
        self.name = name
        self._packets = list(packets)

    def gather(
        self, query: str, *, limit: int, now: datetime
    ) -> list[ContextPacket]:
        if limit <= 0:
            return []
        return self._packets[:limit]


class MemoryContextSource:
    """记忆来源：优先用 ``MemoryManager.search``，旧 ``recall`` 也可兼容。"""

    def __init__(self, manager: Any, *, section: str = "context") -> None:
        self.manager = manager
        self.name = "memory"
        self.section = section

    def gather(
        self, query: str, *, limit: int, now: datetime
    ) -> list[ContextPacket]:
        if limit <= 0 or self.manager is None:
            return []
        if not bool(getattr(self.manager, "enabled", True)):
            return []

        search = getattr(self.manager, "search", None)
        if callable(search):
            entries = list(search(query, top_k=limit) or [])
            packets: list[ContextPacket] = []
            for entry in entries:
                packet = self._entry_to_packet(entry, now)
                if packet is not None:
                    packets.append(packet)
            return packets

        recall = getattr(self.manager, "recall", None)
        if not callable(recall):
            return []
        text = str(recall(query) or "").strip()
        if not text:
            return []
        return [
            ContextPacket(
                content=text,
                timestamp=ensure_aware(now),
                relevance_score=0.6,
                source="memory",
                section=self.section,
                metadata={"kind": "recall"},
                trusted=True,
            )
        ]

    def _entry_to_packet(
        self, entry: Any, now: datetime
    ) -> ContextPacket | None:
        content = str(_entry_value(entry, "content", "") or "").strip()
        if not content:
            return None
        metadata = dict(_entry_value(entry, "metadata", {}) or {})
        score = _entry_value(entry, "score", 0.0)
        try:
            relevance = clamp01(float(score or 0.0))
        except (TypeError, ValueError):
            relevance = 0.0
        timestamp = parse_timestamp(
            metadata.get("created_at") or metadata.get("timestamp"), now
        )
        metadata.setdefault("memory_source", _entry_value(entry, "source", ""))
        metadata.setdefault("memory_type", _entry_value(entry, "memory_type", ""))
        return ContextPacket(
            content=content,
            timestamp=timestamp,
            relevance_score=relevance,
            source="memory",
            section=self.section,
            metadata=metadata,
            trusted=True,
        )


class RetrievalContextSource:
    """RAG 来源：把检索命中转换为 Evidence 信息包，并按排名给相关性。"""

    def __init__(
        self,
        pipeline: Any,
        *,
        section: str = "evidence",
        prefix_citation: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.name = "rag"
        self.section = section
        self.prefix_citation = prefix_citation

    def gather(
        self, query: str, *, limit: int, now: datetime
    ) -> list[ContextPacket]:
        if limit <= 0 or self.pipeline is None:
            return []
        search = getattr(self.pipeline, "search", None)
        if not callable(search):
            return []
        hits = list(search(query, top_k=limit) or [])
        total = len(hits)
        packets: list[ContextPacket] = []
        for rank, hit in enumerate(hits):
            text = str(getattr(hit, "text", "") or "").strip()
            if not text:
                continue
            citation = str(getattr(hit, "citation", "") or "").strip()
            content = f"【{citation}】\n{text}" if (self.prefix_citation and citation) else text
            relevance = 1.0 - (rank / max(1, total))
            packets.append(
                ContextPacket(
                    content=content,
                    timestamp=ensure_aware(now),
                    relevance_score=relevance,
                    source="rag",
                    section=self.section,
                    metadata={
                        "citation": citation,
                        "doc_id": str(getattr(hit, "doc_id", "") or ""),
                        "chunk_id": str(getattr(hit, "chunk_id", "") or ""),
                        "source": str(getattr(hit, "source", "") or ""),
                        "raw_score": float(getattr(hit, "score", 0.0) or 0.0),
                    },
                    trusted=True,
                )
            )
        return packets
