"""MemoryManager：工作记忆与长期记忆的统一门面。

对上层 Agent 暴露 ``add / search / build_context / record_turn`` 等一致接口。
``build_context`` 会合并两路召回结果、按分数排序并裁剪到 Token 预算；
任一子步骤失败时只丢弃该路结果，不阻断主流程。
"""

from __future__ import annotations

from typing import Any

from agentflow.core.config import MemoryConfig
from agentflow.core.exceptions import MemoryError
from agentflow.memory.context import MemoryContext
from agentflow.memory.long_term import LongTermMemory, MemoryHit
from agentflow.memory.message_history import estimate_tokens
from agentflow.memory.working import WorkingMemory, WorkingMemoryItem

_HEADER = "【记忆上下文】"


class MemoryManager:
    """工作记忆 + 长期记忆的统一管理器。

    参数：
        embedder: 向量化实现，需提供 ``embed_query`` 与 ``fingerprint``。
        long_term_store: 已构建的长期记忆存储（其 ``user_id`` 必须与管理器一致）。
        user_id: 当前用户 ID，用于隔离与统计。
        working: 可选的工作记忆实例；为空时按 ``config`` 构建。
        config: 记忆配置；为空时使用默认 ``MemoryConfig()``。
    """

    def __init__(
        self,
        embedder: Any,
        long_term_store: LongTermMemory,
        user_id: str = "default",
        working: WorkingMemory | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        if long_term_store is None:
            raise MemoryError("MemoryManager 需要 long_term_store")
        store_user = str(getattr(long_term_store, "user_id", "default"))
        if user_id != store_user:
            raise MemoryError(
                f"MemoryManager 的 user_id '{user_id}' 与长期记忆存储的 "
                f"user_id '{store_user}' 不一致"
            )
        self.embedder = embedder
        self.long_term = long_term_store
        self.user_id = user_id
        self.config = config or MemoryConfig()
        self.top_k = int(self.config.top_k)
        self.working = (
            working
            if working is not None
            else WorkingMemory(
                max_items=self.config.working_max_items,
                ttl_minutes=self.config.working_ttl_minutes,
            )
        )

    @property
    def enabled(self) -> bool:
        """记忆功能是否开启。"""
        return bool(self.config.enabled)

    def add(
        self,
        content: str,
        *,
        importance: float = 0.5,
        memory_type: str = "semantic",
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> str:
        """同时写入长期记忆与工作记忆，返回长期记忆 ID。"""
        if not self.enabled:
            return ""
        identifier = self.long_term.add(
            content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata,
            memory_id=memory_id,
        )
        working_metadata = dict(metadata or {})
        working_metadata["memory_id"] = identifier
        working_metadata["memory_type"] = memory_type
        self.working.add(content, importance=importance, metadata=working_metadata)
        return identifier

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """合并两路召回，返回按分数排序的去重条目。"""
        if not self.enabled:
            return []
        limit = int(top_k or self.top_k)
        entries: list[dict[str, Any]] = []

        try:
            for hit in self.long_term.search(query, top_k=limit):
                entries.append(_hit_to_entry(hit))
        except Exception:
            pass
        try:
            for item in self.working.search(query, top_k=limit):
                entries.append(_item_to_entry(item))
        except Exception:
            pass

        return _dedupe_and_sort(entries)[:limit]

    def build_context(self, query: str = "", max_tokens: int = 1500) -> MemoryContext:
        """构建记忆上下文；任一子步骤失败时对应结果为空，不抛异常。"""
        if not self.enabled:
            return MemoryContext(enabled=False)
        if max_tokens <= 0:
            return MemoryContext(enabled=True)

        working_items: list[WorkingMemoryItem] = []
        long_term_hits: list[MemoryHit] = []
        entries: list[dict[str, Any]] = []

        try:
            working_items = self.working.search(query, top_k=self.top_k)
            entries.extend(_item_to_entry(item) for item in working_items)
        except Exception:
            working_items = []
        try:
            long_term_hits = self.long_term.search(query, top_k=self.top_k)
            entries.extend(_hit_to_entry(hit) for hit in long_term_hits)
        except Exception:
            long_term_hits = []

        text, kept = _compose(entries, max_tokens)
        return MemoryContext(
            text=text,
            items=kept,
            working_items=working_items,
            long_term_hits=long_term_hits,
            enabled=True,
        )

    def record_turn(
        self,
        user_input: str,
        assistant_output: str,
        *,
        importance: float = 0.5,
    ) -> str:
        """记录一轮对话：写入工作记忆，并以 episodic 类型写入长期记忆。"""
        question = str(user_input or "").strip()
        if not question:
            return ""
        answer = str(assistant_output or "").strip()
        content = f"用户：{question}\n助手：{answer}"
        return self.add(
            content,
            importance=importance,
            memory_type="episodic",
            metadata={"kind": "turn"},
        )

    def stats(self) -> dict[str, Any]:
        """返回工作记忆与长期记忆的统计信息。"""
        try:
            working_stats: dict[str, Any] = self.working.stats()
        except Exception as exc:  # pragma: no cover - 防御性降级
            working_stats = {"error": str(exc)}
        try:
            long_term_stats: dict[str, Any] = self.long_term.stats()
        except Exception as exc:  # pragma: no cover - 防御性降级
            long_term_stats = {"error": str(exc)}
        return {
            "enabled": self.enabled,
            "user_id": self.user_id,
            "working": working_stats,
            "long_term": long_term_stats,
        }

    def clear(self) -> None:
        """清空当前用户的工作记忆与长期记忆。"""
        self.working.clear()
        self.long_term.clear()


def _hit_to_entry(hit: MemoryHit) -> dict[str, Any]:
    """把长期记忆命中转换为统一条目。"""
    return {
        "memory_id": hit.memory_id,
        "content": hit.content,
        "score": float(hit.score),
        "importance": float(hit.importance),
        "source": "long_term",
        "memory_type": hit.memory_type,
        "metadata": dict(hit.metadata),
    }


def _item_to_entry(item: WorkingMemoryItem) -> dict[str, Any]:
    """把工作记忆条目转换为统一条目。"""
    return {
        "memory_id": str(item.metadata.get("memory_id", item.item_id)),
        "content": item.content,
        "score": float(item.score),
        "importance": float(item.importance),
        "source": "working",
        "memory_type": str(item.metadata.get("memory_type", "working")),
        "metadata": dict(item.metadata),
    }


def _dedupe_and_sort(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按内容去重（保留分数更高者）并按分数降序排序。"""
    best: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = " ".join(str(entry.get("content", "")).split())
        current = best.get(key)
        if current is None or entry.get("score", 0.0) > current.get("score", 0.0):
            best[key] = entry
    return sorted(
        best.values(),
        key=lambda item: (-float(item.get("score", 0.0)), str(item.get("content", ""))),
    )


def _compose(
    entries: list[dict[str, Any]], max_tokens: int
) -> tuple[str, list[dict[str, Any]]]:
    """把条目拼成带标题与来源标签的文本，并裁剪到 Token 预算。"""
    if max_tokens <= 0 or not entries:
        return "", []
    used = estimate_tokens(_HEADER)
    if used > max_tokens:
        return "", []

    lines = [_HEADER]
    kept: list[dict[str, Any]] = []
    for entry in _dedupe_and_sort(entries):
        label = "工作记忆" if entry.get("source") == "working" else "长期记忆"
        line = f"- [{label}] {entry.get('content', '')}"
        cost = estimate_tokens(line)
        if used + cost > max_tokens:
            continue
        lines.append(line)
        used += cost
        kept.append(entry)
    if not kept:
        return "", []
    return "\n".join(lines), kept
