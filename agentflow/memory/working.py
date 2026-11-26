"""WorkingMemory：进程内有界工作记忆，带 TTL、重要性加权与 Token 预算。

工作记忆只保留最近一段时间内、当前任务相关的少量信息，不跨进程持久化。
评分公式遵循第 8 章约定：

    score = (relevance × time_decay) × importance_weight
    importance_weight = 0.8 + importance × 0.4

其中 relevance 用查询词与条目的词元覆盖率近似，time_decay 按条目年龄
相对 TTL 指数衰减。过期条目在每次读写前自动清理。
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agentflow.core.exceptions import MemoryError
from agentflow.memory.message_history import estimate_tokens

_CJK_RUN_RE = re.compile(r"[一-鿿]+")
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def _token_set(text: str) -> set[str]:
    """轻量分词：CJK bigram + 英文/数字小写词，用于相关性近似。"""
    tokens: set[str] = set()
    for match in _CJK_RUN_RE.finditer(text or ""):
        run = match.group(0)
        if len(run) == 1:
            tokens.add(run)
        else:
            tokens.update(run[i : i + 2] for i in range(len(run) - 1))
    tokens.update(word.lower() for word in _WORD_RE.findall(text or ""))
    return tokens


@dataclass
class WorkingMemoryItem:
    """工作记忆中的一条记录。"""

    content: str
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    item_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""
        return {
            "item_id": self.item_id,
            "content": self.content,
            "importance": self.importance,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "last_accessed_at": self.last_accessed_at.isoformat(timespec="seconds"),
            "access_count": self.access_count,
            "score": self.score,
        }


class WorkingMemory:
    """进程内工作记忆。

    参数：
        max_items: 最大条目数，超出时淘汰综合得分最低的条目。
        ttl_minutes: 条目存活时间（分钟）；为 0 表示写入后立即过期。
    """

    def __init__(self, max_items: int = 50, ttl_minutes: float = 60) -> None:
        if max_items <= 0:
            raise MemoryError("WorkingMemory 的 max_items 必须为正整数")
        if ttl_minutes < 0:
            raise MemoryError("WorkingMemory 的 ttl_minutes 不能为负数")
        self.max_items = int(max_items)
        self.ttl_minutes = float(ttl_minutes)
        self._items: list[WorkingMemoryItem] = []

    @property
    def ttl_seconds(self) -> float:
        """TTL 对应的秒数。"""
        return self.ttl_minutes * 60.0

    def add(
        self,
        content: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """写入一条工作记忆，返回条目 ID。"""
        text = str(content or "").strip()
        if not text:
            raise MemoryError("工作记忆内容不能为空")
        item = WorkingMemoryItem(
            content=text,
            importance=_clamp_importance(importance),
            metadata=dict(metadata or {}),
        )
        self._items.append(item)
        self._purge_expired()
        self._trim_to_capacity()
        return item.item_id

    def get_context(
        self,
        query: str = "",
        top_k: int = 5,
        max_tokens: int = 1500,
    ) -> str:
        """按查询召回工作记忆，并裁剪到 Token 预算，返回可直接注入提示词的文本。"""
        if max_tokens <= 0:
            return ""
        selected = self.search(query, top_k=top_k)
        if not selected:
            return ""

        now = datetime.now()
        lines: list[str] = []
        used = 0
        for item in selected:
            line = f"- {item.content}"
            cost = estimate_tokens(line)
            if used + cost > max_tokens:
                continue
            lines.append(line)
            used += cost
            item.last_accessed_at = now
            item.access_count += 1
        return "\n".join(lines)

    def search(self, query: str = "", top_k: int = 5) -> list[WorkingMemoryItem]:
        """按综合得分召回工作记忆条目（返回对象引用，访问统计由调用方更新）。"""
        if top_k <= 0:
            return []
        self._purge_expired()
        query_tokens = _token_set(query)
        now = datetime.now()
        scored: list[WorkingMemoryItem] = []
        for item in self._items:
            item.score = self._score(item, query_tokens, now)
            scored.append(item)
        scored.sort(key=lambda entry: (-entry.score, entry.created_at, entry.item_id))
        return scored[:top_k]

    def all_items(self) -> list[WorkingMemoryItem]:
        """返回当前未过期的全部条目（按写入顺序）。"""
        self._purge_expired()
        return list(self._items)

    def remove(self, item_id: str) -> bool:
        """删除指定条目；不存在或已过期时返回 False。"""
        self._purge_expired()
        target = str(item_id or "")
        if not target:
            return False
        before = len(self._items)
        self._items = [item for item in self._items if item.item_id != target]
        return len(self._items) != before

    def clear(self) -> None:
        """清空工作记忆。"""
        self._items.clear()

    def stats(self) -> dict[str, Any]:
        """返回工作记忆统计信息。"""
        self._purge_expired()
        return {
            "items": len(self._items),
            "max_items": self.max_items,
            "ttl_minutes": self.ttl_minutes,
        }

    def __len__(self) -> int:
        self._purge_expired()
        return len(self._items)

    def _score(
        self,
        item: WorkingMemoryItem,
        query_tokens: set[str],
        now: datetime,
    ) -> float:
        """计算单条记忆的综合得分。"""
        relevance = _relevance(item.content, query_tokens)
        decay = _time_decay(item.created_at, now, self.ttl_seconds)
        importance_weight = 0.8 + _clamp_importance(item.importance) * 0.4
        return relevance * decay * importance_weight

    def _purge_expired(self, now: datetime | None = None) -> None:
        """丢弃已超过 TTL 的条目。"""
        current = now or datetime.now()
        self._items = [
            item
            for item in self._items
            if _age_seconds(item.created_at, current) < self.ttl_seconds
        ]

    def _trim_to_capacity(self) -> None:
        """条目数超限时淘汰综合得分最低的条目。"""
        if len(self._items) <= self.max_items:
            return
        now = datetime.now()
        no_query: set[str] = set()
        ranked = sorted(
            self._items,
            key=lambda item: (
                self._score(item, no_query, now),
                item.created_at,
                item.item_id,
            ),
        )
        drop = len(self._items) - self.max_items
        removed = {id(item) for item in ranked[:drop]}
        self._items = [item for item in self._items if id(item) not in removed]


def _clamp_importance(value: float) -> float:
    """把重要性限制在 [0, 1]。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.5
    return min(1.0, max(0.0, number))


def _relevance(content: str, query_tokens: set[str]) -> float:
    """查询词元在条目中的覆盖率；无查询词时视为完全相关。"""
    if not query_tokens:
        return 1.0
    content_tokens = _token_set(content)
    if not content_tokens:
        return 0.0
    return len(query_tokens & content_tokens) / len(query_tokens)


def _age_seconds(created_at: datetime, now: datetime) -> float:
    """条目年龄（秒），负数按 0 处理。"""
    return max(0.0, (now - created_at).total_seconds())


def _time_decay(created_at: datetime, now: datetime, ttl_seconds: float) -> float:
    """指数时间衰减：年龄为 0 时为 1，年龄等于 TTL 时约为 1/e。"""
    if ttl_seconds <= 0:
        return 0.0
    return math.exp(-_age_seconds(created_at, now) / ttl_seconds)
