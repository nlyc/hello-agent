"""笔记数据模型：五类型、优先级与 Markdown 正文。

NoteTool 管「项目式任务记忆」：结构化、长期追踪、人类可读可手动干预；
与 MemoryTool 的「对话式记忆」互补。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agentflow.core.exceptions import NoteError

NOTE_TYPES: tuple[str, ...] = (
    "task_state",
    "conclusion",
    "blocker",
    "action",
    "reference",
)

# 类型优先级直接作为 ContextBuilder 的相关性：blocker 必须最先被看到。
NOTE_PRIORITY: dict[str, float] = {
    "blocker": 0.9,
    "action": 0.8,
    "task_state": 0.75,
    "conclusion": 0.7,
    "reference": 0.6,
}

NOTE_TYPE_LABELS: dict[str, str] = {
    "task_state": "阶段进展",
    "conclusion": "结论",
    "blocker": "阻塞",
    "action": "下一步",
    "reference": "参考",
}


def utc_now() -> datetime:
    """返回当前 UTC 时间（带时区）。"""
    return datetime.now(timezone.utc)


def ensure_aware(value: datetime | None) -> datetime:
    """把可能为 naive 的时间统一为 UTC aware。"""
    if value is None:
        return utc_now()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass
class Note:
    """一条结构化笔记。

    ``id`` 与文件名一致；正文为 Markdown，元数据写入 YAML front matter。
    """

    id: str
    title: str = ""
    type: str = "task_state"
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id or "").strip()
        self.title = str(self.title or "").strip()
        self.type = str(self.type or "task_state").strip().lower()
        if self.type not in NOTE_TYPES:
            raise NoteError(
                f"未知笔记类型 '{self.type}'，可用：{' / '.join(NOTE_TYPES)}"
            )
        self.tags = [
            str(tag).strip() for tag in (self.tags or []) if str(tag).strip()
        ]
        self.content = str(self.content or "").strip()
        self.created_at = ensure_aware(self.created_at)
        self.updated_at = ensure_aware(self.updated_at)
        if self.updated_at < self.created_at:
            self.updated_at = self.created_at

    @property
    def priority(self) -> float:
        """类型优先级，供 ContextBuilder 直接使用。"""
        return NOTE_PRIORITY[self.type]

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""
        return {
            "id": self.id,
            "title": self.title,
            "type": self.type,
            "tags": list(self.tags),
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "updated_at": self.updated_at.isoformat(timespec="seconds"),
            "content": self.content,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Note":
        """从字典恢复；时间字段支持 ISO 字符串。"""
        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            type=str(data.get("type", "task_state")),
            tags=list(data.get("tags", []) or []),
            created_at=_parse_time(data.get("created_at")),
            updated_at=_parse_time(data.get("updated_at")),
            content=str(data.get("content", "")),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def _parse_time(value: Any) -> datetime:
    """解析 ISO 时间字符串；失败时返回当前时间。"""
    if isinstance(value, datetime):
        return ensure_aware(value)
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return ensure_aware(datetime.fromisoformat(text))
        except ValueError:
            pass
    return utc_now()
