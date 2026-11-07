"""MemoryContext：一次记忆构建的结果载体，供三种 Agent 范式统一注入提示词。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _to_dict(item: Any) -> Any:
    """把条目对象或字典统一转换为可 JSON 序列化的字典。"""
    if hasattr(item, "to_dict"):
        return item.to_dict()
    if isinstance(item, dict):
        return dict(item)
    return item


@dataclass
class MemoryContext:
    """记忆上下文的统一结构。

    字段：
        text: 可直接拼进系统提示词的记忆文本；无记忆时为空串。
        items: 工作记忆与长期记忆合并、排序后的条目。
        working_items: 本次命中的工作记忆条目。
        long_term_hits: 本次命中的长期记忆条目。
        enabled: 记忆开关；``False`` 表示本次未构建记忆上下文。
    """

    text: str = ""
    items: list[Any] = field(default_factory=list)
    working_items: list[Any] = field(default_factory=list)
    long_term_hits: list[Any] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""
        return {
            "text": self.text,
            "items": [_to_dict(item) for item in self.items],
            "working_items": [_to_dict(item) for item in self.working_items],
            "long_term_hits": [_to_dict(item) for item in self.long_term_hits],
            "enabled": self.enabled,
        }

    def __bool__(self) -> bool:
        """有可用文本时视为真，便于调用方 ``if context:`` 判断。"""
        return bool(self.text)
