"""MemoryTool：把 MemoryManager 包装成可被 Agent 调用的工具。

P0 动作：add / search / stats / update / remove / clear_all。
所有动作返回紧凑文本，失败时返回“错误: ...”而不是抛异常，便于模型自我修正。
"""

from __future__ import annotations

import json
from typing import Any

from agentflow.memory.manager import MemoryManager
from agentflow.tools.base import Tool, ToolParameter

_ACTIONS = ("add", "search", "stats", "update", "remove", "clear_all")


class MemoryTool(Tool):
    """记忆读写工具。

    参数：
        manager: 已构建的 ``MemoryManager``。
        name: 工具名，默认 ``memory``。
    """

    def __init__(self, manager: MemoryManager, name: str = "memory") -> None:
        super().__init__(
            name=name,
            description=(
                "读写用户长期记忆。action=add 保存内容；action=search 按 query 召回；"
                "action=stats 查看统计；action=update 更新；action=remove 删除；"
                "action=clear_all 清空当前用户全部记忆。"
            ),
        )
        self.manager = manager

    def run(self, parameters: dict[str, Any]) -> str:
        """执行记忆动作。"""
        params = parameters or {}
        action = str(params.get("action", "")).strip().lower()
        if action not in _ACTIONS:
            return f"错误: 未知 action '{action}'，可用：{' / '.join(_ACTIONS)}"

        if action == "add":
            return self._add(params)
        if action == "search":
            return self._search(params)
        if action == "stats":
            return self._stats()
        if action == "update":
            return self._update(params)
        if action == "remove":
            return self._remove(params)
        return self._clear_all()

    def get_parameters(self) -> list[ToolParameter]:
        """工具参数自描述。"""
        return [
            ToolParameter("action", "string", "add / search / stats / update / remove / clear_all", True),
            ToolParameter("content", "string", "要保存或更新的记忆内容（add / update 用）"),
            ToolParameter("query", "string", "检索词（search 用）"),
            ToolParameter("memory_id", "string", "记忆 ID（update / remove 用）"),
            ToolParameter("importance", "number", "重要性 0~1，默认 0.5"),
            ToolParameter("memory_type", "string", "记忆类型，默认 semantic"),
            ToolParameter("metadata", "object", "附加元数据（对象或 JSON 字符串）"),
            ToolParameter("top_k", "integer", "返回条数，默认使用配置值"),
        ]

    def _add(self, params: dict[str, Any]) -> str:
        content = str(params.get("content", "")).strip()
        if not content:
            return "错误: add 需要 content"
        metadata, error = _as_metadata(params.get("metadata"))
        if error:
            return error
        try:
            importance = _as_float(params.get("importance"), 0.5)
        except ValueError:
            return "错误: importance 必须是 0~1 之间的数字"
        memory_type = str(params.get("memory_type", "semantic") or "semantic")
        try:
            memory_id = self.manager.add(
                content,
                importance=importance,
                memory_type=memory_type,
                metadata=metadata,
            )
        except Exception as exc:
            return f"错误: 保存记忆失败：{exc}"
        if not memory_id:
            return "错误: 记忆功能未开启"
        return f"已记住（id={memory_id}）"

    def _search(self, params: dict[str, Any]) -> str:
        query = str(params.get("query", "")).strip()
        if not query:
            return "错误: search 需要 query"
        try:
            limit = _as_int(params.get("top_k"), self.manager.top_k)
        except ValueError:
            return "错误: top_k 必须是正整数"
        if limit <= 0:
            return "错误: top_k 必须是正整数"
        try:
            entries = self.manager.search(query, top_k=limit)
        except Exception as exc:
            return f"错误: 检索记忆失败：{exc}"
        if not entries:
            return "（无相关记忆）"
        lines = []
        for entry in entries:
            lines.append(
                f"- [{entry.get('source')}] {entry.get('content')} "
                f"(score={float(entry.get('score', 0.0)):.3f}, id={entry.get('memory_id')})"
            )
        return "\n".join(lines)

    def _stats(self) -> str:
        try:
            stats = self.manager.stats()
        except Exception as exc:
            return f"错误: 读取记忆统计失败：{exc}"
        working = stats.get("working", {})
        long_term = stats.get("long_term", {})
        types = long_term.get("types", {}) if isinstance(long_term, dict) else {}
        type_text = ", ".join(f"{key}={value}" for key, value in sorted(types.items())) or "无"
        return (
            f"记忆统计：enabled={stats.get('enabled')} user={stats.get('user_id')} "
            f"working={working.get('items', 0)}/{working.get('max_items', 0)} "
            f"long_term={long_term.get('memories', 0)} "
            f"types=[{type_text}]"
        )

    def _update(self, params: dict[str, Any]) -> str:
        memory_id = str(params.get("memory_id", "")).strip()
        if not memory_id:
            return "错误: update 需要 memory_id"
        metadata, error = _as_metadata(params.get("metadata"))
        if error:
            return error

        updates: dict[str, Any] = {}
        if "content" in params and params.get("content") is not None:
            updates["content"] = str(params.get("content"))
        if params.get("importance") is not None:
            try:
                updates["importance"] = _as_float(params.get("importance"), 0.5)
            except ValueError:
                return "错误: importance 必须是 0~1 之间的数字"
        if metadata is not None:
            updates["metadata"] = metadata
        if params.get("memory_type") is not None:
            updates["memory_type"] = str(params.get("memory_type"))
        if not updates:
            return "错误: update 至少需要 content / importance / metadata / memory_type 之一"
        try:
            updated = self.manager.long_term.update(memory_id, **updates)
        except Exception as exc:
            return f"错误: 更新记忆失败：{exc}"
        return f"已更新（id={memory_id}）" if updated else f"错误: 未找到记忆 {memory_id}"

    def _remove(self, params: dict[str, Any]) -> str:
        memory_id = str(params.get("memory_id", "")).strip()
        if not memory_id:
            return "错误: remove 需要 memory_id"
        try:
            removed = self.manager.long_term.remove(memory_id)
        except Exception as exc:
            return f"错误: 删除记忆失败：{exc}"
        return f"已删除（id={memory_id}）" if removed else f"错误: 未找到记忆 {memory_id}"

    def _clear_all(self) -> str:
        try:
            self.manager.clear()
        except Exception as exc:
            return f"错误: 清空记忆失败：{exc}"
        return "已清空当前用户的全部记忆"


def _as_metadata(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    """把对象或 JSON 字符串解析为元数据字典。"""
    if value is None or value == "":
        return None, None
    if isinstance(value, dict):
        return dict(value), None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return None, "错误: metadata 必须是 JSON 对象"
        if isinstance(parsed, dict):
            return parsed, None
        return None, "错误: metadata 必须是 JSON 对象"
    return None, "错误: metadata 必须是对象或 JSON 字符串"


def _as_float(value: Any, default: float) -> float:
    """把参数转换为浮点数。"""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


def _as_int(value: Any, default: int) -> int:
    """把参数转换为整数。"""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
