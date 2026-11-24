"""NoteTool：把 NoteStore 包装成 Agent 可调用的工具。

七操作：create / read / update / search / list / summary / delete。
所有动作返回紧凑文本，失败时返回「错误: ...」而不是抛异常，便于模型自我修正。
"""

from __future__ import annotations

import json
from typing import Any

from agentflow.core.exceptions import NoteError
from agentflow.notes.store import NoteStore
from agentflow.notes.types import NOTE_TYPES
from agentflow.tools.base import Tool, ToolParameter

_ACTIONS = ("create", "read", "update", "search", "list", "summary", "delete")


class NoteTool(Tool):
    """项目式笔记读写工具。"""

    def __init__(self, store: NoteStore, name: str = "note") -> None:
        super().__init__(
            name=name,
            description=(
                "读写项目式笔记（Markdown 持久化）。action=create 新建；"
                "action=read 读取；action=update 更新；action=search 关键词检索；"
                "action=list 按类型列出；action=summary 查看统计与阻塞；"
                "action=delete 删除。类型：task_state / conclusion / blocker / action / reference。"
            ),
        )
        self.store = store

    def run(self, parameters: dict[str, Any]) -> str:
        """执行笔记动作。"""
        params = parameters or {}
        action = str(params.get("action", "")).strip().lower()
        if action not in _ACTIONS:
            return f"错误: 未知 action '{action}'，可用：{' / '.join(_ACTIONS)}"
        try:
            if action == "create":
                return self._create(params)
            if action == "read":
                return self._read(params)
            if action == "update":
                return self._update(params)
            if action == "search":
                return self._search(params)
            if action == "list":
                return self._list(params)
            if action == "summary":
                return self._summary()
            return self._delete(params)
        except NoteError as exc:
            return f"错误: {exc}"
        except Exception as exc:  # 防御性：工具不拖垮 Agent
            return f"错误: 笔记操作失败：{exc}"

    def get_parameters(self) -> list[ToolParameter]:
        """工具参数自描述。"""
        return [
            ToolParameter(
                "action", "string", " / ".join(_ACTIONS), True
            ),
            ToolParameter("note_id", "string", "笔记 ID（read / update / delete 用）"),
            ToolParameter("title", "string", "标题（create 用）"),
            ToolParameter("content", "string", "Markdown 正文（create / update 用）"),
            ToolParameter("type", "string", "笔记类型，默认 task_state"),
            ToolParameter("tags", "array", "标签列表（数组或逗号分隔字符串）"),
            ToolParameter("query", "string", "检索词（search 用）"),
            ToolParameter("top_k", "integer", "返回条数，默认 10"),
            ToolParameter("metadata", "object", "附加元数据（对象或 JSON 字符串）"),
        ]

    def _create(self, params: dict[str, Any]) -> str:
        title = str(params.get("title", "")).strip()
        if not title:
            return "错误: create 需要 title"
        note_type = str(params.get("type", "task_state") or "task_state").strip().lower()
        if note_type not in NOTE_TYPES:
            return f"错误: 未知笔记类型 '{note_type}'，可用：{' / '.join(NOTE_TYPES)}"
        metadata, error = _as_metadata(params.get("metadata"))
        if error:
            return error
        note = self.store.create(
            title,
            str(params.get("content", "") or ""),
            type=note_type,
            tags=_as_tags(params.get("tags")),
            note_id=str(params.get("note_id", "") or "").strip() or None,
            metadata=metadata,
        )
        return f"已创建笔记（id={note.id}, type={note.type}）"

    def _read(self, params: dict[str, Any]) -> str:
        note_id = str(params.get("note_id", "")).strip()
        if not note_id:
            return "错误: read 需要 note_id"
        note = self.store.read(note_id)
        tags = ", ".join(note.tags) if note.tags else "无"
        return (
            f"# {note.title}\n"
            f"- id: {note.id}\n"
            f"- type: {note.type}\n"
            f"- tags: {tags}\n"
            f"- updated_at: {note.updated_at.isoformat(timespec='seconds')}\n\n"
            f"{note.content or '（空）'}"
        )

    def _update(self, params: dict[str, Any]) -> str:
        note_id = str(params.get("note_id", "")).strip()
        if not note_id:
            return "错误: update 需要 note_id"
        updates: dict[str, Any] = {}
        if params.get("title") is not None:
            updates["title"] = str(params.get("title"))
        if params.get("content") is not None:
            updates["content"] = str(params.get("content"))
        if params.get("type") is not None:
            updates["type"] = str(params.get("type"))
        if params.get("tags") is not None:
            updates["tags"] = _as_tags(params.get("tags"))
        if params.get("metadata") is not None:
            metadata, error = _as_metadata(params.get("metadata"))
            if error:
                return error
            updates["metadata"] = metadata
        if not updates:
            return "错误: update 至少需要 title / content / type / tags / metadata 之一"
        note = self.store.update(note_id, **updates)
        return f"已更新笔记（id={note.id}, type={note.type}）"

    def _search(self, params: dict[str, Any]) -> str:
        query = str(params.get("query", "")).strip()
        if not query:
            return "错误: search 需要 query"
        try:
            limit = _as_int(params.get("top_k"), 10)
        except ValueError:
            return "错误: top_k 必须是整数"
        notes = self.store.search(
            query,
            note_type=_optional_text(params.get("type")),
            tags=_as_tags(params.get("tags")),
            limit=limit,
        )
        return _format_notes(notes, empty="（无相关笔记）")

    def _list(self, params: dict[str, Any]) -> str:
        try:
            limit = _as_int(params.get("top_k"), 20)
        except ValueError:
            return "错误: top_k 必须是整数"
        notes = self.store.list(
            note_type=_optional_text(params.get("type")),
            tags=_as_tags(params.get("tags")),
            limit=limit,
        )
        return _format_notes(notes, empty="（暂无笔记）")

    def _summary(self) -> str:
        summary = self.store.summary()
        types = summary.get("types", {})
        type_text = ", ".join(f"{key}={value}" for key, value in types.items()) or "无"
        recent = summary.get("recent", [])
        lines = [
            f"笔记统计：total={summary.get('total', 0)} "
            f"blockers={summary.get('unresolved_blockers', 0)} "
            f"types=[{type_text}] "
            f"index_consistent={summary.get('index_consistent')}",
        ]
        for item in recent[:3]:
            lines.append(f"- [{item.get('type')}] {item.get('title')} (id={item.get('id')})")
        return "\n".join(lines)

    def _delete(self, params: dict[str, Any]) -> str:
        note_id = str(params.get("note_id", "")).strip()
        if not note_id:
            return "错误: delete 需要 note_id"
        removed = self.store.delete(note_id)
        return f"已删除笔记（id={note_id}）" if removed else f"错误: 未找到笔记 {note_id}"


def _format_notes(notes: list[Any], *, empty: str) -> str:
    """把笔记列表格式化为紧凑文本。"""
    if not notes:
        return empty
    lines: list[str] = []
    for note in notes:
        summary = " ".join(str(note.content or "").split())
        snippet = summary[:80] + ("…" if len(summary) > 80 else "")
        tags = ",".join(note.tags) if note.tags else "-"
        lines.append(
            f"- [{note.type}] {note.title} (id={note.id}, tags={tags})"
            + (f"\n  {snippet}" if snippet else "")
        )
    return "\n".join(lines)


def _as_tags(value: Any) -> list[str] | None:
    """解析标签：数组、JSON 数组或逗号分隔字符串。"""
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except ValueError:
            return [item.strip() for item in text.replace("，", ",").split(",") if item.strip()]
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else None


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


def _optional_text(value: Any) -> str | None:
    """把可选字符串参数转为 None 或去空白文本。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_int(value: Any, default: int) -> int:
    """把参数转换为整数。"""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
