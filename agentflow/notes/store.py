"""NoteStore：Markdown + YAML front matter 的项目式笔记存储。

设计要点：
- 文件名即 ID，正文是人类可读的 Markdown，天然 Git 友好；
- ``notes_index.json`` 只做快速检索与完整性校验，文件本身才是事实来源；
- 原子写入，避免半截文件；轻量 front matter 解析，不引入 PyYAML。
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from agentflow.core.env import get_env
from agentflow.core.exceptions import NoteError
from agentflow.notes.types import NOTE_TYPES, Note, ensure_aware, utc_now
from agentflow.retrieval.embedder import tokenize

_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-一-鿿]+$")


class NoteStore:
    """按用户隔离的笔记仓库。

    参数：
        root: 笔记根目录；默认 ``NOTES_DIR`` 或 ``.data/notes``。
        user_id: 用户隔离目录；默认 ``USER_ID`` 或 ``default``。
    """

    def __init__(
        self,
        root: str | Path | None = None,
        user_id: str | None = None,
    ) -> None:
        base = Path(root) if root is not None else Path(
            get_env("NOTES_DIR", ".data/notes")
        )
        self.root = base.expanduser()
        self.user_id = str(user_id or get_env("USER_ID", "default") or "default")
        self.directory = self.root / self.user_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "notes_index.json"
        self.last_errors: list[dict[str, str]] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ CRUD

    def create(
        self,
        title: str,
        content: str = "",
        *,
        type: str = "task_state",
        tags: Sequence[str] | None = None,
        note_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Note:
        """新建笔记；未指定 ID 时按时间戳 + 标题生成。"""
        with self._lock:
            title_text = str(title or "").strip() or "未命名笔记"
            identifier = str(note_id or "").strip() or self._new_id(title_text)
            self._validate_id(identifier)
            path = self._note_path(identifier)
            if path.exists():
                raise NoteError(f"笔记已存在：{identifier}")
            now = utc_now()
            note = Note(
                id=identifier,
                title=title_text,
                type=type,
                tags=list(tags or []),
                created_at=now,
                updated_at=now,
                content=content,
                metadata=dict(metadata or {}),
            )
            self._write_note(note)
            self._update_index(note)
            return note

    def read(self, note_id: str) -> Note:
        """按 ID 读取笔记；文件不存在时抛出 ``NoteError``。"""
        with self._lock:
            self._validate_id(note_id)
            path = self._note_path(note_id)
            if not path.exists():
                raise NoteError(f"未找到笔记：{note_id}")
            return self._parse_note(path)

    def update(
        self,
        note_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        type: str | None = None,
        tags: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Note:
        """更新笔记字段；只更新显式传入的字段。"""
        with self._lock:
            note = self.read(note_id)
            if title is not None:
                note.title = str(title).strip() or note.title
            if content is not None:
                note.content = str(content).strip()
            if type is not None:
                note.type = str(type).strip().lower()
                if note.type not in NOTE_TYPES:
                    raise NoteError(
                        f"未知笔记类型 '{note.type}'，可用：{' / '.join(NOTE_TYPES)}"
                    )
            if tags is not None:
                note.tags = [
                    str(tag).strip() for tag in tags if str(tag).strip()
                ]
            if metadata is not None:
                merged = dict(note.metadata)
                merged.update(metadata)
                note.metadata = merged
            note.updated_at = utc_now()
            self._write_note(note)
            self._update_index(note)
            return note

    def delete(self, note_id: str) -> bool:
        """删除笔记文件与索引项；返回是否确实存在。"""
        with self._lock:
            self._validate_id(note_id)
            path = self._note_path(note_id)
            if not path.exists():
                return False
            path.unlink()
            self._remove_from_index(note_id)
            return True

    # --------------------------------------------------------------- queries

    def list(
        self,
        *,
        note_type: str | None = None,
        tags: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[Note]:
        """列出笔记，按更新时间降序。"""
        notes = self._load_all()
        result = [note for note in notes if self._matches(note, note_type, tags)]
        if limit is not None and limit >= 0:
            result = result[:limit]
        return result

    def search(
        self,
        query: str,
        *,
        note_type: str | None = None,
        tags: Sequence[str] | None = None,
        limit: int = 10,
    ) -> list[Note]:
        """关键词检索：标题 + 正文 + 标签；按分数与更新时间排序。"""
        text = str(query or "").strip()
        if not text:
            return self.list(note_type=note_type, tags=tags, limit=limit)
        notes = [
            note
            for note in self._load_all()
            if self._matches(note, note_type, tags)
        ]
        scored = [
            (self._keyword_score(text, note), note)
            for note in notes
        ]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], -item[1].updated_at.timestamp()))
        if limit >= 0:
            scored = scored[:limit]
        return [note for _, note in scored]

    def summary(self) -> dict[str, Any]:
        """返回类型分布、未解决 blocker、最近更新与索引一致性。"""
        notes = self.list()
        counts = Counter(note.type for note in notes)
        recent = sorted(
            notes, key=lambda note: note.updated_at, reverse=True
        )[:5]
        index_ids = set(self._read_index().get("notes", {}))
        file_ids = {note.id for note in notes}
        return {
            "total": len(notes),
            "types": dict(sorted(counts.items())),
            "unresolved_blockers": counts.get("blocker", 0),
            "recent": [note.to_dict() for note in recent],
            "index_consistent": index_ids == file_ids,
            "directory": str(self.directory),
        }

    def stats(self) -> dict[str, Any]:
        """``summary`` 的别名，便于统一统计接口。"""
        return self.summary()

    # ---------------------------------------------------------------- indexes

    def rebuild_index(self) -> dict[str, Any]:
        """扫描目录重建索引，并报告损坏文件。"""
        with self._lock:
            notes = self._load_all()
            index = {
                "version": 1,
                "notes": {note.id: self._index_entry(note) for note in notes},
            }
            self._write_index(index)
            return {
                "notes": len(notes),
                "errors": list(self.last_errors),
                "index": str(self.index_path),
            }

    # --------------------------------------------------------------- internal

    def _note_path(self, note_id: str) -> Path:
        return self.directory / f"{note_id}.md"

    def _validate_id(self, note_id: str) -> None:
        text = str(note_id or "").strip()
        if not text or not _SAFE_ID.match(text):
            raise NoteError(f"非法笔记 ID：{note_id!r}")

    def _new_id(self, title: str) -> str:
        stamp = utc_now().strftime("%Y%m%d-%H%M%S")
        slug = _slug(title)
        base = f"note-{stamp}-{slug}" if slug else f"note-{stamp}"
        candidate = base
        suffix = 2
        while self._note_path(candidate).exists():
            candidate = f"{base}-{suffix}"
            suffix += 1
        return candidate

    def _write_note(self, note: Note) -> None:
        path = self._note_path(note.id)
        _atomic_write(path, _render_note(note))

    def _parse_note(self, path: Path) -> Note:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise NoteError(f"读取笔记失败：{exc}") from exc
        front, body = _split_front_matter(text)
        data = dict(front)
        data.setdefault("id", path.stem)
        data.setdefault("title", path.stem)
        data.setdefault("type", "task_state")
        data["tags"] = _as_tags(data.get("tags"))
        data["content"] = body
        data["metadata"] = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        try:
            return Note.from_dict(data)
        except NoteError:
            raise
        except Exception as exc:
            raise NoteError(f"解析笔记失败：{path.name}: {exc}") from exc

    def _load_all(self) -> list[Note]:
        notes: list[Note] = []
        errors: list[dict[str, str]] = []
        for path in sorted(self.directory.glob("*.md")):
            try:
                notes.append(self._parse_note(path))
            except NoteError as exc:
                errors.append({"file": path.name, "error": str(exc)})
        self.last_errors = errors
        notes.sort(key=lambda note: note.updated_at, reverse=True)
        return notes

    def _matches(
        self,
        note: Note,
        note_type: str | None,
        tags: Sequence[str] | None,
    ) -> bool:
        if note_type and note.type != str(note_type).strip().lower():
            return False
        expected = {str(tag).strip() for tag in (tags or []) if str(tag).strip()}
        if expected and not (expected & set(note.tags)):
            return False
        return True

    def _keyword_score(self, query: str, note: Note) -> float:
        query_terms = set(tokenize(query))
        if not query_terms:
            return 0.0
        text = f"{note.title}\n{note.content}\n{' '.join(note.tags)}"
        note_terms = set(tokenize(text))
        if not note_terms:
            return 0.0
        overlap = len(query_terms & note_terms)
        if overlap <= 0:
            return 0.0
        score = overlap / len(query_terms)
        title_terms = set(tokenize(note.title))
        if query_terms & title_terms:
            score += 0.2
        return min(1.5, score)

    def _index_entry(self, note: Note) -> dict[str, Any]:
        return {
            "id": note.id,
            "title": note.title,
            "type": note.type,
            "tags": list(note.tags),
            "created_at": note.created_at.isoformat(timespec="seconds"),
            "updated_at": note.updated_at.isoformat(timespec="seconds"),
        }

    def _read_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"version": 1, "notes": {}}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"version": 1, "notes": {}}
        if not isinstance(data, dict) or not isinstance(data.get("notes"), dict):
            return {"version": 1, "notes": {}}
        return data

    def _write_index(self, index: dict[str, Any]) -> None:
        _atomic_write(
            self.index_path,
            json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    def _update_index(self, note: Note) -> None:
        index = self._read_index()
        notes = dict(index.get("notes", {}))
        notes[note.id] = self._index_entry(note)
        index["notes"] = notes
        self._write_index(index)

    def _remove_from_index(self, note_id: str) -> None:
        index = self._read_index()
        notes = dict(index.get("notes", {}))
        notes.pop(note_id, None)
        index["notes"] = notes
        self._write_index(index)


def _render_note(note: Note) -> str:
    """把 Note 序列化为 front matter + Markdown。"""
    front: dict[str, Any] = {
        "id": note.id,
        "title": note.title,
        "type": note.type,
        "tags": list(note.tags),
        "created_at": note.created_at.isoformat(timespec="seconds"),
        "updated_at": note.updated_at.isoformat(timespec="seconds"),
    }
    if note.metadata:
        front["metadata"] = dict(note.metadata)
    lines = ["---"]
    for key, value in front.items():
        # JSON 字符串是合法 YAML，避免手写转义。
        lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    lines.append("---")
    if note.content:
        lines.append(note.content)
    return "\n".join(lines) + "\n"


def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """拆分 YAML front matter 与正文；未闭合时报错。"""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()
    end = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end = index
            break
    if end is None:
        raise NoteError("front matter 未闭合（缺少结束 ---）")
    data: dict[str, Any] = {}
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        if key:
            data[key] = _parse_scalar(raw.strip())
    body = "\n".join(lines[end + 1 :]).lstrip("\n")
    return data, body


def _parse_scalar(raw: str) -> Any:
    """解析前置元数据值：优先 JSON，兼容手写裸值。"""
    if raw == "":
        return ""
    try:
        return json.loads(raw)
    except ValueError:
        pass
    text = raw.strip().strip('"').strip("'")
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [
            item.strip().strip('"').strip("'")
            for item in inner.split(",")
            if item.strip()
        ]
    return text


def _as_tags(value: Any) -> list[str]:
    """把标签统一成字符串列表。"""
    if value is None:
        return []
    if isinstance(value, str):
        return [tag.strip() for tag in re.split(r"[,，、\s]+", value) if tag.strip()]
    if isinstance(value, Iterable):
        return [str(tag).strip() for tag in value if str(tag).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _slug(text: str) -> str:
    """生成文件名安全的中文/英文短标识。"""
    slug = re.sub(r"[^\w一-鿿-]+", "-", str(text or "").strip())
    slug = slug.strip("-_")
    return slug[:32]


def _atomic_write(path: Path, text: str) -> None:
    """同目录临时文件 + ``os.replace``，避免半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)
