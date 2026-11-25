"""NoteContextSource：把 NoteStore 接入 ContextBuilder。

取料顺序：全部 blocker → 关键词搜索 → 全部 action。blocker 优先级最高，
确保「故宫周一闭馆」这类阻塞不会被普通检索结果挤掉。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agentflow.context.types import ContextPacket, ensure_aware
from agentflow.notes.store import NoteStore
from agentflow.notes.types import NOTE_PRIORITY, Note


class NoteContextSource:
    """笔记来源：结构化项目记忆 → ContextPacket。"""

    name = "note"

    def __init__(self, store: NoteStore, *, include_task_state: bool = True) -> None:
        self.store = store
        self.include_task_state = include_task_state

    def gather(
        self, query: str, *, limit: int, now: datetime
    ) -> list[ContextPacket]:
        if limit <= 0:
            return []
        current = ensure_aware(now)
        notes: list[Note] = []
        notes.extend(self.store.list(note_type="blocker", limit=limit))
        notes.extend(self.store.search(query, limit=limit))
        notes.extend(self.store.list(note_type="action", limit=limit))
        if self.include_task_state:
            notes.extend(self.store.list(note_type="task_state", limit=limit))

        packets: list[ContextPacket] = []
        seen: set[str] = set()
        for note in notes:
            if note.id in seen:
                continue
            if not self.include_task_state and note.type == "task_state":
                continue
            seen.add(note.id)
            packets.append(_note_to_packet(note, current))
        packets.sort(key=lambda packet: (-packet.relevance_score, -packet.timestamp.timestamp()))
        return packets[:limit]


def _note_to_packet(note: Note, now: datetime) -> ContextPacket:
    """标题 + 正文转信息包；blocker / action 进入 State 分区。"""
    content = note.title if not note.content else f"{note.title}\n{note.content}"
    section = "state" if note.type in ("blocker", "action") else "context"
    return ContextPacket(
        content=content,
        timestamp=note.updated_at or now,
        relevance_score=NOTE_PRIORITY.get(note.type, 0.5),
        source="note",
        section=section,
        metadata={
            "note_id": note.id,
            "type": note.type,
            "tags": list(note.tags),
        },
        trusted=True,
    )
