"""A2A 最小数据模型：AgentCard / Skill / Task / Artifact / 生命周期。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agentflow.core.exceptions import A2AError


def utc_now() -> datetime:
    """返回当前 UTC 时间（带时区）。"""
    return datetime.now(timezone.utc)


class TaskStatus(str, Enum):
    """任务生命周期：submitted → working → completed / failed / canceled。"""

    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

    @property
    def terminal(self) -> bool:
        """是否终态。"""
        return self in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELED,
        )


_ALLOWED_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.SUBMITTED: {
        TaskStatus.WORKING,
        TaskStatus.FAILED,
        TaskStatus.CANCELED,
    },
    TaskStatus.WORKING: {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELED,
    },
    TaskStatus.COMPLETED: set(),
    TaskStatus.FAILED: set(),
    TaskStatus.CANCELED: set(),
}


@dataclass
class Skill:
    """Agent 暴露的一项能力。"""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Skill":
        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            input_schema=dict(data.get("input_schema", {}) or {}),
            output_schema=dict(data.get("output_schema", {}) or {}),
        )


@dataclass
class AgentCard:
    """Agent 的自描述卡片，暴露在 ``/.well-known/agent-card.json``。"""

    name: str
    description: str = ""
    url: str = ""
    version: str = "2.0.2"
    skills: list[Skill] = field(default_factory=list)
    capabilities: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "skills": [skill.to_dict() for skill in self.skills],
            "capabilities": dict(self.capabilities),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentCard":
        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            url=str(data.get("url", "")),
            version=str(data.get("version", "2.0.2")),
            skills=[Skill.from_dict(item) for item in data.get("skills", []) or []],
            capabilities=dict(data.get("capabilities", {}) or {}),
        )


@dataclass
class Artifact:
    """任务产出的工件（文本 / JSON / 文件描述）。"""

    name: str = "result"
    mime: str = "text/plain"
    content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "mime": self.mime, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Artifact":
        return cls(
            name=str(data.get("name", "result")),
            mime=str(data.get("mime", "text/plain")),
            content=str(data.get("content", "")),
        )


@dataclass
class Task:
    """一次 A2A 任务；生命周期由 ``transition`` 严格校验。"""

    id: str
    skill: str
    status: TaskStatus = TaskStatus.SUBMITTED
    input: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    error: str = ""
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    @classmethod
    def create(cls, skill: str, payload: dict[str, Any] | None = None) -> "Task":
        return cls(
            id=f"task-{uuid.uuid4().hex[:12]}",
            skill=str(skill),
            status=TaskStatus.SUBMITTED,
            input=dict(payload or {}),
        )

    def transition(self, status: TaskStatus | str) -> "Task":
        """按生命周期迁移；非法迁移抛 ``A2AError``。"""
        target = TaskStatus(status)
        if target == self.status:
            return self
        allowed = _ALLOWED_TRANSITIONS.get(self.status, set())
        if target not in allowed:
            raise A2AError(
                f"非法任务状态迁移：{self.status.value} → {target.value}"
            )
        self.status = target
        self.updated_at = utc_now()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "skill": self.skill,
            "status": self.status.value,
            "input": dict(self.input),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "error": self.error,
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "updated_at": self.updated_at.isoformat(timespec="seconds"),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        return cls(
            id=str(data.get("id", "")),
            skill=str(data.get("skill", "")),
            status=TaskStatus(str(data.get("status", "submitted"))),
            input=dict(data.get("input", {}) or {}),
            artifacts=[
                Artifact.from_dict(item) for item in data.get("artifacts", []) or []
            ],
            error=str(data.get("error", "")),
        )
