"""A2A 类型与生命周期测试。"""

from datetime import datetime, timezone

import pytest

from agentflow.core.exceptions import A2AError
from agentflow.protocols.a2a import (
    AgentCard,
    Artifact,
    Skill,
    Task,
    TaskStatus,
)


def test_skill_and_card_roundtrip():
    card = AgentCard(
        name="研究助手",
        description="查资料",
        url="http://127.0.0.1:5001",
        skills=[
            Skill(name="gather_facts", description="收集事实", input_schema={"type": "object"})
        ],
    )
    data = card.to_dict()
    restored = AgentCard.from_dict(data)
    assert restored.name == "研究助手"
    assert restored.skills[0].name == "gather_facts"
    assert restored.to_dict() == data


def test_task_lifecycle_success():
    task = Task.create("plan", {"city": "北京"})
    assert task.status == TaskStatus.SUBMITTED
    task.transition(TaskStatus.WORKING)
    task.transition(TaskStatus.COMPLETED)
    assert task.status.terminal
    assert Task.from_dict(task.to_dict()).status == TaskStatus.COMPLETED


def test_task_lifecycle_failure_and_cancel():
    task = Task.create("plan")
    task.transition("working")
    task.transition("failed")
    assert task.status == TaskStatus.FAILED

    other = Task.create("plan")
    other.transition(TaskStatus.CANCELED)
    assert other.status == TaskStatus.CANCELED


def test_illegal_transition_rejected():
    task = Task.create("plan")
    task.transition(TaskStatus.WORKING)
    task.transition(TaskStatus.COMPLETED)
    with pytest.raises(A2AError):
        task.transition(TaskStatus.WORKING)
    with pytest.raises(A2AError):
        Task.create("x").transition(TaskStatus.COMPLETED)


def test_artifact_roundtrip():
    artifact = Artifact(name="itinerary", mime="text/plain", content="D1 故宫")
    assert Artifact.from_dict(artifact.to_dict()) == artifact


def test_task_timestamps_are_utc():
    task = Task.create("plan")
    assert task.created_at.tzinfo is not None
    assert task.updated_at.tzinfo is not None
    assert isinstance(task.created_at, datetime)
    assert task.created_at.tzinfo == timezone.utc
