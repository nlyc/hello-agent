"""agentflow.protocols.a2a —— 最小 A2A：AgentCard / Task / Artifact / Server / Client / Tool。"""

from agentflow.protocols.a2a.client import A2AClient
from agentflow.protocols.a2a.server import A2AServer
from agentflow.protocols.a2a.tool import A2ATool
from agentflow.protocols.a2a.types import (
    AgentCard,
    Artifact,
    Skill,
    Task,
    TaskStatus,
)

__all__ = [
    "A2AClient",
    "A2AServer",
    "A2ATool",
    "AgentCard",
    "Artifact",
    "Skill",
    "Task",
    "TaskStatus",
]
