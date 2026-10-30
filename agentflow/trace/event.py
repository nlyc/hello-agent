"""TraceEvent：全链路追踪的事件载体。"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TraceEvent:
    """一次关键动作的记录：type 标识动作类别，data 携带上下文。"""

    type: str
    agent_name: str = ""
    step: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    trace_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "agent_name": self.agent_name,
            "step": self.step,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(timespec="milliseconds"),
            "trace_id": self.trace_id,
        }
