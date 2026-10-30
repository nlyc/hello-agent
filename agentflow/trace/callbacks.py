"""内置回调：Console / JSONL 落盘 / 统计。"""

import json
import sys
from collections import Counter
from datetime import datetime
from typing import Optional, TextIO

from agentflow.trace.event import TraceEvent


class ConsoleCallback:
    """把事件打印到控制台（默认 stdout，可注入输出流便于测试）。"""

    def __init__(self, out: Optional[TextIO] = None):
        self.out = out

    def __call__(self, event: TraceEvent) -> None:
        target = self.out or sys.stdout
        print(f"[trace] {event.type} agent={event.agent_name} step={event.step}", file=target)


class JSONLFileCallback:
    """把事件逐行落盘为 JSONL，直接驱动轨迹对比报告。"""

    def __init__(self, path: str):
        self._file = open(path, "a", encoding="utf-8")

    def __call__(self, event: TraceEvent) -> None:
        self._file.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class StatsCallback:
    """按事件类型计数 + 统计 token 与端到端耗时。"""

    def __init__(self):
        self.counts: Counter = Counter()
        self.tokens = 0
        self.first_ts: Optional[datetime] = None
        self.last_ts: Optional[datetime] = None

    def __call__(self, event: TraceEvent) -> None:
        self.counts[event.type] += 1
        if self.first_ts is None or event.timestamp < self.first_ts:
            self.first_ts = event.timestamp
        if self.last_ts is None or event.timestamp > self.last_ts:
            self.last_ts = event.timestamp
        tokens = event.data.get("tokens")
        if isinstance(tokens, (int, float)):
            self.tokens += int(tokens)

    def summary(self) -> dict:
        duration = 0.0
        if self.first_ts is not None and self.last_ts is not None:
            duration = (self.last_ts - self.first_ts).total_seconds()
        return {
            "counts": dict(self.counts),
            "llm_calls": self.counts.get("llm", 0),
            "tool_calls": self.counts.get("tool", 0),
            "tokens": self.tokens,
            "duration_seconds": round(duration, 3),
        }
