"""Tracer：事件流 + 回调订阅/过滤，回调异常不炸业务。"""

import threading
import uuid
from typing import Callable, Optional

from agentflow.trace.event import TraceEvent


class Tracer:
    """全链路追踪中枢。emit 广播事件，subscribe 注册回调（可按 type/agent 过滤）。"""

    def __init__(self, trace_id: str = ""):
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self._callbacks: list[Callable[[TraceEvent], None]] = []
        self._lock = threading.RLock()

    def subscribe(
        self,
        callback: Callable[[TraceEvent], None],
        *,
        event_type: Optional[str] = None,
        agent_name: Optional[str] = None,
        filter_fn: Optional[Callable[[TraceEvent], bool]] = None,
    ) -> Callable[[TraceEvent], None]:
        """注册回调，返回包装后的句柄（供 unsubscribe）。过滤条件全部满足才触发。"""

        def wrapped(event: TraceEvent) -> None:
            if event_type is not None and event.type != event_type:
                return
            if agent_name is not None and event.agent_name != agent_name:
                return
            if filter_fn is not None and not filter_fn(event):
                return
            callback(event)

        with self._lock:
            self._callbacks.append(wrapped)
        return wrapped

    def unsubscribe(self, callback: Callable[[TraceEvent], None]) -> None:
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def emit(
        self,
        event_type: str,
        agent_name: str = "",
        step: int = 0,
        data: Optional[dict] = None,
    ) -> None:
        event = TraceEvent(
            type=event_type,
            agent_name=agent_name,
            step=step,
            data=data or {},
            trace_id=self.trace_id,
        )
        with self._lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                pass  # 回调异常不炸业务

    def __len__(self) -> int:
        return len(self._callbacks)
