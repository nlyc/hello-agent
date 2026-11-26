"""工具调用评估器：从 Tracer 的 ``tool`` 事件提取实际调用并计算 AST 匹配。"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Iterable, Sequence

from agentflow.core.exceptions import EvaluationError
from agentflow.evaluation.metrics.tool_calling import (
    ToolCall,
    evaluate_tool_calls,
)
from agentflow.evaluation.types import EvalCase, EvalReport, EvalResult
from agentflow.trace import Tracer
from agentflow.trace.event import TraceEvent


class ToolCallCollector:
    """订阅 TraceEvent，把 ``tool`` 事件转成 ToolCall。"""

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    def __call__(self, event: TraceEvent) -> None:
        if event.type != "tool":
            return
        name = str(event.data.get("tool", "") or "")
        params = event.data.get("params", {}) or {}
        if name:
            self.calls.append(ToolCall(name=name, arguments=dict(params)))

    def reset(self) -> None:
        self.calls.clear()


class ToolCallingEvaluator:
    """同一个 Agent 在 mini-BFCL 上的工具调用准确率。"""

    def __init__(self, agent: Any, *, tracer: Tracer | None = None) -> None:
        if not hasattr(agent, "run"):
            raise EvaluationError("ToolCallingEvaluator 需要提供 run() 的 Agent")
        self.agent = agent
        self.tracer = tracer
        self.collector = ToolCallCollector()

    def run(
        self,
        cases: Sequence[EvalCase],
        *,
        category: str | None = None,
        limit: int | None = None,
        system: str | None = None,
    ) -> EvalReport:
        """逐题运行 Agent；单题异常不阻断整批。"""
        selected = [
            case
            for case in cases
            if not category or case.category == category
        ]
        if limit is not None and limit >= 0:
            selected = selected[:limit]

        old_tracer = getattr(self.agent, "tracer", None)
        tracer = self.tracer or old_tracer or Tracer()
        handle = tracer.subscribe(self.collector, event_type="tool")
        self.agent.tracer = tracer
        results: list[EvalResult] = []
        try:
            for case in selected:
                self.collector.reset()
                started = time.perf_counter()
                error = ""
                try:
                    self.agent.run(case.question)
                except Exception as exc:  # 记录失败但继续
                    error = f"{type(exc).__name__}: {exc}"
                latency = (time.perf_counter() - started) * 1000
                expected = _expected_calls(case)
                loose_fields = list(case.metadata.get("loose_fields", []) or [])
                metrics = evaluate_tool_calls(
                    expected,
                    self.collector.calls,
                    category=case.category,
                    loose_fields=loose_fields,
                )
                metrics["category"] = case.category
                metrics["loose_fields"] = loose_fields
                metrics["expected"] = [call.to_dict() for call in expected]
                metrics["actual"] = [
                    call.to_dict() for call in self.collector.calls
                ]
                results.append(
                    EvalResult(
                        case_id=case.id,
                        passed=bool(metrics.get("passed")) and not error,
                        score=1.0 if metrics.get("passed") and not error else 0.0,
                        metrics=metrics,
                        error=error,
                        latency_ms=round(latency, 2),
                    )
                )
        finally:
            tracer.unsubscribe(handle)
            self.agent.tracer = old_tracer

        return EvalReport(
            suite="tool_calling",
            system=system or type(self.agent).__name__,
            cases=results,
            metrics=_aggregate(results),
            manifest={"cases": len(results), "category": category or "all"},
        )


def _expected_calls(case: EvalCase) -> list[ToolCall]:
    """从 ground_truth 解析期望调用。"""
    truth = case.ground_truth
    if isinstance(truth, dict):
        truth = truth.get("calls", [])
    if truth is None:
        return []
    if not isinstance(truth, (list, tuple)):
        truth = [truth]
    return [ToolCall.from_dict(item) for item in truth]


def _aggregate(results: Iterable[EvalResult]) -> dict[str, Any]:
    """总体准确率 + 分类准确率。"""
    items = list(results)
    total = len(items)
    passed = sum(1 for item in items if item.passed)
    by_category: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"cases": 0, "passed": 0}
    )
    for item in items:
        category = str(item.metrics.get("category", "") or "unknown")
        by_category[category]["cases"] += 1
        if item.passed:
            by_category[category]["passed"] += 1
    for value in by_category.values():
        value["accuracy"] = (
            value["passed"] / value["cases"] if value["cases"] else 0.0
        )
    return {
        "cases": total,
        "passed": passed,
        "accuracy": passed / total if total else 0.0,
        "extra_calls": sum(
            int(item.metrics.get("extra_calls", 0)) for item in items
        ),
        "missing_calls": sum(
            int(item.metrics.get("missing_calls", 0)) for item in items
        ),
        "by_category": dict(sorted(by_category.items())),
    }
