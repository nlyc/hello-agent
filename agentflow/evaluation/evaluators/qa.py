"""QA 评估器：对任意回答函数 / Agent / RAGPipeline 计算 QEM 与引用覆盖。"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

from agentflow.core.exceptions import EvaluationError
from agentflow.evaluation.metrics.qa import evaluate_qa
from agentflow.evaluation.types import EvalCase, EvalReport, EvalResult
from agentflow.memory.message_history import estimate_tokens


class QAEvaluator:
    """旅行问答评估。

    参数：
        answer_fn: 可调用对象，或带 ``run`` / ``answer`` 方法的对象。
        system: 报告中的系统名。
    """

    def __init__(self, answer_fn: Any, *, system: str = "qa") -> None:
        self.answer_fn = answer_fn
        self.system = system

    def run(
        self,
        cases: Sequence[EvalCase],
        *,
        limit: int | None = None,
        category: str | None = None,
    ) -> EvalReport:
        selected = [
            case
            for case in cases
            if not category or case.category == category
        ]
        if limit is not None and limit >= 0:
            selected = selected[:limit]

        results: list[EvalResult] = []
        for case in selected:
            started = time.perf_counter()
            error = ""
            output = ""
            citations: list[str] = []
            try:
                raw = self._call(case.question)
                output, citations = _extract_output(raw)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            latency = (time.perf_counter() - started) * 1000
            expected_citations = list(
                case.metadata.get("expected_citations", []) or []
            )
            gold = str(case.ground_truth if case.ground_truth is not None else "")
            metrics = evaluate_qa(
                output,
                gold,
                citations=citations,
                expected_citations=expected_citations,
            )
            metrics["error"] = error
            results.append(
                EvalResult(
                    case_id=case.id,
                    passed=bool(metrics.get("passed")) and not error,
                    score=1.0 if metrics.get("passed") and not error else 0.0,
                    metrics=metrics,
                    output=output,
                    error=error,
                    latency_ms=round(latency, 2),
                    tokens=estimate_tokens(output),
                )
            )
        return EvalReport(
            suite="qa",
            system=self.system,
            cases=results,
            metrics=_aggregate(results),
            manifest={"cases": len(results), "category": category or "all"},
        )

    def _call(self, question: str) -> Any:
        if callable(self.answer_fn):
            return self.answer_fn(question)
        if hasattr(self.answer_fn, "answer"):
            return self.answer_fn.answer(question)
        if hasattr(self.answer_fn, "run"):
            return self.answer_fn.run(question)
        raise EvaluationError("answer_fn 必须是 callable 或带 answer/run 方法的对象")


def _extract_output(raw: Any) -> tuple[str, list[str]]:
    """兼容字符串、RAGAnswer、Agent 输出。"""
    if isinstance(raw, str):
        return raw, _extract_citations(raw)
    answer = getattr(raw, "answer", None)
    citations = getattr(raw, "citations", None)
    if answer is not None:
        return str(answer), [str(item) for item in (citations or [])]
    return str(raw), _extract_citations(str(raw))


def _extract_citations(text: str) -> list[str]:
    """从「依据：」后提取引用标记。"""
    marker = text.find("依据：")
    if marker < 0:
        marker = text.find("依据:")
    if marker < 0:
        return []
    tail = text[marker + 3 :]
    return [
        item.strip()
        for item in tail.replace("；", "\n").replace(";", "\n").splitlines()
        if item.strip()
    ]


def _aggregate(results: Sequence[EvalResult]) -> dict[str, Any]:
    total = len(results)
    if not total:
        return {"cases": 0, "passed": 0, "accuracy": 0.0, "qem": 0.0}
    return {
        "cases": total,
        "passed": sum(1 for item in results if item.passed),
        "accuracy": sum(1 for item in results if item.passed) / total,
        "qem": sum(
            1 for item in results if item.metrics.get("exact_match")
        ) / total,
        "contains": sum(
            1 for item in results if item.metrics.get("contains")
        ) / total,
        "citation_coverage": sum(
            float(item.metrics.get("citation_coverage", 0.0))
            for item in results
        ) / total,
        "errors": sum(1 for item in results if item.error),
    }
