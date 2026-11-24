"""行程质量评估器：程序化约束 + 可选 LLM Judge。"""

from __future__ import annotations

import time
from typing import Any, Sequence

from agentflow.core.exceptions import EvaluationError
from agentflow.evaluation.metrics.constraints import check_itinerary
from agentflow.evaluation.metrics.judge import LLMJudge
from agentflow.evaluation.types import EvalCase, EvalReport, EvalResult
from agentflow.memory.message_history import estimate_tokens


class ItineraryQualityEvaluator:
    """对生成式行程打分：硬约束优先，Judge 管质量。"""

    def __init__(self, agent: Any, *, judge: LLMJudge | None = None) -> None:
        if not hasattr(agent, "run"):
            raise EvaluationError("ItineraryQualityEvaluator 需要提供 run() 的 Agent")
        self.agent = agent
        self.judge = judge

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
            constraints = dict(
                case.payload.get("constraints")
                or case.metadata.get("constraints")
                or {}
            )
            try:
                output = str(self.agent.run(case.question) or "")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            latency = (time.perf_counter() - started) * 1000

            constraint_result = check_itinerary(output, constraints)
            metrics: dict[str, Any] = {
                "constraints": constraint_result,
                "constraints_passed": constraint_result["passed"],
            }
            judge_passed = True
            if self.judge is not None and output and not error:
                judge_result = self.judge.judge(
                    case.question,
                    output,
                    reference=str(case.metadata.get("reference", "") or ""),
                )
                metrics["judge"] = judge_result.to_dict()
                metrics["judge_average"] = judge_result.average
                judge_passed = judge_result.passed and not judge_result.error
            passed = (
                not error
                and constraint_result["passed"]
                and judge_passed
            )
            results.append(
                EvalResult(
                    case_id=case.id,
                    passed=passed,
                    score=float(metrics.get("judge_average", 1.0 if passed else 0.0)),
                    metrics=metrics,
                    output=output,
                    error=error,
                    latency_ms=round(latency, 2),
                    tokens=estimate_tokens(output),
                )
            )
        return EvalReport(
            suite="quality",
            system=type(self.agent).__name__,
            cases=results,
            metrics=_aggregate(results),
            manifest={"cases": len(results), "category": category or "all"},
        )


def _aggregate(results: Sequence[EvalResult]) -> dict[str, Any]:
    total = len(results)
    if not total:
        return {"cases": 0, "passed": 0, "accuracy": 0.0}
    judge_values = [
        float(item.metrics.get("judge_average", 0.0))
        for item in results
        if "judge_average" in item.metrics
    ]
    return {
        "cases": total,
        "passed": sum(1 for item in results if item.passed),
        "accuracy": sum(1 for item in results if item.passed) / total,
        "constraint_pass_rate": sum(
            1 for item in results if item.metrics.get("constraints_passed")
        ) / total,
        "judge_cases": len(judge_values),
        "judge_average": (
            sum(judge_values) / len(judge_values) if judge_values else 0.0
        ),
        "errors": sum(1 for item in results if item.error),
    }
