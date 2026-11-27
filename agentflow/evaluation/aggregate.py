"""多次采样的评估聚合：通过率、Wilson 区间、逐题稳定性与失败样例。

重复运行同一基准时，样本并非独立同分布（同一题被重复采样），
因此这里的 95% 区间按“尝试次数”汇总，作为描述性不确定性，
不应当作严格的独立伯努利置信区间。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from agentflow.evaluation.types import EvalReport, EvalResult

Z_95 = 1.96


def wilson_interval(
    successes: int, total: int, *, z: float = Z_95
) -> tuple[float, float]:
    """二项比例的 Wilson 得分区间；``total=0`` 时返回 ``(0, 0)``。"""
    if total <= 0:
        return (0.0, 0.0)
    successes = max(0, min(int(successes), int(total)))
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    """样本标准差（n-1）；少于 2 个值时返回 0。"""
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


def _value_summary(values: Sequence[float]) -> dict[str, Any]:
    """均值 / 标准差 / 最小 / 最大；分数区间用正态近似。"""
    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "ci95_low": None,
            "ci95_high": None,
        }
    mean = _mean(values)
    std = _std(values)
    if len(values) < 2:
        low = high = None
    else:
        margin = Z_95 * std / math.sqrt(len(values))
        low, high = mean - margin, mean + margin
    return {
        "n": len(values),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "ci95_low": None if low is None else round(low, 4),
        "ci95_high": None if high is None else round(high, 4),
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _case_status(passes: int, attempts: int) -> str:
    if attempts <= 0:
        return "missing"
    if passes == attempts:
        return "always_pass"
    if passes == 0:
        return "always_fail"
    return "flaky"


def _failure_detail(result: EvalResult) -> dict[str, Any]:
    """只保留定位失败所需字段，避免把整段轨迹塞进汇总。"""
    metrics = dict(result.metrics or {})
    detail: dict[str, Any] = {}
    if "expected" in metrics or "actual" in metrics:
        detail["expected"] = metrics.get("expected")
        detail["actual"] = metrics.get("actual")
        detail["extra_calls"] = metrics.get("extra_calls")
        detail["missing_calls"] = metrics.get("missing_calls")
    if "answer" in metrics:
        detail["answer"] = metrics.get("answer")
        detail["citations"] = metrics.get("citations")
        detail["expected_citations"] = metrics.get("expected_citations")
    if "constraints" in metrics:
        detail["constraints"] = metrics.get("constraints")
    if "judge" in metrics:
        judge = metrics.get("judge") or {}
        detail["judge"] = {
            "average": judge.get("average"),
            "passed": judge.get("passed"),
            "comment": judge.get("comment"),
            "error": judge.get("error"),
        }
    return detail


def aggregate_reports(
    reports: Sequence[EvalReport],
    *,
    label: str = "",
    max_failure_examples: int = 8,
) -> dict[str, Any]:
    """把同一系统、同一数据集的多次运行聚合为一个可序列化字典。"""
    items = list(reports)
    if not items:
        raise ValueError("aggregate_reports 至少需要一份 EvalReport")

    case_order: list[str] = []
    by_case: dict[str, list[tuple[int, EvalResult]]] = {}
    for run_index, report in enumerate(items, start=1):
        for result in report.cases:
            if result.case_id not in by_case:
                case_order.append(result.case_id)
                by_case[result.case_id] = []
            by_case[result.case_id].append((run_index, result))

    all_results = [result for report in items for result in report.cases]
    total_attempts = len(all_results)
    passed_attempts = sum(1 for result in all_results if result.passed)
    ci_low, ci_high = wilson_interval(passed_attempts, total_attempts)

    per_case: list[dict[str, Any]] = []
    always_pass: list[str] = []
    always_fail: list[str] = []
    flaky: list[str] = []
    for case_id in case_order:
        attempts = by_case[case_id]
        passes = sum(1 for _, result in attempts if result.passed)
        scores = [float(result.score) for _, result in attempts]
        judge_values = [
            float(result.metrics["judge_average"])
            for _, result in attempts
            if _number(result.metrics.get("judge_average")) is not None
        ]
        constraint_values = [
            float(bool(result.metrics["constraints_passed"]))
            for _, result in attempts
            if "constraints_passed" in result.metrics
        ]
        status = _case_status(passes, len(attempts))
        if status == "always_pass":
            always_pass.append(case_id)
        elif status == "always_fail":
            always_fail.append(case_id)
        elif status == "flaky":
            flaky.append(case_id)
        category = ""
        for _, result in attempts:
            category = str(result.metrics.get("category", "") or "")
            if category:
                break
        per_case.append(
            {
                "case_id": case_id,
                "category": category,
                "attempts": len(attempts),
                "passes": passes,
                "pass_rate": round(passes / len(attempts), 4) if attempts else 0.0,
                "score_mean": round(_mean(scores), 4),
                "score_std": round(_std(scores), 4),
                "judge_average": (
                    round(_mean(judge_values), 4) if judge_values else None
                ),
                "constraint_pass_rate": (
                    round(_mean(constraint_values), 4)
                    if constraint_values
                    else None
                ),
                "errors": sum(1 for _, result in attempts if result.error),
                "status": status,
            }
        )

    run_pass_rates: list[float] = []
    run_scores: list[float] = []
    runs: list[dict[str, Any]] = []
    for run_index, report in enumerate(items, start=1):
        cases = len(report.cases)
        passed = sum(1 for result in report.cases if result.passed)
        accuracy = passed / cases if cases else 0.0
        score_mean = _mean([float(result.score) for result in report.cases])
        run_pass_rates.append(accuracy)
        run_scores.append(score_mean)
        runs.append(
            {
                "run": run_index,
                "cases": cases,
                "passed": passed,
                "accuracy": round(accuracy, 4),
                "score_mean": round(score_mean, 4),
            }
        )

    judge_values_all = [
        float(result.metrics["judge_average"])
        for result in all_results
        if _number(result.metrics.get("judge_average")) is not None
    ]
    constraint_values_all = [
        float(bool(result.metrics["constraints_passed"]))
        for result in all_results
        if "constraints_passed" in result.metrics
    ]

    failure_examples: list[dict[str, Any]] = []
    for run_index, report in enumerate(items, start=1):
        for result in report.cases:
            if result.passed:
                continue
            failure_examples.append(
                {
                    "run": run_index,
                    "case_id": result.case_id,
                    "score": result.score,
                    "error": result.error,
                    "output": (result.output or "")[:400],
                    "detail": _failure_detail(result),
                }
            )
            if len(failure_examples) >= max_failure_examples:
                break
        if len(failure_examples) >= max_failure_examples:
            break

    first = items[0]
    return {
        "label": label,
        "suite": first.suite,
        "system": first.system,
        "runs": len(items),
        "unique_cases": len(case_order),
        "total_attempts": total_attempts,
        "passed_attempts": passed_attempts,
        "pass_rate": round(passed_attempts / total_attempts, 4)
        if total_attempts
        else 0.0,
        "pass_rate_ci95": {
            "low": round(ci_low, 4),
            "high": round(ci_high, 4),
            "method": "wilson",
        },
        "run_pass_rate": _value_summary(run_pass_rates),
        "score": _value_summary([float(result.score) for result in all_results]),
        "run_score": _value_summary(run_scores),
        "judge_average": (
            round(_mean(judge_values_all), 4) if judge_values_all else None
        ),
        "judge_cases": len(judge_values_all),
        "constraint_pass_rate": (
            round(_mean(constraint_values_all), 4)
            if constraint_values_all
            else None
        ),
        "errors": sum(1 for result in all_results if result.error),
        "stability": {
            "always_pass": always_pass,
            "always_fail": always_fail,
            "flaky": flaky,
            "always_pass_count": len(always_pass),
            "always_fail_count": len(always_fail),
            "flaky_count": len(flaky),
        },
        "per_case": per_case,
        "runs_detail": runs,
        "failure_examples": failure_examples,
        "manifest": dict(first.manifest),
        "note": (
            "95% 区间按全部尝试汇总，重复样本存在相关性，仅作描述性不确定性；"
            "运行均值/标准差反映不同采样轮次之间的波动。"
        ),
    }


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.1f}%"


def _fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def render_aggregate(
    aggregate: dict[str, Any], *, title: str | None = None
) -> str:
    """渲染单系统多次采样报告。"""
    label = aggregate.get("label") or aggregate.get("system") or "unknown"
    heading = title or f"多次采样评估 · {label}"
    ci = aggregate.get("pass_rate_ci95") or {}
    run_rate = aggregate.get("run_pass_rate") or {}
    score = aggregate.get("score") or {}
    stability = aggregate.get("stability") or {}

    lines = [
        f"# {heading}",
        "",
        f"- 系统：`{aggregate.get('system', '')}`",
        f"- 套件：`{aggregate.get('suite', '')}`",
        f"- 运行次数：{aggregate.get('runs', 0)}",
        f"- 唯一题目：{aggregate.get('unique_cases', 0)}",
        f"- 总尝试次数：{aggregate.get('total_attempts', 0)}",
        "",
        "## 汇总",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 通过 | {aggregate.get('passed_attempts', 0)} / {aggregate.get('total_attempts', 0)} |",
        f"| 通过率 | {_fmt_pct(aggregate.get('pass_rate'))} |",
        f"| 95% CI（Wilson） | {_fmt_pct(ci.get('low'))} – {_fmt_pct(ci.get('high'))} |",
        f"| 单轮通过率均值 ± 标准差 | {_fmt_pct(run_rate.get('mean'))} ± {_fmt_pct(run_rate.get('std'))} |",
        f"| 单轮通过率范围 | {_fmt_pct(run_rate.get('min'))} – {_fmt_pct(run_rate.get('max'))} |",
        f"| 分数均值 ± 标准差 | {_fmt_num(score.get('mean'))} ± {_fmt_num(score.get('std'))} |",
        f"| Judge 均分 | {_fmt_num(aggregate.get('judge_average'))} |",
        f"| 约束通过率 | {_fmt_pct(aggregate.get('constraint_pass_rate'))} |",
        f"| 错误数 | {aggregate.get('errors', 0)} |",
        "",
        "## 稳定性",
        "",
        "| 类别 | 数量 | 样本 |",
        "| --- | --- | --- |",
        f"| 全部通过 | {stability.get('always_pass_count', 0)} | {', '.join(stability.get('always_pass', [])) or '—'} |",
        f"| 全部失败 | {stability.get('always_fail_count', 0)} | {', '.join(stability.get('always_fail', [])) or '—'} |",
        f"| 波动 | {stability.get('flaky_count', 0)} | {', '.join(stability.get('flaky', [])) or '—'} |",
        "",
        "## 逐题结果",
        "",
        "| 样本 | 类别 | 尝试 | 通过 | 通过率 | 分数均值 | Judge | 约束 | 状态 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in aggregate.get("per_case", []):
        lines.append(
            f"| {item['case_id']} | {item.get('category', '') or '—'} | "
            f"{item['attempts']} | {item['passes']} | "
            f"{_fmt_pct(item['pass_rate'])} | {_fmt_num(item['score_mean'])} | "
            f"{_fmt_num(item.get('judge_average'))} | "
            f"{_fmt_pct(item.get('constraint_pass_rate'))} | {item['status']} |"
        )
    if not aggregate.get("per_case"):
        lines.append("| （无样本） | — | — | — | — | — | — | — | — |")

    lines.extend(["", "## 失败样例", ""])
    failures = aggregate.get("failure_examples", [])
    if not failures:
        lines.append("无失败样例。")
    else:
        for index, failure in enumerate(failures, start=1):
            lines.append(
                f"### {index}. run {failure['run']} · `{failure['case_id']}`"
            )
            if failure.get("error"):
                lines.append(f"- 错误：`{failure['error']}`")
            detail = failure.get("detail") or {}
            if detail:
                lines.append(
                    f"- 细节：`{json.dumps(detail, ensure_ascii=False)[:500]}`"
                )
            output = str(failure.get("output", "") or "").replace("\n", " ")
            if output:
                lines.append(f"- 输出：{output[:300]}")
            lines.append("")

    lines.extend(
        [
            "",
            "## 说明",
            "",
            f"- {aggregate.get('note', '')}",
            "",
        ]
    )
    return "\n".join(lines)


def render_aggregate_summary(aggregates: dict[str, dict[str, Any]]) -> str:
    """多系统汇总表。"""
    lines = [
        "# 多次采样评估汇总",
        "",
        "| 系统 | 运行 | 唯一题目 | 尝试 | 通过率 | 95% CI | 单轮均值 ± 标准差 | 分数均值 | Judge | 约束 | 波动 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in sorted(aggregates):
        item = aggregates[name]
        ci = item.get("pass_rate_ci95") or {}
        run_rate = item.get("run_pass_rate") or {}
        score = item.get("score") or {}
        stability = item.get("stability") or {}
        lines.append(
            f"| {name} | {item.get('runs', 0)} | {item.get('unique_cases', 0)} | "
            f"{item.get('total_attempts', 0)} | {_fmt_pct(item.get('pass_rate'))} | "
            f"{_fmt_pct(ci.get('low'))} – {_fmt_pct(ci.get('high'))} | "
            f"{_fmt_pct(run_rate.get('mean'))} ± {_fmt_pct(run_rate.get('std'))} | "
            f"{_fmt_num(score.get('mean'))} | {_fmt_num(item.get('judge_average'))} | "
            f"{_fmt_pct(item.get('constraint_pass_rate'))} | "
            f"{stability.get('flaky_count', 0)} |"
        )
    if not aggregates:
        lines.append("| （无结果） | — | — | — | — | — | — | — | — | — | — |")
    lines.extend(
        [
            "",
            "> 区间的口径见各系统报告；重复样本存在相关性，区间为描述性。",
            "",
        ]
    )
    return "\n".join(lines)


def write_aggregate(
    aggregate: dict[str, Any], path: str | Path
) -> Path:
    """写出 Markdown 与同名 JSON。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_aggregate(aggregate), encoding="utf-8")
    target.with_suffix(".json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def write_aggregate_summary(
    aggregates: dict[str, dict[str, Any]], path: str | Path
) -> Path:
    """写出多系统汇总 Markdown 与 JSON。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_aggregate_summary(aggregates), encoding="utf-8")
    target.with_suffix(".json").write_text(
        json.dumps(aggregates, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target
