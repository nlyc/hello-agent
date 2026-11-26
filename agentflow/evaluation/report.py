"""报告渲染：Markdown 汇总 + 原始 JSON。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentflow.evaluation.types import EvalReport


def render_report(report: EvalReport, *, title: str | None = None) -> str:
    """把 EvalReport 渲染成可读 Markdown。"""
    heading = title or f"v2.0 评估报告 · {report.suite}"
    lines = [f"# {heading}", "", f"- 系统：`{report.system}`", f"- 样本数：{len(report.cases)}"]
    for key, value in report.manifest.items():
        lines.append(f"- {key}：`{value}`")
    lines.extend(["", "## 汇总指标", "", "| 指标 | 值 |", "| --- | --- |"])
    for key, value in report.metrics.items():
        if isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## 逐题结果",
            "",
            "| 样本 | 通过 | 分数 | 延迟(ms) | 错误 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for case in report.cases:
        error = (case.error or "").replace("|", "\\|")[:80]
        lines.append(
            f"| {case.case_id} | {case.passed} | {case.score:.3f} | "
            f"{case.latency_ms:.1f} | {error} |"
        )
    if not report.cases:
        lines.append("| （无样本） | - | - | - | - |")
    lines.append("")
    return "\n".join(lines)


def write_report(
    report: EvalReport,
    path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """同时写出 Markdown 与同名 JSON。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report(report, title=title), encoding="utf-8")
    target.with_suffix(".json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target
