"""v2.0 评估 CLI：加载基准、运行离线基线、输出 Markdown + JSON。

    py -3.14 scripts/eval_v2.py --suite all --limit 5
    py -3.14 scripts/eval_v2.py --suite tool_calling --category irrelevance
    py -3.14 scripts/eval_v2.py --dry-run

默认 ``--system baseline``：
- 工具调用基线「从不调用工具」，用于验证指标与报告管线；
- QA 基线固定拒答；
- 质量基线输出一个满足部分约束的最小行程。
真实系统（ReAct / ContextBuilder / MCP / A2A）在 M9.8 接入。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow.evaluation import (
    EvalDataset,
    ItineraryQualityEvaluator,
    QAEvaluator,
    ToolCallingEvaluator,
    write_report,
)
from agentflow.evaluation.types import EvalReport

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eval"
SUITES = ("tool_calling", "qa", "quality")
NO_ANSWER = "知识库中未找到相关内容。"


class NoToolAgent:
    """工具调用基线：从不调用工具。"""

    tracer = None

    def run(self, question: str) -> str:
        return "我无法调用工具，只能给出通用建议。"


class BadItineraryAgent:
    """质量基线：一个只覆盖部分约束的最小行程。"""

    tracer = None

    def run(self, question: str) -> str:
        return "D1 故宫。D2 颐和园。预算 500 元。"


def build_dataset(suite: str) -> EvalDataset:
    """按套件加载数据集。"""
    mapping = {
        "tool_calling": ("tool_calling.jsonl", "tool_calling"),
        "qa": ("travel_qa.jsonl", "qa"),
        "quality": ("itinerary_tasks.jsonl", "quality"),
    }
    filename, name = mapping[suite]
    return EvalDataset.load(DATA / filename, name=name)


def build_runner(*, system: str = "baseline"):
    """返回 ``runner(suite, limit, category)``，供 CLI 与 EvaluationTool 复用。"""
    if system != "baseline":
        raise ValueError(f"暂不支持的 system：{system}")

    def runner(
        suite: str, limit: int = 5, category: str | None = None
    ) -> EvalReport:
        dataset = build_dataset(suite)
        if suite == "tool_calling":
            report = ToolCallingEvaluator(NoToolAgent()).run(
                dataset.cases,
                category=category,
                limit=limit,
                system="baseline_no_tools",
            )
        elif suite == "qa":
            report = QAEvaluator(
                lambda question: NO_ANSWER, system="baseline_no_answer"
            ).run(dataset.cases, category=category, limit=limit)
        else:
            report = ItineraryQualityEvaluator(BadItineraryAgent()).run(
                dataset.cases, category=category, limit=limit
            )
        report.manifest.update(
            {
                "dataset": str(dataset.path),
                "dataset_sha256": dataset.sha256,
                "limit": limit,
                "category": category or "all",
                "timestamp": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
        )
        return report

    return runner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v2.0 评估基准")
    parser.add_argument(
        "--suite", default="all", choices=("all",) + SUITES, help="基准套件"
    )
    parser.add_argument("--limit", type=int, default=5, help="每个套件的样本上限")
    parser.add_argument("--category", default=None, help="按类别过滤")
    parser.add_argument(
        "--out", default=".cache/eval_v2", help="报告输出目录"
    )
    parser.add_argument("--dry-run", action="store_true", help="只校验数据集")
    parser.add_argument("--system", default="baseline", help="被测系统")
    args = parser.parse_args(argv)

    suites = SUITES if args.suite == "all" else (args.suite,)
    if args.dry_run:
        for suite in suites:
            dataset = build_dataset(suite)
            cases = dataset.filtered(category=args.category, limit=args.limit)
            print(
                f"{suite}: {len(cases)}/{len(dataset.cases)} cases "
                f"sha256={dataset.sha256[:12]}…"
            )
        return 0

    runner = build_runner(system=args.system)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"system": args.system, "limit": args.limit, "reports": {}}
    for suite in suites:
        report = runner(suite, args.limit, args.category)
        target = write_report(
            report, out_dir / f"{suite}.md", title=f"v2.0 评估 · {suite}"
        )
        summary["reports"][suite] = {
            "metrics": report.metrics,
            "manifest": report.manifest,
            "report": str(target),
        }
        print(f"[{suite}] {json.dumps(report.metrics, ensure_ascii=False)}")
        print(f"[{suite}] report: {target}")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"summary: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
