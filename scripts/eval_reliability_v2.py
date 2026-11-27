"""v2.1 可靠性评估：重复运行同一基准，聚合通过率、区间与波动。

    py -3.14 scripts/eval_reliability_v2.py --suites all --repeat 3 \
        --temperature 0.7 --out .cache/eval_reliability

每次运行复用 ``eval_online_v2.py`` 的 S0/S1 对比，输出到 ``<out>/run-XX``；
随后按系统聚合为 ``aggregate_<system>.md/json`` 与总表 ``aggregate_summary.md``。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow.evaluation import (
    EvalDataset,
    EvalReport,
    aggregate_reports,
    write_aggregate,
    write_aggregate_summary,
)
from scripts import eval_online_v2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eval"
SUITES = ("tool_calling", "qa", "quality")


def _load_reports(run_dir: Path) -> dict[str, EvalReport]:
    """读取一次运行写出的所有报告 JSON（排除 summary.json）。"""
    reports: dict[str, EvalReport] = {}
    for path in sorted(run_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "cases" not in data:
            continue
        reports[path.stem] = EvalReport.from_dict(data)
    return reports


def _run_args(args, suites: list[str], run_dir: Path, limit: int) -> list[str]:
    values = [
        "--suites",
        ",".join(suites),
        "--limit",
        str(limit),
        "--offset",
        str(args.offset),
        "--agent",
        args.agent,
        "--max-tool-calls",
        str(args.max_tool_calls),
        "--temperature",
        str(args.temperature),
        "--out",
        str(run_dir),
    ]
    if args.category:
        values.extend(["--category", args.category])
    return values


def _dataset_limit() -> int:
    """全量模式的 limit：取三套数据集的最大条数。"""
    return max(
        len(EvalDataset.load(DATA / name).cases)
        for name in (
            "tool_calling.jsonl",
            "travel_qa.jsonl",
            "itinerary_tasks.jsonl",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v2.1 多次采样可靠性评估")
    parser.add_argument(
        "--suites",
        default="all",
        help="逗号分隔：tool_calling,qa,quality 或 all",
    )
    parser.add_argument("--repeat", type=int, default=3, help="重复采样次数")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="每个套件的样本上限；0 表示全部",
    )
    parser.add_argument("--offset", type=int, default=0, help="样本起始偏移")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="作答采样温度；Judge 始终为 0",
    )
    parser.add_argument(
        "--agent",
        default="react",
        choices=("react", "fc"),
        help="工具调用范式",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=2,
        help="工具调用总预算；<=0 表示不限制",
    )
    parser.add_argument("--category", default=None, help="类别过滤")
    parser.add_argument(
        "--out", default=".cache/eval_reliability", help="输出目录"
    )
    args = parser.parse_args(argv)

    if args.repeat < 2:
        parser.error("--repeat 至少为 2；单次运行请直接用 eval_online_v2.py")
    if args.temperature < 0:
        parser.error("--temperature 不能为负数")

    suites = (
        list(SUITES)
        if args.suites == "all"
        else [item.strip() for item in args.suites.split(",") if item.strip()]
    )
    unknown = [item for item in suites if item not in SUITES]
    if unknown:
        parser.error(f"未知套件：{', '.join(unknown)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = args.limit if args.limit > 0 else _dataset_limit()

    reports_by_system: dict[str, list[EvalReport]] = {}
    run_summaries: list[dict[str, Any]] = []
    for run_index in range(1, args.repeat + 1):
        run_dir = out_dir / f"run-{run_index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[run {run_index}/{args.repeat}] suites={','.join(suites)} "
            f"limit={args.limit or 'all'} temperature={args.temperature}"
        )
        code = eval_online_v2.main(_run_args(args, suites, run_dir, limit))
        if code != 0:
            print(f"第 {run_index} 轮评估失败，退出码 {code}")
            return code
        reports = _load_reports(run_dir)
        if not reports:
            print(f"第 {run_index} 轮没有生成报告：{run_dir}")
            return 1
        for name, report in reports.items():
            reports_by_system.setdefault(name, []).append(report)
        run_summaries.append(
            {
                "run": run_index,
                "reports": sorted(reports),
                "summary": str(run_dir / "summary.json"),
            }
        )

    aggregates: dict[str, dict[str, Any]] = {}
    aggregate_paths: dict[str, str] = {}
    for name in sorted(reports_by_system):
        aggregate = aggregate_reports(reports_by_system[name], label=name)
        target = write_aggregate(aggregate, out_dir / f"aggregate_{name}.md")
        aggregates[name] = aggregate
        aggregate_paths[name] = str(target)
        ci = aggregate["pass_rate_ci95"]
        print(
            f"[{name}] pass={aggregate['pass_rate']:.3f} "
            f"CI=[{ci['low']:.3f}, {ci['high']:.3f}] "
            f"flaky={aggregate['stability']['flaky_count']} -> {target}"
        )

    summary_path = write_aggregate_summary(
        aggregates, out_dir / "aggregate_summary.md"
    )
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repeat": args.repeat,
        "temperature": args.temperature,
        "agent": args.agent,
        "suites": suites,
        "limit": args.limit,
        "offset": args.offset,
        "category": args.category,
        "runs": run_summaries,
        "aggregates": {
            name: {
                "pass_rate": item["pass_rate"],
                "pass_rate_ci95": item["pass_rate_ci95"],
                "run_pass_rate": item["run_pass_rate"],
                "flaky_count": item["stability"]["flaky_count"],
                "report": aggregate_paths[name],
            }
            for name, item in aggregates.items()
        },
    }
    (out_dir / "reliability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"summary: {summary_path}")
    print(f"summary: {out_dir / 'reliability_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
