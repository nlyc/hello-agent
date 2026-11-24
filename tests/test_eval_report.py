"""数据集、报告、EvaluationTool 与 CLI 基线测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentflow.evaluation import (
    EvalDataset,
    EvaluationTool,
    EvalCase,
    EvalReport,
    EvalResult,
    render_report,
    sha256_file,
    write_report,
)
from scripts.eval_v2 import (
    BadItineraryAgent,
    NoToolAgent,
    build_dataset,
    build_runner,
    main as eval_main,
)

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_load_and_filter():
    dataset = EvalDataset.load(ROOT / "data/eval/tool_calling.jsonl")
    assert dataset.name == "tool_calling"
    assert len(dataset.cases) == 12
    assert dataset.sha256 == sha256_file(ROOT / "data/eval/tool_calling.jsonl")
    simple = dataset.filtered(category="simple")
    assert len(simple) == 4
    assert all(case.category == "simple" for case in simple)
    assert len(dataset.filtered(limit=3)) == 3

    qa = EvalDataset.load(ROOT / "data/eval/travel_qa.jsonl", name="qa")
    assert qa.cases[0].ground_truth == "周一闭馆"
    assert qa.cases[0].metadata["city"] == "北京"


def test_report_render_and_write(tmp_path):
    report = EvalReport(
        suite="demo",
        system="fake",
        cases=[
            EvalResult(case_id="c1", passed=True, score=1.0, latency_ms=1.2),
            EvalResult(case_id="c2", passed=False, score=0.0, error="boom"),
        ],
        metrics={"cases": 2, "passed": 1, "accuracy": 0.5},
        manifest={"dataset_sha256": "abc"},
    )
    text = render_report(report)
    assert "# v2.0 评估报告 · demo" in text
    assert "| accuracy | 0.5 |" in text
    assert "boom" in text

    target = write_report(report, tmp_path / "demo.md")
    assert target.read_text(encoding="utf-8").startswith("# v2.0")
    data = json.loads((tmp_path / "demo.json").read_text(encoding="utf-8"))
    assert data["metrics"]["accuracy"] == 0.5


def test_evaluation_tool_actions():
    seen: list[tuple] = []

    def runner(suite, limit, category):
        seen.append((suite, limit, category))
        return EvalReport(
            suite=suite,
            system="fake",
            cases=[],
            metrics={"cases": 0},
            manifest={},
        )

    tool = EvaluationTool(runner)
    assert "tool_calling" in tool.run({"action": "list"})
    assert "cases" in tool.run(
        {"action": "run", "suite": "tool_calling", "limit": 3}
    )
    text = tool.run({"action": "report", "suite": "qa", "limit": 1})
    assert "# v2.0 评估报告 · qa" in text
    assert seen == [
        ("tool_calling", 3, None),
        ("qa", 1, None),
    ]
    assert tool.run({"action": "run", "suite": "missing"}).startswith("错误:")


def test_baseline_runner_and_cli(tmp_path):
    report = build_runner()("tool_calling", 5, None)
    assert report.metrics["cases"] == 5
    # 基线从不调用工具：前 5 条包含 simple + multiple，irrelevance 尚未覆盖。
    assert report.metrics["accuracy"] < 0.5
    assert report.manifest["dataset_sha256"]

    exit_code = eval_main(
        [
            "--suite",
            "all",
            "--limit",
            "2",
            "--out",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["reports"]) == {"tool_calling", "qa", "quality"}
    assert (tmp_path / "tool_calling.md").exists()
    assert (tmp_path / "qa.md").exists()
    assert (tmp_path / "quality.md").exists()


def test_cli_dry_run(tmp_path):
    assert eval_main(["--dry-run", "--limit", "3"]) == 0


def test_baseline_agents_are_honest():
    assert "无法调用工具" in NoToolAgent().run("北京天气")
    assert "D1" in BadItineraryAgent().run("北京两日游")
    assert build_dataset("qa").name == "qa"
    assert build_dataset("quality").name == "quality"
    with pytest.raises(KeyError):
        build_dataset("missing")
