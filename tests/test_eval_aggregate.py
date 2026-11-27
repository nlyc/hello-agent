"""多次采样聚合：Wilson 区间、稳定性、失败样例与可靠性 CLI。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentflow.evaluation import (
    EvalReport,
    EvalResult,
    aggregate_reports,
    render_aggregate,
    render_aggregate_summary,
    wilson_interval,
    write_aggregate,
    write_aggregate_summary,
)
from scripts import eval_reliability_v2


def test_wilson_interval_known_values():
    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert high == pytest.approx(0.2775, abs=1e-3)

    low, high = wilson_interval(10, 10)
    assert low == pytest.approx(0.7225, abs=1e-3)
    assert high == 1.0

    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-3)
    assert high == pytest.approx(0.7634, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 0.0)


def _report(run: int, passed: list[bool]) -> EvalReport:
    cases = [
        EvalResult(
            case_id=f"c{index}",
            passed=value,
            score=1.0 if value else 0.0,
            metrics={"category": "simple" if index == 1 else "multiple"},
        )
        for index, value in enumerate(passed, start=1)
    ]
    return EvalReport(
        suite="tool_calling",
        system="fake",
        cases=cases,
        metrics={
            "cases": len(cases),
            "passed": sum(passed),
            "accuracy": sum(passed) / len(cases),
        },
        manifest={"dataset_sha256": "abc", "run": run},
    )


def test_aggregate_reports_stability_and_failures():
    reports = [
        _report(1, [True, False, True]),
        _report(2, [True, False, True]),
        _report(3, [True, False, False]),
    ]
    result = aggregate_reports(reports, label="fake")

    assert result["runs"] == 3
    assert result["unique_cases"] == 3
    assert result["total_attempts"] == 9
    assert result["passed_attempts"] == 5
    assert result["pass_rate"] == pytest.approx(5 / 9, abs=1e-4)
    assert result["pass_rate_ci95"]["low"] < result["pass_rate"]
    assert result["pass_rate_ci95"]["high"] > result["pass_rate"]
    assert result["run_pass_rate"]["mean"] == pytest.approx(5 / 9, abs=1e-4)
    assert result["run_pass_rate"]["min"] == pytest.approx(1 / 3, abs=1e-4)
    assert result["run_pass_rate"]["max"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["stability"]["always_pass"] == ["c1"]
    assert result["stability"]["always_fail"] == ["c2"]
    assert result["stability"]["flaky"] == ["c3"]
    assert len(result["failure_examples"]) == 4
    assert result["failure_examples"][0]["case_id"] == "c2"
    assert result["per_case"][2]["pass_rate"] == pytest.approx(2 / 3, abs=1e-4)


def test_aggregate_quality_metrics_and_render():
    def quality_report(judge: float, constraints: bool) -> EvalReport:
        return EvalReport(
            suite="quality",
            system="fake",
            cases=[
                EvalResult(
                    case_id="it_01",
                    passed=constraints,
                    score=judge,
                    metrics={
                        "constraints_passed": constraints,
                        "judge_average": judge,
                        "constraints": {"passed": constraints, "checks": {}},
                    },
                )
            ],
        )

    result = aggregate_reports(
        [quality_report(4.0, True), quality_report(3.0, False)]
    )
    assert result["judge_average"] == pytest.approx(3.5)
    assert result["constraint_pass_rate"] == pytest.approx(0.5)
    assert result["stability"]["flaky"] == ["it_01"]
    text = render_aggregate(result)
    assert "多次采样评估" in text
    assert "95% CI（Wilson）" in text
    assert "it_01" in text
    assert "flaky" in text
    assert "Judge 均分" in text

    summary = render_aggregate_summary({"fake": result})
    assert "多次采样评估汇总" in summary
    assert "| fake |" in summary


def test_report_roundtrip_and_write(tmp_path):
    report = _report(1, [True, False])
    restored = EvalReport.from_dict(json.loads(json.dumps(report.to_dict())))
    assert restored.suite == report.suite
    assert restored.cases[1].case_id == "c2"
    assert restored.cases[1].passed is False

    aggregate = aggregate_reports([report], label="fake")
    target = write_aggregate(aggregate, tmp_path / "aggregate_fake.md")
    assert target.exists()
    assert (tmp_path / "aggregate_fake.json").exists()
    assert "多次采样评估" in target.read_text(encoding="utf-8")

    summary_path = write_aggregate_summary(
        {"fake": aggregate}, tmp_path / "aggregate_summary.md"
    )
    assert summary_path.exists()
    assert (tmp_path / "aggregate_summary.json").exists()


def test_reliability_cli_repeats_and_aggregates(tmp_path, monkeypatch):
    """用假 CLI 验证重复运行、报告发现与聚合落盘。"""

    def make_report(run: int) -> dict:
        return EvalReport(
            suite="tool_calling",
            system=f"S0_react:run{run}",
            cases=[
                EvalResult(
                    case_id="tc_simple_01",
                    passed=run == 1,
                    score=1.0 if run == 1 else 0.0,
                    metrics={"category": "simple"},
                )
            ],
            metrics={"cases": 1, "passed": 1 if run == 1 else 0},
            manifest={"dataset_sha256": "abc"},
        ).to_dict()

    def fake_main(argv):
        out_dir = Path(argv[argv.index("--out") + 1])
        run_index = int(out_dir.name.split("-")[-1])
        report = make_report(run_index)
        (out_dir / "tool_calling_S0_react.json").write_text(
            json.dumps(report, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "summary.json").write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        eval_reliability_v2.eval_online_v2, "main", fake_main
    )
    code = eval_reliability_v2.main(
        [
            "--suites",
            "tool_calling",
            "--repeat",
            "2",
            "--limit",
            "1",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert (tmp_path / "run-01" / "tool_calling_S0_react.json").exists()
    assert (tmp_path / "run-02" / "tool_calling_S0_react.json").exists()
    assert (tmp_path / "aggregate_tool_calling_S0_react.md").exists()
    assert (tmp_path / "aggregate_summary.md").exists()

    summary = json.loads(
        (tmp_path / "reliability_summary.json").read_text(encoding="utf-8")
    )
    assert summary["repeat"] == 2
    aggregate = summary["aggregates"]["tool_calling_S0_react"]
    assert aggregate["pass_rate"] == pytest.approx(0.5)
    assert aggregate["flaky_count"] == 1
