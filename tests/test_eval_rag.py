"""RAG 评估脚本测试：评估集自洽、指标口径稳定、报告可读。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from scripts.eval_rag import (
    corpus_chunks,
    evaluate_no_rag,
    load_cases,
    run_eval,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = REPO_ROOT / "data" / "eval" / "travel_qa.jsonl"


def test_eval_cases_are_complete_and_unique():
    cases = load_cases(CASES_PATH)

    assert len(cases) == 9
    assert len({case["id"] for case in cases}) == 9
    for case in cases:
        assert case["question"].strip()
        assert case["gold_answer"].strip()
        assert case["gold_chunk_ids"], case["id"]


def test_gold_chunks_exist_in_corpus():
    chunk_ids = {chunk.chunk_id for chunk in corpus_chunks()}
    cases = load_cases(CASES_PATH)

    for case in cases:
        for gold_id in case["gold_chunk_ids"]:
            assert gold_id in chunk_ids, f"{case['id']} 的 gold chunk 不存在：{gold_id}"


def test_hashing_backend_metrics():
    result = run_eval("hashing", cases_path=CASES_PATH)
    metrics = result["results"][0]

    assert metrics["backend"] == "hashing"
    assert metrics["recall"]["recall@5"] >= 1.0
    assert metrics["recall"]["recall@3"] >= 0.9
    assert metrics["mrr"] > 0.5
    assert metrics["fact_rate"] >= 0.8
    assert metrics["completeness"] >= 4.0


def test_bm25_backend_still_runs():
    result = run_eval("bm25", cases_path=CASES_PATH)
    metrics = result["results"][0]

    assert metrics["backend"] == "bm25"
    assert metrics["embedding_fingerprint"] == "bm25"
    assert metrics["recall"]["recall@5"] >= 0.9


def test_no_rag_baseline_never_hits_facts():
    cases = load_cases(CASES_PATH)
    baseline = evaluate_no_rag(cases)

    assert baseline["fact_rate"] == 0.0
    assert baseline["citation_rate"] == 0.0
    assert baseline["recall"]["recall@5"] == 0.0


def test_report_contains_required_tables():
    result = run_eval("hashing", cases_path=CASES_PATH)
    markdown = result["markdown"]

    assert "## 召回命中率" in markdown
    assert "| 后端 | Recall@1 | Recall@3 | Recall@5 | MRR |" in markdown
    assert "## 回答层指标（带 RAG vs 不带 RAG）" in markdown
    assert "| no_rag |" in markdown
    assert "故宫周几闭馆？" in markdown


def test_all_backend_skips_fastembed_without_online():
    result = run_eval("all", cases_path=CASES_PATH)

    assert [item["backend"] for item in result["results"]] == ["bm25", "hashing"]
    assert any("fastembed" in item for item in result["skipped"])


@pytest.mark.skipif(
    importlib.util.find_spec("fastembed") is not None,
    reason="fastembed 已安装，无需验证缺失提示",
)
def test_fastembed_backend_reports_missing_dependency():
    with pytest.raises(RuntimeError, match="fastembed"):
        run_eval("fastembed", cases_path=CASES_PATH)
