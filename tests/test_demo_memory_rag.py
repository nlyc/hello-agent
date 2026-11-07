"""M8 综合 Demo 测试：离线跑通、轨迹落盘、记忆与 RAG 产物完整。"""

from __future__ import annotations

import json
from pathlib import Path

from examples.demo_travel_memory_rag import (
    DEMO_USER_ID,
    KNOWLEDGE_DOCS,
    knowledge_documents,
    run_demo,
)


def _run(tmp_path: Path) -> dict:
    """在临时目录跑一次 Demo（离线 Hashing 后端）。"""
    return run_demo(tmp_path)


def test_demo_writes_three_artifacts(tmp_path):
    result = _run(tmp_path)

    for name in ("trace.jsonl", "memory.json", "rag_eval.md"):
        assert (tmp_path / name).exists(), name
        assert (tmp_path / name).stat().st_size > 0, name
    assert result["stats"]["chunks"] >= len(KNOWLEDGE_DOCS)


def test_demo_trace_has_memory_and_rag_events(tmp_path):
    _run(tmp_path)
    with open(tmp_path / "trace.jsonl", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]

    types = {event["type"] for event in events}
    assert {"start", "end", "memory.build", "memory.write", "rag.answer"} <= types


def test_demo_memory_is_persisted_and_recallable(tmp_path):
    result = _run(tmp_path)
    payload = json.loads((tmp_path / "memory.json").read_text(encoding="utf-8"))

    assert payload["user_id"] == DEMO_USER_ID
    assert payload["stats"]["long_term"]["memories"] >= 2
    assert payload["stats"]["long_term"]["dimension"] == 512
    assert payload["paradigms"] and all(row["with_memory"] for row in payload["paradigms"])
    assert payload["cross_session"]["memory_context"]
    assert result["cross_session"]["answer"]


def test_three_paradigms_answers_differ_with_memory(tmp_path):
    result = _run(tmp_path)
    rows = {row["paradigm"]: row for row in result["paradigms"]}

    assert set(rows) == {"react", "reflection", "plan_solve"}
    for row in rows.values():
        assert row["without_memory"].strip()
        assert row["with_memory"].strip()
        assert row["without_memory"] != row["with_memory"]


def test_uploads_cover_pdf_docx_and_url(tmp_path):
    result = _run(tmp_path)
    uploads = result["uploads"]

    assert len(uploads) == 3
    assert any(row["source"] == "PDF 上传" for row in uploads)
    assert any(row["source"] == "DOCX 上传" for row in uploads)
    assert any(row["source"].startswith("URL 入库") for row in uploads)
    for row in uploads:
        assert row["answer"].strip()
        assert row["citations"], row["source"]


def test_rag_comparison_uses_citations(tmp_path):
    result = _run(tmp_path)
    rows = result["rag_compare"]

    assert len(rows) == 3
    for row in rows:
        assert row["without_rag"].strip()
        assert row["with_rag"].strip()
        assert "依据：" in row["with_rag"]
        assert row["citations"]


def test_demo_cli_accepts_query_and_ingest(tmp_path):
    extra = tmp_path / "extra.md"
    extra.write_text("# 补充说明\n\n## 开放时间\n\n景山公园早上六点开门。\n", encoding="utf-8")

    result = run_demo(
        tmp_path / "run",
        ingest=str(extra),
        query="景山公园几点开门？",
    )
    report = (tmp_path / "run" / "rag_eval.md").read_text(encoding="utf-8")

    assert result["stats"]["chunks"] >= len(KNOWLEDGE_DOCS) + 1
    assert "## 自定义问题" in report
    assert "景山公园早上六点开门" in report


def test_knowledge_documents_have_explicit_ids():
    documents = knowledge_documents()
    assert len(documents) == len(KNOWLEDGE_DOCS)
    assert len({document.doc_id for document in documents}) == len(documents)
    for document in documents:
        assert document.doc_id.startswith("kb_")
        assert document.text.strip()
