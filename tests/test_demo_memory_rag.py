"""M8 综合 Demo 测试：离线跑通、轨迹落盘、记忆与 RAG 产物完整。"""

from __future__ import annotations

import json
from pathlib import Path

from examples.demo_travel_memory_rag import (
    DEMO_USER_ID,
    KNOWLEDGE_DOCS,
    knowledge_documents,
    main,
    run_demo,
)
import examples.demo_travel_memory_rag as demo_module


class _FakeAnswerLLM:
    """只用于额外问答的在线 LLM 替身。"""

    def chat(self, messages, **kwargs):
        return "在线总结：北京雨天优先选择故宫博物院等室内景点。"


class _FakeWebAnswerLLM:
    """模拟在线 LLM 依据网页搜索资料作答。"""

    def chat(self, messages, **kwargs):
        return "南昌今天小雨，18℃，适合室内景点。"


class _FakeSearchTool:
    """联网搜索替身，返回固定摘要。"""

    def run_with_backend(self, parameters):
        return (
            "[1] 南昌天气 — https://example.com/weather\n"
            "    南昌今天小雨，18℃，适合室内景点。",
            "fake-search",
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


def test_demo_custom_question_uses_online_llm_and_is_prominent(tmp_path):
    result = run_demo(
        tmp_path / "run",
        query="北京下雨天适合去哪？",
        answer_llm=_FakeAnswerLLM(),
        llm_label="在线 LLM · fake-model",
    )
    report = (tmp_path / "run" / "rag_eval.md").read_text(encoding="utf-8")

    assert result["mode"] == "online"
    assert "- 回答模型：在线 LLM · fake-model" in report
    assert "在线总结：北京雨天优先选择故宫博物院等室内景点。" in report
    assert result["extra_answer"]["answer_source"] == "kb"
    assert result["extra_answer"]["llm"] == "在线 LLM · fake-model"
    # 在线报告只围绕本次问题组织，不再包含固定北京演示章节。
    assert report.startswith("# 智游·山水行：在线 RAG Demo")
    assert "## 问题" in report
    assert "## 回答" in report
    assert "## 检索证据" in report
    assert "## 固定演示章节" not in report
    assert "## 自定义问题" not in report


def test_demo_search_fallback_when_knowledge_misses(tmp_path, monkeypatch):
    monkeypatch.setattr(demo_module, "SearchTool", lambda: _FakeSearchTool())

    result = run_demo(
        tmp_path / "run",
        query="南昌下雨适合去哪？",
        answer_llm=_FakeWebAnswerLLM(),
        llm_label="在线 LLM · fake-model",
        search_fallback=True,
    )
    report = (tmp_path / "run" / "rag_eval.md").read_text(encoding="utf-8")

    assert result["mode"] == "online"
    assert result["extra_answer"]["answer_source"] == "web_search"
    assert result["extra_answer"]["web_search"]["backend"] == "fake-search"
    assert result["extra_answer"]["answer"].startswith("南昌今天小雨")
    assert "## 联网搜索补充" in report
    assert "南昌今天小雨，18℃" in report


def test_demo_cli_prints_report(tmp_path, capsys):
    """命令行应直接打印报告内容，而不只是计数。"""
    code = main(["--out-dir", str(tmp_path / "cli")])
    out = capsys.readouterr().out

    assert code == 0
    assert "## 三范式：无记忆 vs 有记忆" in out
    assert "## 带 / 不带 RAG 的回答对照" in out
    assert "## 跨会话召回" in out
    assert "rag_eval.md" in out and "memory.json" in out


def test_knowledge_documents_have_explicit_ids():
    documents = knowledge_documents()
    assert len(documents) == len(KNOWLEDGE_DOCS)
    assert len({document.doc_id for document in documents}) == len(documents)
    for document in documents:
        assert document.doc_id.startswith("kb_")
        assert document.text.strip()
