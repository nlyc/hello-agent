"""RAGPipeline 测试：入库、去重、更新、检索、上下文与回答。"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentflow.core.exceptions import EmbeddingMismatchError
from agentflow.retrieval import (
    BM25Index,
    Chunker,
    HashingEmbedder,
    RAGPipeline,
    SQLiteVectorStore,
)

GUIDE = "# 北京旅游\n故宫每天八点三十分开放。\n## 美食\n北京烤鸭推荐全聚德。\n"
UPDATED = "# 北京旅游\n上海外滩夜景很美。\n## 美食\n小笼包推荐南翔。\n"
NANCHANG = "# 南昌旅行\n## 滕王阁\n滕王阁大部分展馆在室内，雨天可以正常参观。\n"


def _write(path: Path, content: str) -> Path:
    """写入测试文档。"""
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def pipeline():
    """构建使用内存 SQLite 的默认离线管线。"""
    store = SQLiteVectorStore(":memory:")
    bm25 = BM25Index()
    instance = RAGPipeline(HashingEmbedder(), store, bm25)
    try:
        yield instance
    finally:
        store.close()


def test_ingest_and_search(tmp_path: Path, pipeline: RAGPipeline):
    result = pipeline.ingest(_write(tmp_path / "guide.md", GUIDE))
    assert result.chunk_count == 2
    assert result.skipped == 0
    assert result.title == "北京旅游"

    hits = pipeline.search("故宫开放时间", top_k=2)
    assert hits
    assert "故宫" in hits[0].text
    assert hits[0].citation

    stats = pipeline.stats()
    assert stats["chunks"] == 2
    assert stats["bm25_chunks"] == 2


def test_duplicate_ingest_is_skipped(tmp_path: Path, pipeline: RAGPipeline):
    path = _write(tmp_path / "guide.md", GUIDE)
    first = pipeline.ingest(path)
    before = pipeline.vector_store.count()

    second = pipeline.ingest(path)
    assert second.chunk_count == 0
    assert second.skipped == first.chunk_count
    assert second.metadata.get("unchanged") is True
    assert pipeline.vector_store.count() == before
    assert len(pipeline.bm25) == before


def test_update_replaces_old_chunks(tmp_path: Path, pipeline: RAGPipeline):
    path = _write(tmp_path / "guide.md", GUIDE)
    pipeline.ingest(path)
    old_hashes = pipeline.vector_store.content_hashes(namespace="travel")
    old_count = pipeline.vector_store.count()

    path.write_text(UPDATED, encoding="utf-8")
    result = pipeline.ingest(path)

    assert result.chunk_count == 2
    assert pipeline.vector_store.count() == old_count == 2
    current_hashes = pipeline.vector_store.content_hashes(namespace="travel")
    assert old_hashes.isdisjoint(current_hashes)
    assert "故宫" not in "".join(chunk.text for chunk in pipeline.bm25._chunks.values())


def test_cross_document_duplicate_is_skipped(tmp_path: Path, pipeline: RAGPipeline):
    pipeline.ingest(_write(tmp_path / "a.md", GUIDE))
    before = pipeline.vector_store.count()

    result = pipeline.ingest(_write(tmp_path / "b.md", GUIDE))
    assert result.chunk_count == 0
    assert result.skipped == 2
    assert "unchanged" not in result.metadata
    assert pipeline.vector_store.count() == before


def test_namespace_filtering(tmp_path: Path, pipeline: RAGPipeline):
    pipeline.ingest(_write(tmp_path / "policy.md", GUIDE), namespace="policy")
    assert pipeline.search("故宫", top_k=5) == []

    hits = pipeline.search("故宫", top_k=5, filters={"namespace": "policy"})
    assert hits


def test_build_context_contains_citation_and_budget(
    tmp_path: Path, pipeline: RAGPipeline
):
    pipeline.ingest(_write(tmp_path / "guide.md", GUIDE))
    hits = pipeline.search("故宫", top_k=2)
    context = pipeline.build_context(hits, max_tokens=2000)
    assert context.startswith("以下是从知识库检索到的资料")
    assert "<knowledge>" in context and "</knowledge>" in context
    assert "【资料 1｜" in context
    assert "故宫" in context

    assert pipeline.build_context([], max_tokens=2000) == ""
    assert pipeline.build_context(hits, max_tokens=0) == ""
    tiny = pipeline.build_context(hits, max_tokens=10)
    assert tiny == "" or len(tiny) < len(context)


def test_answer_extractive_without_llm(tmp_path: Path, pipeline: RAGPipeline):
    pipeline.ingest(_write(tmp_path / "guide.md", GUIDE))
    answer = pipeline.answer("故宫几点开放？")
    assert answer.has_answer is True
    assert "故宫" in answer.answer
    assert answer.citations


def test_answer_without_hits(pipeline: RAGPipeline):
    answer = pipeline.answer("量子力学是什么？")
    assert answer.has_answer is False
    assert answer.answer == "知识库中未找到相关内容。"
    assert answer.hits == []


def test_answer_refuses_uncovered_city(tmp_path: Path, pipeline: RAGPipeline):
    """问题点名南昌，但知识库只有北京：不能拿北京资料凑答案。"""
    pipeline.ingest(_write(tmp_path / "guide.md", GUIDE))

    answer = pipeline.answer("南昌故宫几点开放？")
    assert answer.has_answer is False
    assert "未收录" in answer.answer
    assert "南昌" in answer.answer
    assert answer.hits == []
    assert answer.citations == []


def test_answer_refuses_uncovered_scenic_spot(tmp_path: Path, pipeline: RAGPipeline):
    """问题点名词表未收录的景点时，也不能用其它城市的资料凑答案。"""
    pipeline.ingest(_write(tmp_path / "guide.md", GUIDE))

    answer = pipeline.answer("云台山故宫几点开放？")
    assert answer.has_answer is False
    assert "未收录" in answer.answer
    assert "云台山" in answer.answer
    assert answer.hits == []


def test_answer_keeps_only_covered_city(tmp_path: Path, pipeline: RAGPipeline):
    pipeline.ingest(_write(tmp_path / "guide.md", GUIDE))
    pipeline.ingest(_write(tmp_path / "nanchang.md", NANCHANG))

    answer = pipeline.answer("南昌滕王阁雨天能去吗？")
    assert answer.has_answer is True
    assert answer.hits
    assert all(
        "南昌" in (hit.text + hit.citation) for hit in answer.hits
    ), [hit.citation for hit in answer.hits]


def test_place_guard_can_be_disabled(tmp_path: Path):
    store = SQLiteVectorStore(":memory:")
    try:
        pipeline = RAGPipeline(
            HashingEmbedder(), store, BM25Index(), place_guard=False
        )
        pipeline.ingest(_write(tmp_path / "guide.md", GUIDE))
        answer = pipeline.answer("南昌故宫几点开放？")
        assert answer.has_answer is True
    finally:
        store.close()


class _FakeLLM:
    """记录提示词的假 LLM。"""

    def __init__(self) -> None:
        self.messages = []

    def chat(self, messages, *, temperature=None, json_mode=False, tools=None) -> str:
        """返回固定回答并记录输入。"""
        self.messages = messages
        return "故宫每天八点三十分开放。依据：资料 1"


def test_answer_with_llm(tmp_path: Path, pipeline: RAGPipeline):
    pipeline.ingest(_write(tmp_path / "guide.md", GUIDE))
    llm = _FakeLLM()
    answer = pipeline.answer("故宫几点开放？", llm=llm)
    assert answer.has_answer is True
    assert "依据" in answer.answer
    system = llm.messages[0]["content"]
    assert "智游·山水行" in system
    assert "故宫" in llm.messages[1]["content"]


def test_fingerprint_mismatch_is_rejected(tmp_path: Path):
    store = SQLiteVectorStore(":memory:")
    try:
        pipeline = RAGPipeline(HashingEmbedder(), store, BM25Index())
        pipeline.ingest(_write(tmp_path / "guide.md", GUIDE))
        other = RAGPipeline(
            HashingEmbedder(dimension=256), store, BM25Index(), chunker=Chunker()
        )
        with pytest.raises(EmbeddingMismatchError):
            other.ingest(_write(tmp_path / "other.md", "# 其它\n上海外滩夜景很美。"))
    finally:
        store.close()


def test_min_score_and_place_guard_read_from_env(monkeypatch):
    monkeypatch.setenv("RAG_MIN_SCORE", "0.42")
    monkeypatch.setenv("RAG_PLACE_GUARD", "false")
    store = SQLiteVectorStore(":memory:")
    try:
        pipeline = RAGPipeline(HashingEmbedder(), store, BM25Index())
        assert pipeline.retriever.min_score == pytest.approx(0.42)
        assert pipeline.place_guard is False
    finally:
        store.close()
