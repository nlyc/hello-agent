"""Chunker 测试：标题路径、PDF 页码、token 上限、overlap 与稳定 ID。"""

import pytest

from agentflow.core.exceptions import DocumentLoadError, RAGError
from agentflow.retrieval import Chunker, Document
from agentflow.retrieval.chunker import MAX_DOCUMENT_CHUNKS


def make_document(text: str, **kwargs) -> Document:
    """构造测试文档。"""
    return Document(
        doc_id=kwargs.pop("doc_id", "doc-1"),
        namespace=kwargs.pop("namespace", "travel"),
        source_type=kwargs.pop("source_type", "file"),
        source_uri=kwargs.pop("source_uri", "D:/data/guide.md"),
        title=kwargs.pop("title", "北京旅游指南"),
        text=text,
        metadata=kwargs.pop("metadata", {"markdown": True}),
        **kwargs,
    )


def test_heading_path_is_built_from_markdown():
    document = make_document(
        "# 北京旅游\n故宫每天八点三十分开放。\n## 美食\n北京烤鸭推荐全聚德。\n"
    )
    chunks = Chunker().split(document)
    paths = [chunk.heading_path for chunk in chunks]
    assert paths == ["北京旅游", "北京旅游 / 美食"]
    assert "故宫" in chunks[0].text
    assert "全聚德" in chunks[1].text


def test_pdf_pages_are_preserved():
    document = make_document(
        "第一页介绍故宫。\f第二页介绍长城。",
        source_type="pdf",
        metadata={"markdown": False},
    )
    chunks = Chunker().split(document)
    assert [chunk.page for chunk in chunks] == [1, 2]
    assert chunks[0].text == "第一页介绍故宫。"
    assert chunks[1].text == "第二页介绍长城。"


def test_chunk_never_exceeds_max_tokens():
    text = "故宫" * 400  # 800 个 CJK 字符
    chunks = Chunker(chunk_tokens=400, overlap=60).split(make_document(text))
    assert len(chunks) >= 2
    assert all(chunk.token_count <= 512 for chunk in chunks)
    assert all(len(chunk.text) <= 512 for chunk in chunks)


def test_overlap_between_adjacent_chunks():
    # 每个 CJK 字符是一个 token 单元，便于精确断言窗口边界。
    text = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥天地玄黄宇宙洪荒"
    assert len(text) == 30
    chunks = Chunker(chunk_tokens=10, overlap=3).split(make_document(text))
    assert len(chunks) == 4
    assert [len(chunk.text) for chunk in chunks] == [10, 10, 10, 9]
    # 相邻块共享 3 个字符的 overlap。
    assert chunks[0].text[-3:] == chunks[1].text[:3]
    assert chunks[1].text[-3:] == chunks[2].text[:3]


def test_chunk_ids_are_stable_and_unique():
    document = make_document("# 标题\n" + "内容" * 300)
    first = Chunker(chunk_tokens=50, overlap=10).split(document)
    second = Chunker(chunk_tokens=50, overlap=10).split(document)
    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]
    assert len({chunk.chunk_id for chunk in first}) == len(first)


def test_chunk_metadata_and_hash():
    document = make_document("# 北京旅游\n故宫每天八点三十分开放。")
    chunk = Chunker().split(document)[0]
    assert chunk.doc_id == "doc-1"
    assert chunk.namespace == "travel"
    assert chunk.metadata["title"] == "北京旅游指南"
    assert chunk.metadata["doc_type"] == "file"
    assert len(chunk.content_hash) == 64
    assert chunk.token_count > 0


def test_empty_document_returns_no_chunks():
    assert Chunker().split(make_document("   \n\n  ")) == []


def test_too_many_chunks_raise():
    document = make_document("一二三四五六七八九十")
    with pytest.raises(DocumentLoadError, match="chunk 数超过上限"):
        Chunker(chunk_tokens=2, overlap=0, max_chunks=2).split(document)


def test_invalid_parameters_raise():
    with pytest.raises(RAGError):
        Chunker(chunk_tokens=0)
    with pytest.raises(RAGError):
        Chunker(chunk_tokens=10, overlap=10)
    with pytest.raises(RAGError):
        Chunker(max_chunks=0)


def test_max_document_chunks_constant():
    assert MAX_DOCUMENT_CHUNKS == 5000
