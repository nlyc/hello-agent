"""QdrantVectorStore 测试：参数校验、过滤器构造与可选集成测试。

默认测试不联网；设置 ``QDRANT_URL``（可选 ``QDRANT_API_KEY``）后才运行集成用例。
"""

from __future__ import annotations

import importlib.util
import os
import uuid

import pytest

from agentflow.core.exceptions import RAGError
from agentflow.retrieval import Chunk, HashingEmbedder, QdrantVectorStore
from agentflow.retrieval.store_qdrant import _build_filter, _point_id

HAS_QDRANT = importlib.util.find_spec("qdrant_client") is not None


def test_empty_url_is_rejected():
    with pytest.raises(RAGError, match="url"):
        QdrantVectorStore("")


def test_point_id_is_stable_uuid():
    first = _point_id("chunk_abc")
    second = _point_id("chunk_abc")
    assert first == second
    assert uuid.UUID(first)


def test_missing_qdrant_client_raises_rag_error():
    if HAS_QDRANT:
        pytest.skip("当前环境已安装 qdrant-client")
    store = QdrantVectorStore("http://127.0.0.1:6333")
    with pytest.raises(RAGError, match="qdrant-client"):
        store.count()


def test_build_filter():
    pytest.importorskip("qdrant_client")
    assert _build_filter({}) is None
    query_filter = _build_filter(
        {"namespace": "travel", "doc_id": ["d1", "d2"], "city": "北京"}
    )
    assert query_filter is not None
    assert len(query_filter.must) == 3
    keys = {condition.key for condition in query_filter.must}
    assert keys == {"namespace", "doc_id", "metadata.city"}


def _make_chunk(text: str, namespace: str = "travel", doc_id: str = "doc-1") -> Chunk:
    """构造带向量的测试 chunk。"""
    embedder = HashingEmbedder()
    return Chunk(
        chunk_id=f"chunk_{uuid.uuid4().hex[:16]}",
        doc_id=doc_id,
        namespace=namespace,
        text=text,
        content_hash=uuid.uuid4().hex,
        embedding=embedder.embed_query(text),
        embedding_fingerprint=embedder.fingerprint,
    )


@pytest.mark.skipif(not HAS_QDRANT, reason="未安装 qdrant-client")
def test_qdrant_roundtrip():
    url = os.environ.get("QDRANT_URL")
    if not url:
        pytest.skip("未设置 QDRANT_URL，跳过 Qdrant 集成测试")
    collection = f"agentflow_test_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(
        url,
        api_key=os.environ.get("QDRANT_API_KEY"),
        collection=collection,
        dimension=512,
    )
    try:
        chunk = _make_chunk("故宫每天八点三十分开放。")
        assert store.upsert([chunk]) == 1
        assert store.count() == 1
        assert store.fingerprint("travel") == chunk.embedding_fingerprint
        assert chunk.content_hash in store.content_hashes(namespace="travel")

        hits = store.search(
            HashingEmbedder().embed_query("故宫开放时间"),
            top_k=3,
            filters={"namespace": "travel"},
        )
        assert hits and hits[0].chunk_id == chunk.chunk_id

        assert store.delete(doc_id=chunk.doc_id, namespace="travel") == 1
        assert store.count() == 0
    finally:
        try:
            store.clear()
        finally:
            store.close()
