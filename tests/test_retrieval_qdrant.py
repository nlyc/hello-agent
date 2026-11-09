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


class _FakePoint:
    """Qdrant 返回记录的最小替身。"""

    def __init__(self, point_id: str, payload: dict) -> None:
        self.id = point_id
        self.payload = payload


class _FakeClient:
    """只实现 list_chunks 需要的接口，验证分页与过滤，不联网。"""

    def __init__(self, points: list[_FakePoint]) -> None:
        self.points = points
        self.scroll_calls: list[dict] = []

    def collection_exists(self, collection: str) -> bool:
        return True

    def scroll(self, *, collection_name, scroll_filter, limit, offset, **kwargs):
        self.scroll_calls.append(
            {
                "collection": collection_name,
                "filter": scroll_filter,
                "limit": limit,
                "offset": offset,
            }
        )
        start = int(offset or 0)
        page = self.points[start : start + limit]
        next_offset = start + limit
        if next_offset >= len(self.points):
            next_offset = None
        return page, next_offset


def test_list_chunks_scrolls_all_pages():
    pytest.importorskip("qdrant_client")  # 过滤器需要 qdrant_client.models
    store = QdrantVectorStore("http://127.0.0.1:6333", collection="travel")
    points = [
        _FakePoint(f"id-{index}", {"text": f"文本 {index}", "namespace": "travel"})
        for index in range(300)
    ]
    client = _FakeClient(points)
    store._client = client

    chunks = store.list_chunks("travel")

    assert len(chunks) == 300
    assert chunks[0].text == "文本 0"
    assert chunks[-1].text == "文本 299"
    assert all(chunk.embedding is None for chunk in chunks)
    assert [call["limit"] for call in client.scroll_calls] == [256, 256]
    assert client.scroll_calls[0]["offset"] is None
    assert client.scroll_calls[1]["offset"] == 256
    assert client.scroll_calls[0]["filter"] is not None


def test_list_chunks_without_namespace_has_no_filter():
    store = QdrantVectorStore("http://127.0.0.1:6333", collection="travel")
    client = _FakeClient([_FakePoint("id-1", {"text": "文本", "namespace": "travel"})])
    store._client = client

    assert [chunk.text for chunk in store.list_chunks()] == ["文本"]
    assert client.scroll_calls[0]["filter"] is None


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
