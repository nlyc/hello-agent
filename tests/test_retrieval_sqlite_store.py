"""SQLiteVectorStore 测试：写入、检索、持久化、过滤与删除。"""

import pytest

from agentflow.core.exceptions import EmbeddingMismatchError
from agentflow.retrieval import Chunk, HashingEmbedder, SQLiteVectorStore


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    doc_id: str = "doc-1",
    namespace: str = "travel",
    metadata: dict | None = None,
    fingerprint: str = "hashing:char-ngram:512",
) -> Chunk:
    vector = HashingEmbedder().embed_query(text)
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        namespace=namespace,
        text=text,
        metadata=metadata or {},
        content_hash=f"hash-{chunk_id}",
        embedding=vector,
        embedding_fingerprint=fingerprint,
    )


def test_upsert_and_search(tmp_path):
    store = SQLiteVectorStore(tmp_path / "rag.sqlite3")
    try:
        assert store.upsert(
            [
                make_chunk("c1", "故宫开放时间是每天八点三十分到十七点。"),
                make_chunk("c2", "北京烤鸭推荐全聚德。"),
                make_chunk("c3", "上海外滩夜景很漂亮。"),
            ]
        ) == 3
        hits = store.search(HashingEmbedder().embed_query("故宫开放时间"), top_k=2)
        assert hits
        assert hits[0].chunk_id == "c1"
        assert len(hits) == 2
        assert hits[0].citation
    finally:
        store.close()


def test_persistence_across_instances(tmp_path):
    path = tmp_path / "rag.sqlite3"
    store = SQLiteVectorStore(path)
    store.upsert([make_chunk("c1", "故宫开放时间是每天八点三十分到十七点。")])
    store.close()

    reopened = SQLiteVectorStore(path)
    try:
        assert reopened.count() == 1
        hits = reopened.search(HashingEmbedder().embed_query("故宫"), top_k=1)
        assert hits and hits[0].chunk_id == "c1"
    finally:
        reopened.close()


def test_namespace_count_and_filter(tmp_path):
    store = SQLiteVectorStore(tmp_path / "rag.sqlite3")
    try:
        store.upsert(
            [
                make_chunk("c1", "故宫开放时间", namespace="travel"),
                make_chunk("c2", "故宫开放时间", namespace="policy"),
            ]
        )
        assert store.count() == 2
        assert store.count("travel") == 1
        hits = store.search(
            HashingEmbedder().embed_query("故宫"),
            top_k=5,
            filters={"namespace": "travel"},
        )
        assert [hit.chunk_id for hit in hits] == ["c1"]
    finally:
        store.close()


def test_metadata_filter(tmp_path):
    store = SQLiteVectorStore(tmp_path / "rag.sqlite3")
    try:
        store.upsert(
            [
                make_chunk("c1", "故宫开放时间", metadata={"city": "北京"}),
                make_chunk("c2", "故宫开放时间", metadata={"city": "沈阳"}),
            ]
        )
        hits = store.search(
            HashingEmbedder().embed_query("故宫"),
            top_k=5,
            filters={"city": "北京"},
        )
        assert [hit.chunk_id for hit in hits] == ["c1"]
        assert hits[0].metadata["city"] == "北京"
    finally:
        store.close()


def test_update_same_chunk_id(tmp_path):
    store = SQLiteVectorStore(tmp_path / "rag.sqlite3")
    try:
        store.upsert([make_chunk("c1", "北京故宫开放时间")])
        store.upsert([make_chunk("c1", "上海外滩夜景")])
        assert store.count() == 1
        hits = store.search(HashingEmbedder().embed_query("上海"), top_k=1)
        assert hits and hits[0].chunk_id == "c1"
        assert "上海" in hits[0].text
    finally:
        store.close()


def test_delete_and_clear(tmp_path):
    store = SQLiteVectorStore(tmp_path / "rag.sqlite3")
    try:
        store.upsert(
            [
                make_chunk("c1", "故宫开放时间", doc_id="doc-1"),
                make_chunk("c2", "北京烤鸭", doc_id="doc-1"),
                make_chunk("c3", "上海外滩", doc_id="doc-2"),
            ]
        )
        assert store.delete(doc_id="doc-1") == 2
        assert store.count() == 1
        assert store.delete(namespace="travel") == 1
        assert store.count() == 0

        store.upsert([make_chunk("c4", "故宫开放时间")])
        store.clear()
        assert store.count() == 0
    finally:
        store.close()


def test_fingerprint_mismatch_raises(tmp_path):
    store = SQLiteVectorStore(tmp_path / "rag.sqlite3")
    try:
        store.upsert([make_chunk("c1", "故宫开放时间", fingerprint="hashing:512")])
        with pytest.raises(EmbeddingMismatchError):
            store.upsert([make_chunk("c2", "北京烤鸭", fingerprint="fastembed:512")])
        assert store.fingerprint("travel") == "hashing:512"
    finally:
        store.close()


def test_query_dimension_mismatch_raises(tmp_path):
    store = SQLiteVectorStore(tmp_path / "rag.sqlite3")
    try:
        store.upsert([make_chunk("c1", "故宫开放时间")])
        with pytest.raises(EmbeddingMismatchError):
            store.search([0.1, 0.2, 0.3], top_k=1)
    finally:
        store.close()


def test_search_empty_store_and_zero_top_k(tmp_path):
    store = SQLiteVectorStore(tmp_path / "rag.sqlite3")
    try:
        assert store.search(HashingEmbedder().embed_query("故宫"), top_k=5) == []
        assert store.search(HashingEmbedder().embed_query("故宫"), top_k=0) == []
    finally:
        store.close()
