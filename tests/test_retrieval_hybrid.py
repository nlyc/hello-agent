"""HybridRetriever 测试：RRF 融合、top_k、去重与过滤。"""

from agentflow.retrieval import (
    BM25Index,
    Chunk,
    HashingEmbedder,
    HybridRetriever,
    SQLiteVectorStore,
)


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    namespace: str = "travel",
    metadata: dict | None = None,
    content_hash: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        namespace=namespace,
        text=text,
        metadata=metadata or {},
        content_hash=content_hash or chunk_id,
        embedding=HashingEmbedder().embed_query(text),
        embedding_fingerprint="hashing:char-ngram:512",
    )


def build_retriever(chunks):
    embedder = HashingEmbedder()
    store = SQLiteVectorStore(":memory:")
    store.upsert(chunks)
    bm25 = BM25Index()
    bm25.add(chunks)
    return HybridRetriever(embedder, store, bm25), store


def test_hybrid_returns_top_k_and_sorted():
    retriever, store = build_retriever(
        [
            make_chunk("c1", "故宫开放时间是每天八点三十分到十七点。"),
            make_chunk("c2", "故宫附近北京烤鸭推荐全聚德。"),
            make_chunk("c3", "上海外滩夜景很漂亮。"),
        ]
    )
    try:
        hits = retriever.search("故宫开放时间", top_k=2)
        assert len(hits) == 2
        assert hits[0].score >= hits[1].score
        assert hits[0].chunk_id == "c1"
    finally:
        store.close()


def test_hybrid_dedup_by_content_hash():
    same_hash = "same-content-hash"
    retriever, store = build_retriever(
        [
            make_chunk("c1", "故宫开放时间", content_hash=same_hash),
            make_chunk("c1-dup", "故宫开放时间", content_hash=same_hash),
            make_chunk("c2", "故宫附近北京烤鸭推荐全聚德。"),
        ]
    )
    try:
        hits = retriever.search("故宫开放时间", top_k=5)
        assert len(hits) == 2
        assert len({hit.content_hash for hit in hits}) == 2
    finally:
        store.close()


def test_hybrid_namespace_filter():
    retriever, store = build_retriever(
        [
            make_chunk("c1", "故宫开放时间", namespace="travel"),
            make_chunk("c2", "故宫开放时间", namespace="policy"),
        ]
    )
    try:
        hits = retriever.search("故宫", top_k=5, filters={"namespace": "travel"})
        assert [hit.chunk_id for hit in hits] == ["c1"]
    finally:
        store.close()


def test_hybrid_is_stable():
    retriever, store = build_retriever(
        [
            make_chunk("c1", "故宫开放时间"),
            make_chunk("c2", "北京烤鸭推荐"),
            make_chunk("c3", "上海外滩夜景"),
        ]
    )
    try:
        first = retriever.search("北京 故宫", top_k=3)
        second = retriever.search("北京 故宫", top_k=3)
        assert [(hit.chunk_id, hit.score) for hit in first] == [
            (hit.chunk_id, hit.score) for hit in second
        ]
    finally:
        store.close()


def test_hybrid_empty_query_and_zero_top_k():
    retriever, store = build_retriever([make_chunk("c1", "故宫开放时间")])
    try:
        assert retriever.search("", top_k=5) == []
        assert retriever.search("故宫", top_k=0) == []
    finally:
        store.close()


def test_min_score_filters_weak_dense_hits():
    embedder = HashingEmbedder()
    store = SQLiteVectorStore(":memory:")
    store.upsert([make_chunk("c1", "故宫开放时间是每天八点三十分到十七点。")])
    try:
        # BM25 为空，只剩稠密召回；阈值高于实际余弦分（约 0.17）。
        retriever = HybridRetriever(embedder, store, BM25Index(), min_score=0.99)
        assert retriever.search("故宫开放时间", top_k=5) == []
    finally:
        store.close()


def test_min_score_does_not_drop_bm25_hits():
    embedder = HashingEmbedder()
    store = SQLiteVectorStore(":memory:")
    chunks = [make_chunk("c1", "故宫开放时间")]
    store.upsert(chunks)
    bm25 = BM25Index()
    bm25.add(chunks)
    try:
        retriever = HybridRetriever(embedder, store, bm25, min_score=0.99)
        hits = retriever.search("故宫开放时间", top_k=5)
        assert [hit.chunk_id for hit in hits] == ["c1"]
    finally:
        store.close()
