"""BM25 稀疏索引测试：中英文排序、过滤、标题路径与更新。"""

from agentflow.retrieval import BM25Index, Chunk


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    namespace: str = "travel",
    metadata: dict | None = None,
    heading_path: str = "",
    content_hash: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        namespace=namespace,
        text=text,
        heading_path=heading_path,
        metadata=metadata or {},
        content_hash=content_hash or chunk_id,
    )


def test_bm25_chinese_ranking():
    index = BM25Index()
    index.add(
        [
            make_chunk("c1", "故宫开放时间是每天八点三十分到十七点。"),
            make_chunk("c2", "北京烤鸭推荐全聚德。"),
            make_chunk("c3", "故宫门票需要提前预约。"),
        ]
    )
    hits = index.search("故宫开放时间", top_k=2)
    assert hits[0].chunk_id == "c1"
    assert hits[0].score > hits[1].score
    assert len(hits) == 2


def test_bm25_english_ranking():
    index = BM25Index()
    index.add(
        [
            make_chunk("c1", "Beijing weather is sunny in autumn."),
            make_chunk("c2", "Shanghai food is famous."),
        ]
    )
    hits = index.search("Beijing weather", top_k=5)
    assert hits
    assert hits[0].chunk_id == "c1"


def test_bm25_namespace_filter():
    index = BM25Index()
    index.add(
        [
            make_chunk("c1", "故宫开放时间", namespace="travel"),
            make_chunk("c2", "故宫开放时间", namespace="policy"),
        ]
    )
    hits = index.search("故宫", top_k=5, filters={"namespace": "travel"})
    assert [hit.chunk_id for hit in hits] == ["c1"]


def test_bm25_metadata_filter():
    index = BM25Index()
    index.add(
        [
            make_chunk("c1", "故宫开放时间", metadata={"city": "北京"}),
            make_chunk("c2", "故宫开放时间", metadata={"city": "沈阳"}),
        ]
    )
    hits = index.search("故宫", top_k=5, filters={"city": "北京"})
    assert [hit.chunk_id for hit in hits] == ["c1"]


def test_bm25_heading_path_is_indexed():
    index = BM25Index()
    index.add(
        [
            make_chunk("c1", "每天八点三十分到十七点。", heading_path="北京 / 故宫 / 开放时间"),
            make_chunk("c2", "全聚德烤鸭需要排队。", heading_path="北京 / 美食"),
        ]
    )
    hits = index.search("故宫", top_k=5)
    assert hits
    assert hits[0].chunk_id == "c1"


def test_bm25_empty_query_returns_empty():
    index = BM25Index()
    index.add([make_chunk("c1", "故宫开放时间")])
    assert index.search("", top_k=5) == []
    assert index.search("   ", top_k=5) == []


def test_bm25_duplicate_chunk_id_updates_text():
    index = BM25Index()
    index.add([make_chunk("c1", "北京故宫开放时间")])
    index.add([make_chunk("c1", "上海外滩夜景")])
    assert index.search("北京", top_k=5) == []
    hit = index.search("上海", top_k=5)
    assert hit and hit[0].chunk_id == "c1"


def test_bm25_remove_and_clear():
    index = BM25Index()
    index.add([make_chunk("c1", "故宫开放时间"), make_chunk("c2", "北京天气")])
    assert len(index) == 2
    assert index.remove("c1") is True
    assert index.remove("c1") is False
    assert len(index) == 1
    index.clear()
    assert len(index) == 0
