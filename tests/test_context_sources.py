"""上下文来源测试：Memory / RAG / History / Static。"""

from datetime import datetime, timezone

from agentflow.context import (
    ContextPacket,
    HistoryContextSource,
    MemoryContextSource,
    RetrievalContextSource,
    StaticContextSource,
    history_packets,
)
from agentflow.core.message import Message
from agentflow.memory.message_history import MessageHistory

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeManager:
    enabled = True

    def __init__(self, entries=None):
        self.entries = entries or []

    def search(self, query, top_k=5):
        return self.entries[:top_k]


class DisabledManager:
    enabled = False

    def search(self, query, top_k=5):
        raise AssertionError("禁用时不应调用 search")


class RecallOnlyManager:
    enabled = True

    def recall(self, query):
        return "旧版记忆：预算 1500"


class FakeHit:
    def __init__(self, text, citation="", score=0.0, doc_id="d1", chunk_id="c1"):
        self.text = text
        self.citation = citation
        self.score = score
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.source = "guide.md"


class FakePipeline:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, top_k=5):
        return self.hits[:top_k]


def test_memory_source_converts_and_clamps():
    manager = FakeManager(
        [
            {
                "content": "带父母，少走路",
                "score": 0.92,
                "metadata": {"created_at": "2026-01-01T00:00:00", "memory_type": "semantic"},
                "source": "long_term",
                "memory_type": "semantic",
            },
            {"content": "预算 1500", "score": 2.0, "metadata": {}},
            {"content": "", "score": 1.0},
        ]
    )
    packets = MemoryContextSource(manager).gather("北京", limit=10, now=NOW)
    assert len(packets) == 2
    assert packets[0].source == "memory"
    assert packets[0].trusted
    assert packets[0].relevance_score == 0.92
    assert packets[1].relevance_score == 1.0
    assert packets[0].timestamp.tzinfo is not None


def test_memory_source_disabled_and_limit():
    assert MemoryContextSource(DisabledManager()).gather("x", limit=5, now=NOW) == []
    manager = FakeManager([{"content": "记忆", "score": 1.0}])
    assert MemoryContextSource(manager).gather("x", limit=0, now=NOW) == []


def test_memory_source_recall_fallback():
    packets = MemoryContextSource(RecallOnlyManager()).gather("预算", limit=5, now=NOW)
    assert len(packets) == 1
    assert packets[0].content == "旧版记忆：预算 1500"
    assert packets[0].metadata["kind"] == "recall"


def test_retrieval_source_prefixes_citation_and_ranks():
    pipeline = FakePipeline(
        [
            FakeHit("故宫周一闭馆。", "故宫参观须知 p.1", score=0.016),
            FakeHit("故宫需要提前预约。", "故宫参观须知 p.2", score=0.012),
        ]
    )
    packets = RetrievalContextSource(pipeline).gather("故宫", limit=5, now=NOW)
    assert len(packets) == 2
    assert packets[0].section == "evidence"
    assert packets[0].content.startswith("【故宫参观须知 p.1】")
    assert packets[0].relevance_score == 1.0
    assert packets[1].relevance_score == 0.5
    assert packets[0].metadata["citation"] == "故宫参观须知 p.1"
    assert packets[0].metadata["raw_score"] == 0.016


def test_retrieval_source_without_pipeline():
    assert RetrievalContextSource(None).gather("x", limit=5, now=NOW) == []


def test_static_source_respects_limit():
    source = StaticContextSource(
        [
            ContextPacket(content="a", relevance_score=1.0, trusted=True),
            ContextPacket(content="b", relevance_score=1.0, trusted=True),
        ]
    )
    assert len(source.gather("x", limit=1, now=NOW)) == 1
    assert source.gather("x", limit=0, now=NOW) == []


def test_history_packets_last_n_and_prefix():
    history = [
        Message(role="user", content=f"历史{i}") for i in range(5)
    ]
    packets = history_packets(history, limit=2, now=NOW)
    assert len(packets) == 2
    assert packets[0].content == "[user] 历史3"
    assert packets[1].content == "[user] 历史4"
    assert packets[0].source == "history"
    assert packets[0].trusted
    assert packets[0].relevance_score == 0.6


def test_history_packets_supports_message_history_and_empty():
    history = MessageHistory()
    history.add(Message(role="assistant", content="你好"))
    packets = history_packets(history, limit=5, now=NOW)
    assert [packet.content for packet in packets] == ["[assistant] 你好"]
    assert history_packets(None, limit=5, now=NOW) == []
    assert history_packets(history, limit=0, now=NOW) == []


def test_history_source_wrapper():
    source = HistoryContextSource([{"role": "user", "content": "问题"}])
    packets = source.gather("x", limit=5, now=NOW)
    assert packets[0].content == "[user] 问题"
