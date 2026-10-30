"""memory（MessageHistory + 三种记忆后端 + as_tool）测试。"""

from agentflow.core.message import Message
from agentflow.memory import (
    FileMemory,
    MessageHistory,
    SQLiteMemory,
    ShortTermMemory,
    estimate_tokens,
)


def test_message_history_add_get_copy():
    h = MessageHistory()
    h.add(Message(role="user", content="hi"))
    h.add(Message(role="assistant", content="hello"))
    assert len(h) == 2
    got = h.get()
    got.clear()
    assert len(h) == 2  # 内部未被外部清空


def test_message_history_max_length_trims():
    h = MessageHistory(max_length=3)
    for i in range(5):
        h.add(Message(role="user", content=str(i)))
    assert len(h) == 3
    assert [m.content for m in h.get()] == ["2", "3", "4"]


def test_message_history_truncate_token_budget():
    h = MessageHistory()
    h.add(Message(role="user", content="word " * 50))  # 约 50 token
    h.add(Message(role="assistant", content="short"))  # 约 1 token
    kept = h.truncate(max_tokens=5)
    assert kept[-1].content == "short"
    assert len(kept) == 1  # 那条长历史超预算，被裁掉


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("你好世界") == 4  # 4 个 CJK 字
    assert estimate_tokens("hello world") == 2  # 2 个词


def test_short_term_memory_sliding_window():
    m = ShortTermMemory(max_items=2)
    m.remember("a", "1")
    m.remember("b", "2")
    m.remember("c", "3")
    assert m.recall("") == "c: 3\nb: 2"  # 最近优先，只保留 2 条


def test_short_term_memory_query_filter():
    m = ShortTermMemory()
    m.remember("天气", "北京今天晴")
    m.remember("美食", "烤鸭")
    assert "北京今天晴" in m.recall("天气")
    assert "烤鸭" not in m.recall("天气")


def test_file_memory_persistence(tmp_path):
    p = tmp_path / "mem.json"
    m1 = FileMemory(str(p))
    m1.remember("天气", "北京晴")
    m2 = FileMemory(str(p))  # 新实例，模拟进程重启
    assert "北京晴" in m2.recall("天气")


def test_file_memory_clear(tmp_path):
    p = tmp_path / "mem.json"
    m = FileMemory(str(p))
    m.remember("a", "1")
    m.clear()
    assert FileMemory(str(p)).recall("") == ""


def test_sqlite_memory_persistence(tmp_path):
    p = tmp_path / "mem.db"
    m1 = SQLiteMemory(str(p))
    m1.remember("天气", "北京晴")
    m1.close()
    m2 = SQLiteMemory(str(p))
    assert "北京晴" in m2.recall("天气")
    m2.close()


def test_sqlite_memory_overwrite(tmp_path):
    p = tmp_path / "mem.db"
    m = SQLiteMemory(str(p))
    m.remember("k", "v1")
    m.remember("k", "v2")
    assert m.recall("k") == "k: v2"
    m.close()


def test_memory_as_tool():
    m = ShortTermMemory()
    tool = m.as_tool()
    tool.run({"action": "remember", "key": "天气", "value": "北京晴"})
    out = tool.run({"action": "recall", "query": "天气"})
    assert "北京晴" in out
