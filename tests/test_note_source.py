"""NoteContextSource 测试：blocker 优先、分区与去重。"""

from datetime import datetime, timezone

from agentflow.notes import NoteContextSource, NoteStore

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_store(tmp_path) -> NoteStore:
    return NoteStore(tmp_path, user_id="tester")


def test_blocker_comes_first_and_dedupes(tmp_path):
    store = make_store(tmp_path)
    store.create("故宫周一闭馆", "需改到周二。", type="blocker", tags=["故宫"])
    store.create("预约门票", "提前三天预约。", type="action", tags=["故宫"])
    store.create("预算 1500", "两人两天。", type="conclusion", tags=["故宫"])
    source = NoteContextSource(store)

    packets = source.gather("故宫", limit=10, now=NOW)
    assert packets[0].metadata["type"] == "blocker"
    assert packets[0].section == "state"
    assert packets[0].relevance_score == 0.9
    assert packets[0].source == "note"
    # 同一篇笔记只出现一次。
    ids = [packet.metadata["note_id"] for packet in packets]
    assert len(ids) == len(set(ids))
    assert any(packet.metadata["type"] == "action" for packet in packets)
    assert any(packet.metadata["type"] == "conclusion" for packet in packets)


def test_limit_zero_and_empty_store(tmp_path):
    store = make_store(tmp_path)
    source = NoteContextSource(store)
    assert source.gather("故宫", limit=0, now=NOW) == []
    assert source.gather("故宫", limit=10, now=NOW) == []


def test_exclude_task_state(tmp_path):
    store = make_store(tmp_path)
    store.create("已定行程", "两天。", type="task_state")
    source = NoteContextSource(store, include_task_state=False)
    assert source.gather("行程", limit=10, now=NOW) == []
    source = NoteContextSource(store, include_task_state=True)
    packets = source.gather("行程", limit=10, now=NOW)
    assert packets and packets[0].metadata["type"] == "task_state"
    assert packets[0].section == "context"


def test_gather_respects_global_limit(tmp_path):
    store = make_store(tmp_path)
    for index in range(6):
        store.create(f"阻塞 {index}", "需要处理", type="blocker", tags=["北京"])
    packets = NoteContextSource(store).gather("北京", limit=2, now=NOW)
    assert len(packets) == 2
