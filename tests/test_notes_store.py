"""NoteStore 测试：CRUD、front matter、索引、检索与隔离。"""

from pathlib import Path

import pytest

from agentflow.core.exceptions import NoteError
from agentflow.notes import NoteStore


def make_store(tmp_path: Path, user_id: str = "tester") -> NoteStore:
    return NoteStore(tmp_path, user_id=user_id)


def test_create_read_roundtrip(tmp_path):
    store = make_store(tmp_path)
    note = store.create(
        "故宫周一闭馆",
        "周一闭馆，需改到周二或周三。",
        type="blocker",
        tags=["北京", "故宫"],
        metadata={"source": "guide.md"},
    )
    assert note.id.startswith("note-")
    assert (store.directory / f"{note.id}.md").exists()

    loaded = store.read(note.id)
    assert loaded.title == "故宫周一闭馆"
    assert loaded.type == "blocker"
    assert loaded.tags == ["北京", "故宫"]
    assert "周一闭馆" in loaded.content
    assert loaded.metadata == {"source": "guide.md"}
    assert loaded.created_at.tzinfo is not None

    text = (store.directory / f"{note.id}.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert 'type: "blocker"' in text
    assert text.rstrip().endswith("周一闭馆，需改到周二或周三。")


def test_duplicate_explicit_id_rejected(tmp_path):
    store = make_store(tmp_path)
    store.create("第一条", note_id="note-fixed")
    with pytest.raises(NoteError):
        store.create("第二条", note_id="note-fixed")


def test_update_and_delete(tmp_path):
    store = make_store(tmp_path)
    note = store.create("计划", "待定", type="action")
    updated = store.update(
        note.id,
        content="已确认：周二去故宫",
        type="task_state",
        tags=["北京"],
        metadata={"day": 2},
    )
    assert updated.type == "task_state"
    assert "周二去故宫" in updated.content
    assert updated.tags == ["北京"]
    assert store.read(note.id).metadata["day"] == 2

    assert store.delete(note.id) is True
    assert store.delete(note.id) is False
    assert not (store.directory / f"{note.id}.md").exists()
    assert note.id not in store._read_index()["notes"]


def test_list_filters_and_limit(tmp_path):
    store = make_store(tmp_path)
    store.create("阻塞一", type="blocker")
    store.create("阻塞二", type="blocker")
    store.create("结论一", type="conclusion")
    blockers = store.list(note_type="blocker")
    assert len(blockers) == 2
    assert all(note.type == "blocker" for note in blockers)
    assert len(store.list(limit=1)) == 1
    tagged = store.create("标签笔记", tags=["雨天"])
    assert [note.id for note in store.list(tags=["雨天"])] == [tagged.id]


def test_search_prefers_title_match(tmp_path):
    store = make_store(tmp_path)
    hit = store.create("故宫开放时间", "周一闭馆。", type="conclusion")
    store.create("上海外滩", "晚上七点至九点最佳。", type="reference")
    results = store.search("故宫 闭馆", limit=5)
    assert results
    assert results[0].id == hit.id
    assert store.search("完全不存在的词", limit=5) == []


def test_summary_counts_and_recent(tmp_path):
    store = make_store(tmp_path)
    store.create("阻塞", type="blocker")
    store.create("结论", type="conclusion")
    summary = store.summary()
    assert summary["total"] == 2
    assert summary["unresolved_blockers"] == 1
    assert summary["types"] == {"blocker": 1, "conclusion": 1}
    assert summary["index_consistent"] is True
    assert len(summary["recent"]) == 2


def test_rebuild_index_and_corrupt_file(tmp_path):
    store = make_store(tmp_path)
    manual = store.directory / "note-manual.md"
    manual.write_text(
        "---\n"
        'id: "note-manual"\n'
        'title: "手工笔记"\n'
        'type: "conclusion"\n'
        'tags: ["手工"]\n'
        'created_at: "2026-01-01T00:00:00+00:00"\n'
        'updated_at: "2026-01-01T00:00:00+00:00"\n'
        "---\n"
        "手工写入的结论\n",
        encoding="utf-8",
    )
    bad = store.directory / "note-bad.md"
    bad.write_text(
        "---\n"
        'id: "note-bad"\n'
        'title: "坏文件"\n'
        'type: "unknown"\n'
        "---\n"
        "坏内容\n",
        encoding="utf-8",
    )
    notes = store.list()
    assert [note.id for note in notes] == ["note-manual"]
    report = store.rebuild_index()
    assert report["notes"] == 1
    assert report["errors"] and report["errors"][0]["file"] == "note-bad.md"
    assert store.list()[0].content == "手工写入的结论"


def test_invalid_id_and_type(tmp_path):
    store = make_store(tmp_path)
    for bad_id in ("", "../escape", "a/b", "note.bad"):
        with pytest.raises(NoteError):
            store.read(bad_id)
    with pytest.raises(NoteError):
        store.create("类型错误", type="unknown")


def test_user_isolation(tmp_path):
    alice = make_store(tmp_path, "alice")
    bob = make_store(tmp_path, "bob")
    note = alice.create("Alice 的笔记")
    assert bob.list() == []
    assert alice.read(note.id).title == "Alice 的笔记"
