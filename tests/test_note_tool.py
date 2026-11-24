"""NoteTool 测试：七操作、参数解析与错误返回。"""

import json

from agentflow.notes import NoteStore
from agentflow.tools.note_tool import NoteTool


def make_tool(tmp_path) -> tuple[NoteStore, NoteTool]:
    store = NoteStore(tmp_path, user_id="tester")
    return store, NoteTool(store)


def test_create_and_read(tmp_path):
    _, tool = make_tool(tmp_path)
    result = tool.run(
        {
            "action": "create",
            "title": "故宫周一闭馆",
            "content": "改到周二。",
            "type": "blocker",
            "tags": json.dumps(["北京", "故宫"], ensure_ascii=False),
            "metadata": json.dumps({"source": "guide"}, ensure_ascii=False),
        }
    )
    assert result.startswith("已创建笔记")
    note_id = result.split("id=")[1].split(",")[0]
    read = tool.run({"action": "read", "note_id": note_id})
    assert "# 故宫周一闭馆" in read
    assert "type: blocker" in read
    assert "改到周二" in read


def test_update_search_list_summary_delete(tmp_path):
    store, tool = make_tool(tmp_path)
    created = store.create("待办", "预订门票", type="action", tags=["北京"])
    updated = tool.run(
        {
            "action": "update",
            "note_id": created.id,
            "content": "门票已预订",
            "type": "task_state",
        }
    )
    assert "已更新笔记" in updated

    search = tool.run({"action": "search", "query": "门票"})
    assert created.id in search

    listed = tool.run({"action": "list", "type": "task_state"})
    assert "待办" in listed

    summary = tool.run({"action": "summary"})
    assert "笔记统计" in summary
    assert "index_consistent=True" in summary

    deleted = tool.run({"action": "delete", "note_id": created.id})
    assert "已删除笔记" in deleted
    assert tool.run({"action": "read", "note_id": created.id}).startswith("错误:")


def test_tags_comma_and_metadata_object(tmp_path):
    store, tool = make_tool(tmp_path)
    result = tool.run(
        {
            "action": "create",
            "title": "雨天备选",
            "tags": "北京，室内",
            "metadata": {"source": "manual"},
        }
    )
    assert "已创建笔记" in result
    note_id = result.split("id=")[1].split(",")[0]
    note = store.read(note_id)
    assert note.tags == ["北京", "室内"]
    assert note.metadata == {"source": "manual"}


def test_invalid_actions_and_missing_fields(tmp_path):
    _, tool = make_tool(tmp_path)
    assert tool.run({"action": "unknown"}).startswith("错误:")
    assert tool.run({"action": "create", "title": ""}).startswith("错误:")
    assert tool.run({"action": "read"}).startswith("错误:")
    assert tool.run({"action": "update", "note_id": "note-x"}).startswith("错误:")
    assert tool.run({"action": "search"}).startswith("错误:")
    assert tool.run({"action": "delete"}).startswith("错误:")
    assert tool.run(
        {"action": "create", "title": "坏类型", "type": "unknown"}
    ).startswith("错误:")


def test_create_with_explicit_id_and_delete_missing(tmp_path):
    _, tool = make_tool(tmp_path)
    result = tool.run(
        {"action": "create", "title": "固定 ID", "note_id": "note-fixed"}
    )
    assert "note-fixed" in result
    assert tool.run({"action": "delete", "note_id": "note-missing"}).startswith("错误:")
