"""MemoryTool 测试：P0 六个动作与 ToolRegistry 注册执行。"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agentflow.memory import LongTermMemory, MemoryManager
from agentflow.retrieval import HashingEmbedder
from agentflow.tools import MemoryTool, ToolRegistry


@pytest.fixture()
def tool(tmp_path: Path):
    """构建绑定默认记忆管理器的工具。"""
    store = LongTermMemory(HashingEmbedder(), tmp_path / "memory.sqlite3")
    manager = MemoryManager(HashingEmbedder(), store)
    try:
        yield MemoryTool(manager)
    finally:
        store.close()


def _extract_id(output: str) -> str:
    """从 add 返回文本中提取 memory_id。"""
    match = re.search(r"id=([0-9a-f]{32})", output)
    assert match is not None, output
    return match.group(1)


def test_add_action(tool: MemoryTool):
    output = tool.run({"action": "add", "content": "用户偏好自然风光"})
    assert output.startswith("已记住")
    assert tool.manager.long_term.count() == 1


def test_search_action(tool: MemoryTool):
    tool.run({"action": "add", "content": "用户偏好自然风光"})
    output = tool.run({"action": "search", "query": "自然风光"})
    assert "自然风光" in output
    assert "score=" in output


def test_stats_action(tool: MemoryTool):
    tool.run({"action": "add", "content": "用户偏好自然风光"})
    output = tool.run({"action": "stats"})
    assert "long_term=1" in output
    assert "user=default" in output


def test_update_action(tool: MemoryTool):
    memory_id = _extract_id(
        tool.run({"action": "add", "content": "用户偏好自然风光"})
    )
    output = tool.run(
        {
            "action": "update",
            "memory_id": memory_id,
            "content": "用户偏好人文历史",
            "importance": 0.9,
        }
    )
    assert output == f"已更新（id={memory_id}）"
    hit = tool.manager.long_term.get(memory_id)
    assert hit is not None and hit.content == "用户偏好人文历史"


def test_remove_action(tool: MemoryTool):
    memory_id = _extract_id(
        tool.run({"action": "add", "content": "用户偏好自然风光"})
    )
    assert tool.run({"action": "remove", "memory_id": memory_id}).startswith("已删除")
    assert tool.manager.long_term.get(memory_id) is None
    assert tool.run({"action": "remove", "memory_id": memory_id}).startswith("错误")


def test_clear_all_action(tool: MemoryTool):
    tool.run({"action": "add", "content": "用户偏好自然风光"})
    output = tool.run({"action": "clear_all"})
    assert output.startswith("已清空")
    assert tool.manager.long_term.count() == 0
    assert len(tool.manager.working) == 0


def test_invalid_arguments(tool: MemoryTool):
    assert tool.run({"action": "unknown"}).startswith("错误")
    assert tool.run({"action": "add"}).startswith("错误")
    assert tool.run({"action": "search"}).startswith("错误")
    assert tool.run({"action": "update", "memory_id": "x"}).startswith("错误")
    assert tool.run({"action": "add", "content": "内容", "importance": "高"}).startswith(
        "错误"
    )
    assert tool.run(
        {"action": "add", "content": "内容", "metadata": "not-json"}
    ).startswith("错误")


def test_metadata_accepts_json_string(tool: MemoryTool):
    memory_id = _extract_id(
        tool.run(
            {
                "action": "add",
                "content": "用户偏好自然风光",
                "metadata": '{"kind": "preference"}',
            }
        )
    )
    hit = tool.manager.long_term.get(memory_id)
    assert hit is not None and hit.metadata == {"kind": "preference"}


def test_registry_registration_and_execution(tool: MemoryTool):
    registry = ToolRegistry()
    registry.register_tool(tool)
    assert "memory" in registry.names()
    output = registry.execute("memory", {"action": "add", "content": "用户偏好自然风光"})
    assert output.startswith("已记住")
    schema = registry.to_openai_schema()[0]
    assert schema["function"]["name"] == "memory"
    assert "action" in schema["function"]["parameters"]["required"]


def test_parameters_are_self_described(tool: MemoryTool):
    names = [param.name for param in tool.get_parameters()]
    assert names[:3] == ["action", "content", "query"]
    assert tool.validate_parameters({"action": "stats"}) == []
    assert tool.validate_parameters({}) == ["action"]
