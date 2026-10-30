"""端到端演示（D4）测试：四形态跑通 + 工具热插拔 + Memory 持久化 + Trace 落盘。"""

import json
import os

from examples.demo_e2e import demo_hot_plug, demo_memory, make_registry, run_all, run_four_forms


def test_four_forms_run_offline():
    results = run_four_forms(make_registry())
    for label in ("react", "reflection", "plan_solve", "workflow"):
        assert results[label].strip(), label
    assert "晴" in results["react"]


def test_hot_plug_swaps_implementation_in_place():
    registry = make_registry()
    hp = demo_hot_plug(registry)

    assert hp["removed"] is True
    # 同名替换：工具名的集合不变（注册表内部顺序可因重新注册而变化）
    assert sorted(hp["names_before"]) == sorted(hp["names_after"])
    assert "晴" in hp["answer_before"]
    assert "暴雨" in hp["answer_after"]
    assert "暴雨" in hp["execute_now"] and "晴" not in hp["execute_now"]


def test_memory_persists_across_instances(tmp_path):
    path = os.path.join(str(tmp_path), "memory.json")
    mem = demo_memory(path)

    assert "怕暴雨" in mem["recall"]          # 重启后 recall
    assert "故宫打卡" in mem["via_tool"]      # as_tool 桥接
    assert os.path.exists(path)


def test_run_all_writes_trace_and_stats(tmp_path):
    out = run_all(str(tmp_path))

    assert os.path.exists(out["trace_path"])
    with open(out["trace_path"], encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]

    assert events, "轨迹应非空"
    assert any(e["type"] == "tool" for e in events)
    assert out["stats"]["llm_calls"] > 0
    assert out["stats"]["tool_calls"] >= 2
