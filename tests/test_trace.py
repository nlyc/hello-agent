"""trace（Tracer 订阅/过滤 + 内置回调 + Agent 集成）测试。"""

import json
from io import StringIO

from agentflow.agents import ReActAgent
from agentflow.trace import (
    ConsoleCallback,
    JSONLFileCallback,
    StatsCallback,
    TraceEvent,
    Tracer,
)
from tests.fakes import ScriptedLLM


def test_tracer_emit_and_subscribe():
    tracer = Tracer(trace_id="tid")
    seen = []
    tracer.subscribe(seen.append)
    tracer.emit("llm", agent_name="a", step=1, data={"x": 1})
    assert len(seen) == 1
    assert seen[0].type == "llm" and seen[0].trace_id == "tid"


def test_tracer_filter_by_type_and_agent():
    tracer = Tracer()
    seen = []
    tracer.subscribe(seen.append, event_type="llm", agent_name="react")
    tracer.emit("llm", agent_name="react")
    tracer.emit("tool", agent_name="react")
    tracer.emit("llm", agent_name="other")
    assert len(seen) == 1


def test_tracer_unsubscribe():
    tracer = Tracer()
    seen = []
    handle = tracer.subscribe(seen.append)
    tracer.unsubscribe(handle)
    tracer.emit("x")
    assert seen == []


def test_tracer_callback_exception_does_not_propagate():
    tracer = Tracer()

    def boom(event):
        raise RuntimeError("boom")

    tracer.subscribe(boom)
    seen = []
    tracer.subscribe(seen.append)
    tracer.emit("x")  # 不应抛出
    assert len(seen) == 1


def test_console_callback():
    out = StringIO()
    cb = ConsoleCallback(out=out)
    cb(TraceEvent(type="llm", agent_name="a", step=1))
    assert "[trace] llm" in out.getvalue()


def test_jsonl_callback(tmp_path):
    p = tmp_path / "trace.jsonl"
    cb = JSONLFileCallback(str(p))
    cb(TraceEvent(type="llm", agent_name="a", data={"output": "hi"}))
    cb.close()
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["type"] == "llm" and rec["agent_name"] == "a"


def test_stats_callback():
    cb = StatsCallback()
    cb(TraceEvent(type="start", data={}))
    cb(TraceEvent(type="llm", data={"tokens": 10}))
    cb(TraceEvent(type="llm", data={"tokens": 5}))
    cb(TraceEvent(type="tool", data={}))
    cb(TraceEvent(type="end", data={}))
    s = cb.summary()
    assert s["llm_calls"] == 2
    assert s["tool_calls"] == 1
    assert s["tokens"] == 15
    assert s["counts"].get("start") == 1


def test_agent_emits_trace():
    tracer = Tracer()
    stats = StatsCallback()
    tracer.subscribe(stats)
    llm = ScriptedLLM(["Thought: 总结\nAction: Finish[答案]"])
    agent = ReActAgent("a", llm, tracer=tracer)
    agent.run("问")
    s = stats.summary()
    assert s["llm_calls"] == 1
    assert s["counts"].get("start") == 1
    assert s["counts"].get("end") == 1
