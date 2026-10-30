"""生成三范式轨迹（交付物 D6 的数据源）：离线运行三范式，落盘 JSONL 并打印统计表。

用法：
    py -3.14 scripts/gen_trajectory_report.py

输出：
    docs/trajectory/{react,reflection,plan_solve}.jsonl —— 逐事件 JSONL 轨迹
    控制台 —— 三范式对比表（步数 / LLM 调用 / 工具调用 / 事件序列形态）
"""

from __future__ import annotations

import json
import os
import sys

# 直接以脚本方式运行时，把仓库根目录加进 sys.path
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow.trace import JSONLFileCallback, StatsCallback, Tracer
from examples.demo_e2e import TASK, make_registry, run_plan_solve, run_react, run_reflection

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "docs", "trajectory")


def collect(name: str, run_fn) -> dict:
    tracer = Tracer(trace_id=name)
    events: list = []
    tracer.subscribe(events.append)
    stats = StatsCallback()
    tracer.subscribe(stats)

    os.makedirs(OUT_DIR, exist_ok=True)
    jsonl = JSONLFileCallback(os.path.join(OUT_DIR, f"{name}.jsonl"))
    tracer.subscribe(jsonl)

    answer = run_fn(tracer)
    jsonl.close()

    llm_calls = sum(1 for e in events if e.type == "llm")
    tool_calls = sum(1 for e in events if e.type == "tool")
    steps = [e.step for e in events if e.step]
    sequence = " → ".join(e.type for e in events)

    return {
        "answer": answer,
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "max_step": max(steps) if steps else 0,
        "event_count": len(events),
        "sequence": sequence,
        "stats": stats.summary(),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rows = {
        "react": collect("react", lambda t: run_react(make_registry(), tracer=t)),
        "reflection": collect("reflection", lambda t: run_reflection(tracer=t)),
        "plan_solve": collect("plan_solve", lambda t: run_plan_solve(tracer=t)),
    }

    print("| 范式 | LLM调用 | 工具调用 | 最大step | 事件数 | 事件序列形态 |")
    print("|---|---|---|---|---|---|")
    for name, r in rows.items():
        print(f"| {name} | {r['llm_calls']} | {r['tool_calls']} | {r['max_step']} | {r['event_count']} | {r['sequence']} |")

    print("\n各范式最终答案：")
    for name, r in rows.items():
        print(f"  [{name}] {r['answer']}")

    print(f"\nJSONL 轨迹已落盘至 {OUT_DIR}")


if __name__ == "__main__":
    main()
