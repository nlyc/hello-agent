"""端到端演示（交付物 D4）：三种范式 + 低代码工作流 + 工具热插拔 + Memory 持久化 + Trace 落盘。

离线默认（不联网、可复现）：
    py -3.14 -m examples.demo_e2e

在线（走公司中转站真实模型，需先配置环境变量）：
    $env:LLM_API_KEY="..."        # PowerShell
    py -3.14 -m examples.demo_e2e --online

四形态共享同一旅行助手任务；三个范式共享同一 Tracer，输出 JSONL 轨迹与统计，
直接支撑 D6 轨迹对比报告。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

# 直接以脚本方式运行时，把仓库根目录加进 sys.path，保证 `import agentflow` / `import examples`
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow.agents import PlanAndSolveAgent, ReActAgent, ReflectionAgent
from agentflow.memory import FileMemory
from agentflow.tools import ToolParameter, ToolRegistry
from agentflow.trace import JSONLFileCallback, StatsCallback, Tracer
from examples.travel_workflow import TravelDeps, build_travel_graph

TASK = "我想去北京玩一天，天气怎么样？有什么景点推荐？"


class ScriptedLLM:
    """按队列返回预设回复的离线 LLM，记录每次调用，供 Agent 循环 / 工作流使用。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, *, temperature=None, json_mode=False, tools=None):
        self.calls.append({"messages": list(messages), "json_mode": json_mode})
        if not self._responses:
            return ""
        return self._responses.pop(0)


# --- 工具与四种形态 -------------------------------------------------------

def make_registry(weather_fn=None, attraction_fn=None) -> ToolRegistry:
    """注册旅行助手的两把工具（万物皆为工具，走统一 ToolRegistry）。"""
    registry = ToolRegistry()

    def get_weather(city):
        return weather_fn(city) if weather_fn else f"{city}今天晴，20~25℃，适合户外"

    def get_attraction(city, weather):
        if attraction_fn:
            return attraction_fn(city, weather)
        return f"{city}天气{weather}，推荐：故宫、颐和园、长城"

    registry.register_function(
        "get_weather",
        "查询城市当天天气",
        get_weather,
        [ToolParameter("city", "string", "城市名", True)],
    )
    registry.register_function(
        "get_attraction",
        "按当天天气推荐城市景点",
        get_attraction,
        [
            ToolParameter("city", "string", "城市名", True),
            ToolParameter("weather", "string", "当天天气", True),
        ],
    )
    return registry


REACT_SUNNY = [
    'Thought: 先查北京今天天气\nAction: get_weather(city="北京")',
    'Thought: 天气不错，查适合的景点\nAction: get_attraction(city="北京", weather="晴")',
    "Thought: 信息齐了\nAction: Finish[北京今天晴，推荐故宫、颐和园、长城，适合户外游玩。]",
]

REACT_RAINY = [
    'Thought: 查北京今天天气\nAction: get_weather(city="北京")',
    'Thought: 暴雨天，改查室内景点\nAction: get_attraction(city="北京", weather="暴雨")',
    "Thought: 信息齐了\nAction: Finish[北京今天暴雨，建议选择故宫等室内景点，出门备伞。]",
]


def run_react(registry, responses=REACT_SUNNY, tracer=None) -> str:
    llm = ScriptedLLM(responses)
    agent = ReActAgent("react", llm, tools=registry, tracer=tracer, system_prompt="旅行助手")
    return agent.run(TASK)


def run_reflection(llm=None, tracer=None) -> str:
    llm = llm or ScriptedLLM(
        [
            "北京今天天气晴朗，可以去故宫。",
            "可以补充天气对出行的具体影响，并多推荐几个景点。",
            "北京今天天气晴朗，适合户外活动，推荐故宫、颐和园、长城，记得防晒补水。",
            "无需改进，已满足要求。",
        ]
    )
    agent = ReflectionAgent("reflection", llm, tracer=tracer, max_iterations=3, role="旅行规划专家")
    return agent.run(TASK)


def run_plan_solve(llm=None, tracer=None) -> str:
    llm = llm or ScriptedLLM(
        [
            '```python\n["查询北京天气", "根据天气推荐适合的景点"]\n```',
            "北京今天天气晴朗。",
            "晴天推荐故宫、颐和园、长城，适合散步拍照。",
        ]
    )
    agent = PlanAndSolveAgent("plan_solve", llm, tracer=tracer)
    return agent.run(TASK)


def run_workflow(llm=None, tracer=None) -> dict:
    """低代码工作流形态：理解 → 天气 → 景点 → 回答。"""
    if llm is None:
        llm = ScriptedLLM(
            [
                json.dumps(
                    {
                        "thought": "先确认城市与意图，再改写检索词。",
                        "intent": "用户想了解北京天气并获得景点建议。",
                        "weather_query": "北京今天天气",
                        "attraction_query": "北京 适合游玩的景点 推荐",
                    },
                    ensure_ascii=False,
                ),
                "北京今天晴，20~25℃，适合户外。推荐故宫、颐和园、长城，注意防晒补水。",
            ]
        )

    def weather(city):
        return "北京当前天气:晴，气温20~25℃"

    def attraction(city, weather, optimized_query=""):
        return f"故宫、颐和园、长城，适合{weather}天游玩"

    graph = build_travel_graph(TravelDeps(llm=llm, weather_tool=weather, attraction_tool=attraction))
    return graph.invoke({"province": "北京", "city": "北京", "user_query": TASK})


def run_four_forms(registry, tracer=None) -> dict:
    """三种范式 + 工作流四形态跑通，返回各自答案。"""
    return {
        "react": run_react(registry, tracer=tracer),
        "reflection": run_reflection(tracer=tracer),
        "plan_solve": run_plan_solve(tracer=tracer),
        "workflow": run_workflow(tracer=tracer)["final_answer"],
    }


# --- 热插拔 / Memory / Trace ------------------------------------------------

def demo_hot_plug(registry, tracer=None) -> dict:
    """运行中注销旧实现、注册同名新实现：工具名不变，行为即刻改变。"""
    names_before = registry.names()
    answer_before = run_react(registry, responses=REACT_SUNNY, tracer=tracer)

    removed = registry.unregister("get_weather")
    registry.register_function(
        "get_weather",
        "查询城市当天天气",
        lambda city: f"{city}今天暴雨，23~28℃，请备伞",
        [ToolParameter("city", "string", "城市名", True)],
    )
    names_after = registry.names()
    answer_after = run_react(registry, responses=REACT_RAINY, tracer=tracer)

    return {
        "removed": removed,
        "names_before": names_before,
        "names_after": names_after,
        "answer_before": answer_before,
        "answer_after": answer_after,
        "execute_now": registry.execute("get_weather", {"city": "北京"}),
    }


def demo_memory(path) -> dict:
    """FileMemory 落盘 → 新建实例（模拟进程重启）→ 仍能 recall；并演示 as_tool 桥接。"""
    if os.path.exists(path):
        os.remove(path)
    m1 = FileMemory(path)
    m1.remember("用户偏好", "喜欢晴天户外，故宫打卡，怕暴雨")

    m2 = FileMemory(path)  # 模拟重启后的新进程
    recall = m2.recall("暴雨")

    memory_tool = m2.as_tool(name="memory")
    via_tool = memory_tool.run({"action": "recall", "query": "晴天"})
    return {"path": path, "recall": recall, "via_tool": via_tool}


def run_all(out_dir: str) -> dict:
    """一次跑完整套演示，返回结构化结果（供测试与打印复用）。"""
    os.makedirs(out_dir, exist_ok=True)

    registry = make_registry()

    # 全链路 trace：JSONL 落盘 + 统计回调
    tracer = Tracer(trace_id="demo-e2e")
    trace_path = os.path.join(out_dir, "trace.jsonl")
    jsonl_cb = JSONLFileCallback(trace_path)
    stats_cb = StatsCallback()
    tracer.subscribe(jsonl_cb)
    tracer.subscribe(stats_cb)

    results = run_four_forms(registry, tracer)
    hot_plug = demo_hot_plug(registry, tracer)
    memory = demo_memory(os.path.join(out_dir, "memory.json"))

    jsonl_cb.close()

    return {
        "results": results,
        "hot_plug": hot_plug,
        "memory": memory,
        "trace_path": trace_path,
        "stats": stats_cb.summary(),
    }


def build_online_llm():
    """在线模式：公司中转站 OpenAI 兼容网关，三家云端只是不同 model 名。"""
    from agentflow.llm import ModelRegistry, OpenAICompatibleLLM

    registry = ModelRegistry()
    registry.register("deepseek", "deepseek-chat")
    registry.register("glm", "glm-4-flash")
    registry.register("openai", "gpt-4o-mini")
    llm = OpenAICompatibleLLM(registry=registry)
    return llm, registry


def _print_section(title: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="自研智能体框架 v1.0 端到端演示")
    parser.add_argument("--online", action="store_true", help="走公司中转站真实模型（需 LLM_API_KEY 等环境变量）")
    parser.add_argument("--out-dir", default="demo_output", help="输出目录（trace.jsonl / memory.json）")
    args = parser.parse_args(argv)

    out_dir = os.path.join(args.out_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))

    if args.online:
        llm, _ = build_online_llm()
        _print_section(f"在线模式 · 模型={llm.model} · 中转站={llm.base_url or '(默认)'}")
        results = {
            "react": run_react(make_registry()),
            "reflection": run_reflection(llm=llm),
            "plan_solve": run_plan_solve(llm=llm),
            "workflow": run_workflow(llm=llm)["final_answer"],
        }
    else:
        out = run_all(out_dir)
        results = out["results"]

    for label in ("react", "reflection", "plan_solve", "workflow"):
        print(f"\n[{label}] 最终答案：\n{results[label]}")

    if not args.online:
        _render_diagnostics(out["hot_plug"], out["memory"], out)
    return 0


def _render_diagnostics(hot_plug, memory, out) -> None:
    hp = hot_plug
    _print_section("工具热插拔（运行中换实现，工具名不变）")
    print(f"  替换前工具名册: {hp['names_before']}")
    print(f"  注销旧实现成功: {hp['removed']}")
    print(f"  替换后工具名册: {hp['names_after']}")
    print(f"  替换前答案: {hp['answer_before']}")
    print(f"  替换后答案: {hp['answer_after']}")
    print(f"  同一次 execute 现在返回: {hp['execute_now']}")

    _print_section("Memory 持久化（进程重启后仍在）")
    print(f"  记忆文件: {memory['path']}")
    print(f"  重启后 recall('暴雨') → {memory['recall']}")
    print(f"  as_tool() 桥接 recall('晴天') → {memory['via_tool']}")

    _print_section("全链路 Trace")
    print(f"  轨迹文件: {out['trace_path']}")
    print(f"  统计: {out['stats']}")


if __name__ == "__main__":
    raise SystemExit(main())
