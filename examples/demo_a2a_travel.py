"""A2A Demo：资料员 → 规划师 → 审查员 三 Agent 协作（离线）。

    py -3.14 -m examples.demo_a2a_travel

三个 A2AServer 各自暴露一个技能；协调 Agent 通过 A2ATool 把它们当作工具，
按「收集事实 → 生成行程 → 审查」的顺序完成任务。全程只绑定 127.0.0.1。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agentflow.agents import ReActAgent
from agentflow.llm.base import BaseLLM
from agentflow.protocols.a2a import A2AServer, A2ATool
from agentflow.tools import ToolRegistry
from agentflow.trace import Tracer


class CoordinatorLLM(BaseLLM):
    """按 ReAct 轨迹决定下一步调用，无需真实模型即可演示协作。"""

    def __init__(self) -> None:
        self.step = 0

    def chat(self, messages, *, temperature=None, json_mode=False, tools=None) -> str:
        self.step += 1
        observations = [
            message["content"]
            for message in messages
            if message.get("role") == "user"
            and str(message.get("content", "")).startswith("Observation:")
        ]
        last = (
            observations[-1][len("Observation:") :].strip()
            if observations
            else ""
        )
        if self.step == 1:
            return (
                "Thought: 先让资料员收集北京事实\n"
                'Action: research(skill="gather_facts", input="北京")'
            )
        if self.step == 2:
            return (
                "Thought: 根据事实生成两日行程\n"
                f'Action: planner(skill="plan_itinerary", input="{last}")'
            )
        if self.step == 3:
            return (
                "Thought: 让审查员检查行程\n"
                f'Action: reviewer(skill="review_plan", input="{last}")'
            )
        return f"Thought: 协作完成\nAction: Finish[{last}]"


def build_servers() -> tuple[A2AServer, A2AServer, A2AServer]:
    """构建资料 / 规划 / 审查三个 A2A Agent。"""
    research = A2AServer("资料员", "收集旅行事实")

    @research.skill("gather_facts", "收集城市事实")
    def gather_facts(payload):
        city = str(payload.get("input", "北京") or "北京")
        return (
            f"{city}：故宫周一闭馆，需提前预约；"
            "带父母建议东宫门到长廊一线；预算 1500 可行。"
        )

    planner = A2AServer("规划师", "生成两日行程")

    @planner.skill("plan_itinerary", "根据事实生成行程")
    def plan_itinerary(payload):
        facts = str(payload.get("input", "") or "")
        return (
            "D1 故宫（改到周二）→ 颐和园东宫门平缓路线；"
            f"D2 香山或室内备选。依据：{facts}"
        )

    reviewer = A2AServer("审查员", "审查行程可行性")

    @reviewer.skill("review_plan", "审查行程")
    def review_plan(payload):
        plan = str(payload.get("input", "") or "")
        return (
            "审查通过：已避开周一闭馆，路线适合腿脚不便者。"
            f"待办：提前预约故宫。原计划：{plan}"
        )

    return research, planner, reviewer


def main(out_dir: str | Path | None = None) -> int:
    tracer = Tracer()
    events: list[dict] = []

    def on_trace(event) -> None:
        if event.type == "a2a.call":
            events.append(dict(event.data or {}))
            print(
                f"[a2a] {event.data.get('agent')} / {event.data.get('skill')} "
                f"({event.data.get('duration_ms')} ms)"
            )

    tracer.subscribe(on_trace)
    research, planner, reviewer = build_servers()
    urls, tools = [], {}
    try:
        urls = [research.start(), planner.start(), reviewer.start()]
        for server, url, name in zip(
            (research, planner, reviewer), urls, ("research", "planner", "reviewer")
        ):
            tools[name] = A2ATool(
                url, name=name, tracer=tracer, agent_name="coordinator"
            )
            card = tools[name].card
            print(f"[a2a] discovered {card.name}: {[s.name for s in card.skills]}")

        registry = ToolRegistry()
        for tool in tools.values():
            registry.register_tool(tool)

        agent = ReActAgent(
            name="a2a-coordinator",
            llm=CoordinatorLLM(),
            tools=registry,
            tracer=tracer,
        )
        answer = agent.run("为带父母的北京两日游做计划并审查。")
        print(f"[coordinator] {answer}")

        output = Path(out_dir or "demo_output") / "a2a_run.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "cards": [server.card().to_dict() for server in (research, planner, reviewer)],
                    "calls": events,
                    "answer": answer,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[a2a] trace saved: {output}")
        return 0
    finally:
        for server in (research, planner, reviewer):
            server.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A2A 三 Agent 协作 Demo")
    parser.add_argument("--out", default="demo_output", help="报告输出目录")
    args = parser.parse_args()
    raise SystemExit(main(args.out))
