"""S2–S5 组合验证：Notes / MCP / A2A 叠加在 S0–S1 之上。

    py -3.14 scripts/eval_combo_v2.py --suite all --limit 3 --out .cache/eval_combo

- S2：ContextRAG + NoteContextSource（笔记注入）
- S4：ReAct + MCP 展开工具（与 S0 本地工具同题对比）
- S5：A2A 资料员→规划师→审查员 链路 + 质量 Judge（集成烟测，1 例）

S3（TerminalTool）由 `examples.demo_context_engineering` 离线验证，不重复在线调用。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow.agents import ReActAgent
from agentflow.context import (
    ContextBuilder,
    ContextConfig,
    RetrievalContextSource,
)
from agentflow.core.config import AgentConfig, Config
from agentflow.evaluation import (
    EvalDataset,
    ItineraryQualityEvaluator,
    LLMJudge,
    QAEvaluator,
    ToolCallingEvaluator,
    write_report,
)
from agentflow.notes import NoteContextSource, NoteStore
from agentflow.protocols.a2a import A2AClient
from agentflow.protocols.mcp import MCPTool
from agentflow.retrieval.pipeline import RAG_SYSTEM_PROMPT
from agentflow.retrieval.types import RAGAnswer
from agentflow.tools import ToolRegistry
from examples.demo_a2a_travel import build_servers
from scripts.eval_online_v2 import (
    CallableAgent,
    ContextRAG,
    build_llm,
    build_pipeline,
    build_registry,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eval"

_NOTES = (
    ("note-gugong", "故宫周一闭馆", "故宫周一闭馆，需提前预约。", "blocker", ["北京", "故宫"]),
    ("note-yiheyuan", "颐和园平缓路线", "东宫门到长廊一线平缓，不爬坡。", "conclusion", ["北京", "颐和园"]),
    ("note-beijing-rain", "北京雨天室内景点", "北京雨天优先故宫、中国国家博物馆、首都博物馆。", "reference", ["北京", "雨天"]),
    ("note-xiangshan", "香山红叶时间", "十月中旬到十一月初最佳，登顶需爬坡。", "reference", ["北京", "香山"]),
    ("note-waitan", "外滩夜景时间", "晚上七点至九点最佳。", "reference", ["上海", "外滩"]),
    ("note-shanghai-rain", "上海雨天备选", "上海雨天优先上海博物馆与上海科技馆。", "reference", ["上海", "雨天"]),
    ("note-xihu", "西湖环湖骑行", "环湖约十五公里，适合骑行。", "reference", ["杭州", "西湖"]),
    ("note-panda", "大熊猫基地时间", "早上开园时前往最佳。", "reference", ["成都", "熊猫"]),
    ("note-dujiangyan", "都江堰雨天", "雨天山路湿滑，建议改期。", "reference", ["成都", "都江堰"]),
)


def seed_notes(root: Path) -> NoteStore:
    """写入与 travel_qa 对应的演示笔记。"""
    store = NoteStore(root, user_id="eval")
    existing = {note.id for note in store.list()}
    for note_id, title, content, note_type, tags in _NOTES:
        if note_id not in existing:
            store.create(
                title,
                content,
                type=note_type,
                tags=tags,
                note_id=note_id,
            )
    return store


class NotesContextRAG:
    """S2：检索 + 笔记 → ContextBuilder → LLM 作答。"""

    def __init__(self, pipeline: Any, llm: Any, notes: NoteStore) -> None:
        self.pipeline = pipeline
        self.llm = llm
        self.builder = ContextBuilder(
            [RetrievalContextSource(pipeline), NoteContextSource(notes)],
            config=ContextConfig(max_tokens=3000),
            role_prompt=RAG_SYSTEM_PROMPT,
            output_rules="回答末尾用“依据：”列出引用。",
        )

    def answer(self, question: str) -> RAGAnswer:
        hits = self.pipeline.search(question, top_k=5)
        built = self.builder.build(question)
        answer = self.llm.chat(
            [
                {"role": "system", "content": built.prompt_text},
                {
                    "role": "user",
                    "content": f"用户问题：{question}\n请给出简洁、可核验的回答。",
                },
            ],
            temperature=0.0,
        )
        return RAGAnswer(
            question=question,
            answer=str(answer).strip(),
            citations=[hit.citation for hit in hits if hit.citation],
            hits=hits,
            has_answer=bool(hits),
        )


def run_s2(llm, args) -> list[tuple[str, Any]]:
    dataset = EvalDataset.load(DATA / "travel_qa.jsonl", name="qa")
    pipeline = build_pipeline()
    notes = seed_notes(Path(args.out) / "notes")
    s1 = QAEvaluator(
        ContextRAG(pipeline, llm),
        system=f"S1_context_rag:{llm.model}",
    ).run(dataset.cases, limit=args.limit)
    s2 = QAEvaluator(
        NotesContextRAG(pipeline, llm, notes),
        system=f"S2_context_rag_notes:{llm.model}",
    ).run(dataset.cases, limit=args.limit)
    for report in (s1, s2):
        report.manifest["dataset_sha256"] = dataset.sha256
    return [("qa_S1", s1), ("qa_S2_notes", s2)]


def build_mcp_registry() -> tuple[ToolRegistry, MCPTool]:
    """S4：MCP 工具展开后以原名注册，保证与数据集工具名一致。"""
    mcp = MCPTool(
        "travel",
        command=[sys.executable, "-m", "examples.mcp_server_travel"],
        timeout=30.0,
        connect_timeout=10.0,
        tracer=None,
        agent_name="mcp-eval",
    )
    try:
        wrappers = mcp.expand()
    except Exception:
        mcp.close()
        raise
    registry = ToolRegistry()
    for wrapper in wrappers:
        wrapper.name = wrapper.remote_name
        registry.register_tool(wrapper)
    return registry, mcp


def run_s4(llm, args) -> list[tuple[str, Any]]:
    dataset = EvalDataset.load(DATA / "tool_calling.jsonl", name="tool_calling")
    cases = dataset.cases[: args.limit]
    s0_agent = ReActAgent(
        name="local-tools",
        llm=llm,
        tools=build_registry(),
        config=Config(agent=AgentConfig(max_steps=8)),
    )
    s0 = ToolCallingEvaluator(s0_agent).run(
        cases, system=f"S0_local_tools:{llm.model}"
    )
    mcp = None
    try:
        registry, mcp = build_mcp_registry()
        s4_agent = ReActAgent(
            name="mcp-tools",
            llm=llm,
            tools=registry,
            config=Config(agent=AgentConfig(max_steps=8)),
        )
        s4 = ToolCallingEvaluator(s4_agent).run(
            cases, system=f"S4_mcp_tools:{llm.model}"
        )
    finally:
        if mcp is not None:
            mcp.close()
    for report in (s0, s4):
        report.manifest["dataset_sha256"] = dataset.sha256
    return [("tool_calling_S0", s0), ("tool_calling_S4_mcp", s4)]


def run_s5(llm, args) -> list[tuple[str, Any]]:
    dataset = EvalDataset.load(DATA / "itinerary_tasks.jsonl", name="quality")
    cases = dataset.cases[:1]  # 集成烟测：固定北京案例
    research, planner, reviewer = build_servers()
    try:
        urls = [research.start(), planner.start(), reviewer.start()]
        clients = [A2AClient(url, timeout=30) for url in urls]

        def answer(question: str) -> str:
            facts = clients[0].execute_skill("gather_facts", {"input": "北京"})[0].content
            plan = clients[1].execute_skill(
                "plan_itinerary", {"input": facts}
            )[0].content
            review = clients[2].execute_skill(
                "review_plan", {"input": plan}
            )[0].content
            return review

        report = ItineraryQualityEvaluator(
            CallableAgent(answer), judge=LLMJudge(llm)
        ).run(cases)
        report.manifest["dataset_sha256"] = dataset.sha256
        return [("quality_S5_a2a", report)]
    finally:
        for server in (research, planner, reviewer):
            server.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S2–S5 组合验证")
    parser.add_argument(
        "--suite", default="all", choices=("all", "s2", "s4", "s5")
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--out", default=".cache/eval_combo")
    args = parser.parse_args(argv)

    llm, llm_manifest = build_llm()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suites = ["s2", "s4", "s5"] if args.suite == "all" else [args.suite]
    summary: dict[str, Any] = {
        "llm": llm_manifest,
        "limit": args.limit,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reports": {},
    }
    for suite in suites:
        print(f"[{suite}] start")
        if suite == "s2":
            reports = run_s2(llm, args)
        elif suite == "s4":
            reports = run_s4(llm, args)
        else:
            reports = run_s5(llm, args)
        for name, report in reports:
            report.manifest.update(
                {
                    "limit": args.limit,
                    "timestamp": summary["timestamp"],
                }
            )
            target = write_report(
                report, out_dir / f"{name}.md", title=f"v2.0 组合评估 · {name}"
            )
            summary["reports"][name] = {
                "metrics": report.metrics,
                "manifest": report.manifest,
                "report": str(target),
            }
            print(f"[{name}] {json.dumps(report.metrics, ensure_ascii=False)}")
            print(f"[{name}] report: {target}")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"summary: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
