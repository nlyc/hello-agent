"""v2.0 在线对比：S0（v1.1）vs S1（ContextBuilder）。

    py -3.14 scripts/eval_online_v2.py --suites all --limit 3 --out .cache/eval_online

需要项目 `.env` 中配置 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`。
默认 limit=3，避免误触发大额调用；确认结果后再逐步放大。
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

from agentflow.agents import FunctionCallAgent, ReActAgent
from agentflow.context import (
    ContextBuilder,
    ContextConfig,
    RetrievalContextSource,
)
from agentflow.core.config import AgentConfig, Config
from agentflow.core.env import get_env
from agentflow.evaluation import (
    EvalDataset,
    ItineraryQualityEvaluator,
    LLMJudge,
    QAEvaluator,
    ToolCallingEvaluator,
    write_report,
)
from agentflow.llm.openai_compatible import OpenAICompatibleLLM
from agentflow.retrieval import HashingEmbedder
from agentflow.retrieval.pipeline import RAG_SYSTEM_PROMPT
from agentflow.retrieval.types import RAGAnswer
from agentflow.tools import FunctionTool, ToolParameter, ToolRegistry
from examples.mcp_server_travel import get_attraction, get_weather, search_notes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eval"


class CallableAgent:
    """把普通函数包装成 Evaluator 需要的 Agent 形态。"""

    tracer = None

    def __init__(self, func) -> None:
        self.func = func

    def run(self, question: str) -> str:
        return self.func(question)


class ContextRAG:
    """S1：检索命中 → ContextBuilder 六分区 → LLM 作答。"""

    def __init__(self, pipeline: Any, llm: Any) -> None:
        self.pipeline = pipeline
        self.llm = llm
        self.builder = ContextBuilder(
            [RetrievalContextSource(pipeline)],
            config=ContextConfig(max_tokens=3000),
            role_prompt=RAG_SYSTEM_PROMPT,
            output_rules="回答末尾用“依据：”列出引用。",
        )

    def answer(self, question: str) -> RAGAnswer:
        hits = self.pipeline.search(question, top_k=5)
        if not hits:
            return RAGAnswer(
                question=question,
                answer="知识库中未找到相关内容。",
                citations=[],
                hits=[],
                has_answer=False,
            )
        built = self.builder.build(question)
        messages = [
            {"role": "system", "content": built.prompt_text},
            {
                "role": "user",
                "content": f"用户问题：{question}\n请给出简洁、可核验的回答。",
            },
        ]
        answer = self.llm.chat(messages, temperature=0.0)
        return RAGAnswer(
            question=question,
            answer=str(answer).strip(),
            citations=[hit.citation for hit in hits if hit.citation],
            hits=hits,
            has_answer=True,
        )


def build_llm() -> tuple[OpenAICompatibleLLM, dict[str, Any]]:
    """从 .env 构建 LLM；缺失 Key 时明确报错。"""
    config = Config.from_env()
    if not config.llm.api_key:
        raise RuntimeError("未配置 LLM_API_KEY，无法运行在线评估")
    llm = OpenAICompatibleLLM(
        base_url=config.llm.base_url,
        api_key=config.llm.api_key,
        model=config.llm.model,
        temperature=0.0,
        timeout=90.0,
    )
    manifest = {
        "model": config.llm.model,
        "temperature": 0.0,
        "base_url_configured": bool(config.llm.base_url),
    }
    return llm, manifest


def build_judge_llm(main_llm: OpenAICompatibleLLM) -> tuple[Any, dict[str, Any]]:
    """独立 Judge 模型：读取 EVAL_JUDGE_MODEL / BASE_URL / API_KEY。

    未配置时回退主模型，并在 manifest 中标记 ``judge_independent=False``。
    """
    model = get_env("EVAL_JUDGE_MODEL", "")
    if not model:
        return main_llm, {
            "judge_model": main_llm.model,
            "judge_independent": False,
            "judge_note": "EVAL_JUDGE_MODEL 未配置，使用主模型自评",
        }
    base_url = get_env("EVAL_JUDGE_BASE_URL", "") or main_llm.base_url
    api_key = get_env("EVAL_JUDGE_API_KEY", "") or main_llm.api_key
    judge = OpenAICompatibleLLM(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=0.0,
        timeout=90.0,
    )
    return judge, {
        "judge_model": model,
        "judge_independent": model != main_llm.model,
        "judge_note": "",
    }


def build_registry() -> ToolRegistry:
    """S0 工具调用基线：三个本地旅行工具。"""
    registry = ToolRegistry()
    registry.register_tool(
        FunctionTool(
            "get_weather",
            "查询城市天气；只在用户明确询问天气时使用。",
            get_weather,
            [ToolParameter("city", "string", "城市", True)],
        )
    )
    registry.register_tool(
        FunctionTool(
            "get_attraction",
            "推荐城市景点；只在用户要求景点推荐时使用。",
            get_attraction,
            [ToolParameter("city", "string", "城市", True)],
        )
    )
    registry.register_tool(
        FunctionTool(
            "search_notes",
            "检索旅行笔记中的事实（开放时间、闭馆、预约、最佳时间等）；"
            "一次只传一个检索词，不要对同一问题并行多个查询变体。",
            search_notes,
            [ToolParameter("query", "string", "检索词", True)],
        )
    )
    return registry


def build_pipeline() -> Any:
    """复用 M8 评估脚本的内存索引（Hashing + BM25）。"""
    from scripts.eval_rag import _index_pipeline

    return _index_pipeline(HashingEmbedder())


def run_tool_calling(llm, dataset, args) -> list[tuple[str, Any]]:
    agent_cls = FunctionCallAgent if args.agent == "fc" else ReActAgent
    label = "S0_fc" if args.agent == "fc" else "S0_react"
    agent = agent_cls(
        name=f"online-{args.agent}",
        llm=llm,
        tools=build_registry(),
        config=Config(agent=AgentConfig(max_steps=8)),
    )
    report = ToolCallingEvaluator(agent).run(
        dataset.cases,
        category=args.category,
        limit=args.limit,
        system=f"{label}:{llm.model}",
    )
    return [("tool_calling_" + label, report)]


def run_qa(llm, dataset, args) -> list[tuple[str, Any]]:
    pipeline = build_pipeline()
    reports = []
    s0 = QAEvaluator(
        lambda question: pipeline.answer(question, llm=llm),
        system=f"S0_rag:{llm.model}",
    ).run(dataset.cases, limit=args.limit, category=args.category)
    reports.append(("qa_S0", s0))

    s1 = QAEvaluator(
        ContextRAG(pipeline, llm), system=f"S1_context_rag:{llm.model}"
    ).run(dataset.cases, limit=args.limit, category=args.category)
    reports.append(("qa_S1", s1))
    return reports


def run_quality(llm, judge_llm, dataset, args) -> list[tuple[str, Any]]:
    pipeline = build_pipeline()
    judge = LLMJudge(judge_llm, temperature=0.0)
    reports = []

    s0_prompt = (
        "你是旅行规划师。请根据用户要求给出两日行程，"
        "必须写出 D1/D2、预算与同行人约束。"
    )

    def s0_answer(question: str) -> str:
        return llm.chat(
            [
                {"role": "system", "content": s0_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
        )

    reports.append(
        (
            "quality_S0",
            ItineraryQualityEvaluator(
                CallableAgent(s0_answer), judge=judge
            ).run(dataset.cases, limit=args.limit, category=args.category),
        )
    )

    builder = ContextBuilder(
        [RetrievalContextSource(pipeline)],
        config=ContextConfig(max_tokens=3000),
        role_prompt="你是旅行规划师，只基于给定资料回答。",
        output_rules="写出 D1/D2；满足预算与同行人约束；标注依据。",
    )

    def s1_answer(question: str) -> str:
        built = builder.build(question)
        return llm.chat(
            [
                {"role": "system", "content": built.prompt_text},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
        )

    reports.append(
        (
            "quality_S1",
            ItineraryQualityEvaluator(
                CallableAgent(s1_answer), judge=judge
            ).run(dataset.cases, limit=args.limit, category=args.category),
        )
    )
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v2.0 在线对比评估")
    parser.add_argument(
        "--suites",
        default="all",
        help="逗号分隔：tool_calling,qa,quality 或 all",
    )
    parser.add_argument("--limit", type=int, default=3, help="每个系统样本上限")
    parser.add_argument("--category", default=None, help="类别过滤")
    parser.add_argument(
        "--agent",
        default="react",
        choices=("react", "fc"),
        help="工具调用范式：react 文本协议 / fc 原生 Function Calling",
    )
    parser.add_argument("--out", default=".cache/eval_online", help="输出目录")
    args = parser.parse_args(argv)

    suites = (
        ["tool_calling", "qa", "quality"]
        if args.suites == "all"
        else [item.strip() for item in args.suites.split(",") if item.strip()]
    )
    llm, llm_manifest = build_llm()
    judge_llm, judge_manifest = build_judge_llm(llm)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "llm": llm_manifest,
        "judge": judge_manifest,
        "limit": args.limit,
        "category": args.category,
        "agent": args.agent,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reports": {},
    }

    dataset_files = {
        "tool_calling": "tool_calling.jsonl",
        "qa": "travel_qa.jsonl",
        "quality": "itinerary_tasks.jsonl",
    }
    for suite in suites:
        dataset = EvalDataset.load(DATA / dataset_files[suite], name=suite)
        print(f"[{suite}] cases={len(dataset.cases)} limit={args.limit}")
        if suite == "tool_calling":
            reports = run_tool_calling(llm, dataset, args)
        elif suite == "qa":
            reports = run_qa(llm, dataset, args)
        elif suite == "quality":
            reports = run_quality(llm, judge_llm, dataset, args)
        else:
            raise ValueError(f"未知 suite：{suite}")
        for name, report in reports:
            report.manifest.update(
                {
                    "dataset_sha256": dataset.sha256,
                    "limit": args.limit,
                    "category": args.category or "all",
                    "timestamp": summary["timestamp"],
                }
            )
            target = write_report(
                report, out_dir / f"{name}.md", title=f"v2.0 在线评估 · {name}"
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
