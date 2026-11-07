"""RAG 评估脚本：召回命中率与「带 / 不带 RAG」的回答对照。

默认完全离线：BM25 单路 + Hashing 稠密 + BM25 混合两行结果，不下载模型。

    py -3.14 scripts/eval_rag.py --out docs/m8-memory-rag-eval.md

可选语义后端（需安装 extras，首次会下载模型）：

    py -3.14 -m pip install -e ".[embed]"
    py -3.14 scripts/eval_rag.py --backend fastembed --online --out docs/m8-memory-rag-eval.md

指标：
    Recall@k —— top-k 是否命中任一 gold chunk；
    MRR      —— 首个命中排名的倒数均值；
    事实命中率 / 引用覆盖率 / 拒答率 / 平均完整度 —— 回答层指标。

评估集 ``data/eval/travel_qa.jsonl`` 里的 ``gold_chunk_ids`` 与语料文本、
默认 ``Chunker`` 参数绑定；改动语料后必须重新生成。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow.retrieval import (
    BM25Index,
    Chunker,
    Document,
    FastEmbedEmbedder,
    HashingEmbedder,
    RAGPipeline,
    SQLiteVectorStore,
    citation_for_chunk,
)
from examples.demo_travel_memory_rag import KNOWLEDGE_DOCS, knowledge_documents

DEFAULT_CASES = Path("data/eval/travel_qa.jsonl")
NO_ANSWER = "知识库中未找到相关内容。"
NO_RAG_ANSWER = "推荐当地热门景点，建议出发前查询开放时间并关注天气。"
K_VALUES = (1, 3, 5)
NAMESPACE = "travel"


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 评估集，校验必需字段。"""
    cases: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            missing = {"question", "gold_chunk_ids", "gold_answer"} - set(item)
            if missing:
                raise ValueError(f"{path}:{line_no} 缺少字段 {sorted(missing)}")
            item["id"] = item.get("id") or f"qa_{len(cases) + 1:02d}"
            item["city"] = item.get("city", "")
            cases.append(item)
    if not cases:
        raise ValueError(f"{path} 中没有任何评估样本")
    return cases


def build_corpus() -> list[Document]:
    """返回评估用文档（与 Demo 共用同一份语料）。"""
    return knowledge_documents()


def corpus_chunks() -> list[Any]:
    """按默认 Chunker 切出语料 chunk，用于 BM25 单路与引用映射。"""
    chunker = Chunker()
    chunks: list[Any] = []
    for document in build_corpus():
        chunks.extend(chunker.split(document))
    return chunks


def _index_pipeline(embedder) -> RAGPipeline:
    """为指定 Embedder 建一套内存索引（不做磁盘写入）。"""
    pipeline = RAGPipeline(
        embedder, SQLiteVectorStore(":memory:"), BM25Index(), namespace=NAMESPACE
    )
    for document in build_corpus():
        chunks = pipeline.chunker.split(document)
        if not chunks:
            continue
        vectors = embedder.embed([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector
            chunk.embedding_fingerprint = embedder.fingerprint
        pipeline.vector_store.upsert(chunks)
        pipeline.bm25.add(chunks)
    return pipeline


def _bm25_only() -> BM25Index:
    """只建 BM25 索引，作为无稠密向量的对照。"""
    index = BM25Index()
    index.add(corpus_chunks())
    return index


def _search(
    backend: str, pipeline: RAGPipeline | None, bm25: BM25Index | None, query: str, top_k: int
) -> list[Any]:
    """按后端取回 top_k 命中。"""
    if backend == "bm25":
        assert bm25 is not None
        return bm25.search(query, top_k=top_k)
    assert pipeline is not None
    return pipeline.search(query, top_k=top_k)


def _bigram_overlap(query: str, text: str) -> int:
    """字符二元组重合度，用于模拟「读完全部资料后挑选证据」。"""
    query_grams = {query[i : i + 2] for i in range(len(query) - 1)}
    text_grams = {text[i : i + 2] for i in range(len(text) - 1)}
    return len(query_grams & text_grams)


def _answer_with_rag(hits: Sequence[Any], question: str) -> tuple[str, str]:
    """从 top-k 命中里选最相关证据作答，返回 (回答, 使用的引用)。"""
    if not hits:
        return NO_ANSWER, ""
    best = max(hits, key=lambda hit: _bigram_overlap(question, hit.text))
    citation = best.citation or ""
    body = best.text.strip()
    return (f"{body}\n依据：{citation}" if citation else body), citation


def evaluate(
    backend: str,
    cases: Sequence[dict[str, Any]],
    *,
    embedder: Any = None,
    pipeline: RAGPipeline | None = None,
    bm25: BM25Index | None = None,
    citation_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """对一个后端跑完整评估，返回指标与逐题明细。"""
    chunk_citations = citation_map or {}
    details: list[dict[str, Any]] = []
    ranks: list[int | None] = []
    for case in cases:
        hits = _search(backend, pipeline, bm25, case["question"], max(K_VALUES))
        chunk_ids = [hit.chunk_id for hit in hits]
        gold = set(case["gold_chunk_ids"])
        rank = next(
            (index for index, chunk_id in enumerate(chunk_ids, start=1) if chunk_id in gold),
            None,
        )
        ranks.append(rank)

        answer, used_citation = _answer_with_rag(hits, case["question"])
        citations = [hit.citation for hit in hits if hit.citation]
        fact_hit = case["gold_answer"] in answer
        gold_citations = [chunk_citations.get(chunk_id, "") for chunk_id in gold]
        citation_hit = any(
            citation and citation in citations for citation in gold_citations
        )
        details.append(
            {
                "id": case["id"],
                "question": case["question"],
                "gold_answer": case["gold_answer"],
                "rank": rank,
                "top_chunk_ids": chunk_ids,
                "answer": answer,
                "used_citation": used_citation,
                "citations": citations,
                "fact_hit": fact_hit,
                "citation_hit": citation_hit,
                "no_answer": answer == NO_ANSWER,
            }
        )

    total = len(cases)
    recall = {
        f"recall@{k}": round(
            sum(1 for rank in ranks if rank is not None and rank <= k) / total, 4
        )
        for k in K_VALUES
    }
    mrr = round(
        sum(1.0 / rank for rank in ranks if rank is not None) / total, 4
    )
    completeness = [score_completeness(item) for item in details]

    return {
        "backend": backend,
        "embedding_fingerprint": embedder.fingerprint if embedder is not None else "bm25",
        "recall": recall,
        "mrr": mrr,
        "fact_rate": round(sum(item["fact_hit"] for item in details) / total, 4),
        "citation_rate": round(sum(item["citation_hit"] for item in details) / total, 4),
        "refusal_rate": round(sum(item["no_answer"] for item in details) / total, 4),
        "completeness": round(sum(completeness) / total, 2),
        "details": details,
    }


def score_completeness(item: dict[str, Any]) -> int:
    """确定性完整度评分（1–5）：事实越准、引用越全，得分越高。"""
    if item["no_answer"]:
        return 2
    if item["fact_hit"] and item["citation_hit"]:
        return 5
    if item["fact_hit"]:
        return 4
    if item["citation_hit"]:
        return 2
    return 1


def evaluate_no_rag(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """无 RAG 基线：固定泛化回答，事实命中恒为 0。"""
    details = []
    for case in cases:
        details.append(
            {
                "id": case["id"],
                "question": case["question"],
                "gold_answer": case["gold_answer"],
                "rank": None,
                "answer": NO_RAG_ANSWER,
                "citations": [],
                "fact_hit": case["gold_answer"] in NO_RAG_ANSWER,
                "citation_hit": False,
                "no_answer": False,
            }
        )
    total = len(cases)
    return {
        "backend": "no_rag",
        "embedding_fingerprint": "-",
        "recall": {"recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0},
        "mrr": 0.0,
        "fact_rate": 0.0,
        "citation_rate": 0.0,
        "refusal_rate": 0.0,
        "completeness": round(sum(score_completeness(item) for item in details) / total, 2),
        "details": details,
    }


def _citation_map() -> dict[str, str]:
    """chunk_id → 引用串，用于判断 gold chunk 是否被引用。"""
    return {chunk.chunk_id: citation_for_chunk(chunk) for chunk in corpus_chunks()}


def _markdown_cell(value: Any) -> str:
    """转义 Markdown 表格单元格。"""
    text = str(value if value is not None else "")
    return text.replace("|", "\\|").replace("\n", "<br>")


def render_markdown(
    results: Sequence[dict[str, Any]],
    cases: Sequence[dict[str, Any]],
    *,
    skipped: Sequence[str] = (),
) -> str:
    """把评估结果渲染成 Markdown 报告（含原始 JSON）。"""
    lines = [
        "# M8 检索与 RAG 评估报告",
        "",
        "## 评估设置",
        "",
        f"- 语料：{len(KNOWLEDGE_DOCS)} 篇旅行知识，共 {len(corpus_chunks())} 个 chunk",
        f"- 评估集：`data/eval/travel_qa.jsonl`，{len(cases)} 条单跳问题",
        "- 检索：BM25 单路 / Hashing 稠密 + BM25 混合（RRF）/ 可选 FastEmbed + BM25",
        f"- 生成方式：`py -3.14 scripts/eval_rag.py --out docs/m8-memory-rag-eval.md`",
    ]
    if skipped:
        lines.extend(["", "跳过的后端：" + "；".join(skipped)])
    lines.extend(
        [
            "",
            "## 召回命中率",
            "",
            "| 后端 | Recall@1 | Recall@3 | Recall@5 | MRR |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for result in results:
        recall = result["recall"]
        lines.append(
            f"| {result['backend']} | {recall['recall@1']:.2f} | "
            f"{recall['recall@3']:.2f} | {recall['recall@5']:.2f} | {result['mrr']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 回答层指标（带 RAG vs 不带 RAG）",
            "",
            "| 后端 | 事实命中率 | 引用覆盖率 | 拒答率 | 平均完整度(1–5) |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    no_rag = evaluate_no_rag(cases)
    for result in list(results) + [no_rag]:
        lines.append(
            f"| {result['backend']} | {result['fact_rate']:.2f} | "
            f"{result['citation_rate']:.2f} | {result['refusal_rate']:.2f} | "
            f"{result['completeness']:.2f} |"
        )

    primary = next(
        (result for result in results if result["backend"] == "hashing"), results[0]
    )
    lines.extend(
        [
            "",
            f"## 逐题对照（后端：{primary['backend']}）",
            "",
            "| 问题 | 标准答案 | 命中排名 | 不带 RAG | 带 RAG（摘录） | 事实命中 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for detail in primary["details"]:
        snippet = detail["answer"].split("\n依据：")[0]
        lines.append(
            f"| {_markdown_cell(detail['question'])} | {_markdown_cell(detail['gold_answer'])} | "
            f"{detail['rank'] if detail['rank'] is not None else '未命中'} | "
            f"{_markdown_cell(NO_RAG_ANSWER)} | {_markdown_cell(snippet)} | "
            f"{'是' if detail['fact_hit'] else '否'} |"
        )

    payload = {
        "cases": list(cases),
        "results": list(results),
        "no_rag": no_rag,
        "skipped": list(skipped),
    }
    lines.extend(
        [
            "",
            "## 原始结果",
            "",
            "<details>",
            "<summary>展开 JSON</summary>",
            "",
            "```json",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "```",
            "",
            "</details>",
            "",
        ]
    )
    return "\n".join(lines)


def run_eval(
    backend: str = "all",
    *,
    cases_path: str | Path = DEFAULT_CASES,
    online: bool = False,
) -> dict[str, Any]:
    """执行评估并返回 Markdown 文本与结构化结果。"""
    cases = load_cases(cases_path)
    citation_map = _citation_map()
    results: list[dict[str, Any]] = []
    skipped: list[str] = []

    requested = ["bm25", "hashing", "fastembed"] if backend == "all" else [backend]
    for name in requested:
        if name == "bm25":
            index = _bm25_only()
            results.append(
                evaluate(name, cases, bm25=index, citation_map=citation_map)
            )
            continue
        if name == "fastembed":
            if importlib.util.find_spec("fastembed") is None:
                if backend == "fastembed":
                    raise RuntimeError(
                        "未安装 fastembed，无法使用语义后端。请执行："
                        "py -3.14 -m pip install 'fastembed>=0.9,<1.0'"
                    )
                skipped.append("fastembed（未安装，跳过语义模式）")
                continue
            if not online and backend == "all":
                skipped.append(
                    "fastembed（未指定 --online，跳过，避免自动下载模型）"
                )
                continue
            embedder = FastEmbedEmbedder()
        else:
            embedder = HashingEmbedder()
        pipeline = _index_pipeline(embedder)
        results.append(
            evaluate(
                name,
                cases,
                embedder=embedder,
                pipeline=pipeline,
                citation_map=citation_map,
            )
        )

    return {
        "markdown": render_markdown(results, cases, skipped=skipped),
        "results": results,
        "cases": cases,
        "skipped": skipped,
    }


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="RAG 召回与回答评估")
    parser.add_argument(
        "--backend",
        choices=["all", "bm25", "hashing", "fastembed"],
        default="all",
        help="评估后端；all = BM25 + Hashing（fastembed 需 --online 且已安装）",
    )
    parser.add_argument(
        "--cases", default=str(DEFAULT_CASES), help="JSONL 评估集路径"
    )
    parser.add_argument("--out", default=None, help="Markdown 输出路径；缺省打印到标准输出")
    parser.add_argument(
        "--online",
        action="store_true",
        help="允许加载 FastEmbed 语义后端（首次会下载模型）",
    )
    args = parser.parse_args(argv)

    result = run_eval(args.backend, cases_path=args.cases, online=args.online)
    markdown = result["markdown"]
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8", newline="\n")
        print(f"评估报告已写入 {output}")
    else:
        print(markdown)
    for skipped in result["skipped"]:
        print(f"跳过：{skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
