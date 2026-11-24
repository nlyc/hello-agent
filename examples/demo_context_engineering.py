"""上下文工程 Demo：Terminal 即时探索 → Note 持久化 → ContextBuilder 装配。

    py -3.14 -m examples.demo_context_engineering --out .cache/context_demo

全程离线，不需要 LLM / API Key。演示第 9 章的分层上下文：
即时访问（Terminal）+ 会话/事实（Memory 静态示例）+ 持久笔记（Note），
最后由 ContextBuilder 按 GSSC 选出六分区上下文，并对比朴素拼接的 token。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentflow.context import (
    ContextBuilder,
    ContextConfig,
    ContextPacket,
    StaticContextSource,
)
from agentflow.core.message import Message
from agentflow.memory.message_history import estimate_tokens
from agentflow.notes import NoteContextSource, NoteStore
from agentflow.tools import TerminalTool

ROLE = "你是「智游·山水行」旅行助手，只基于给定资料回答。"
OUTPUT = "给出每天可执行的安排；资料不足时明确说明。"


def prepare_workspace(workspace: Path) -> None:
    """写入两篇攻略，供 TerminalTool 即时检索。"""
    guides = workspace / "guides"
    guides.mkdir(parents=True, exist_ok=True)
    (guides / "beijing.md").write_text(
        "# 北京\n\n故宫周一闭馆，需提前预约。\n"
        "颐和园东宫门到长廊一线地势平缓，适合不爬坡的游客。\n",
        encoding="utf-8",
    )
    (guides / "shanghai.md").write_text(
        "# 上海\n\n外滩夜景最佳时间 19:00–21:00。\n"
        "雨天优先上海博物馆与上海科技馆。\n",
        encoding="utf-8",
    )


def main(out_dir: str | Path | None = None) -> int:
    base = Path(out_dir or ".cache/context_demo")
    workspace = base / "workspace"
    notes_root = base / "notes"
    prepare_workspace(workspace)

    # 1) JIT：终端即时探索，而不是预索引全部攻略。
    terminal = TerminalTool(workspace)
    grep_output = terminal.run({"command": 'grep -n "闭馆" guides'})
    print("[terminal] grep 闭馆 ->")
    print(grep_output)

    # 2) NoteTool：把探索结论沉淀为跨会话笔记。
    store = NoteStore(notes_root, user_id="demo")
    existing = {note.id for note in store.list()}
    if "note-gugong-closed" not in existing:
        store.create(
            "故宫周一闭馆",
            "需把故宫改到周二或周三。",
            type="blocker",
            tags=["北京", "故宫"],
            note_id="note-gugong-closed",
        )
    if "note-yiheyuan-flat" not in existing:
        store.create(
            "颐和园平缓路线",
            "东宫门到长廊一线适合不爬坡。",
            type="conclusion",
            tags=["北京", "颐和园"],
            note_id="note-yiheyuan-flat",
        )
    if "note-gugong-booking" not in existing:
        store.create(
            "提前预约故宫",
            "旺季建议提前三天预约。",
            type="action",
            tags=["北京", "故宫"],
            note_id="note-gugong-booking",
        )

    memory_packets = [
        ContextPacket(
            content="同行人：父母；偏好：少走路；预算：1500 元。",
            relevance_score=0.95,
            source="memory",
            section="context",
            trusted=True,
        )
    ]
    evidence_packets = [
        ContextPacket(
            content=f"即时检索结果：\n{grep_output}",
            relevance_score=0.9,
            source="rag",
            section="evidence",
            trusted=True,
        )
    ]

    builder = ContextBuilder(
        [
            StaticContextSource(memory_packets, name="memory"),
            StaticContextSource(evidence_packets, name="rag"),
            NoteContextSource(store),
        ],
        config=ContextConfig(max_tokens=600, min_section_tokens=20),
        role_prompt=ROLE,
        output_rules=OUTPUT,
    )
    # 20 轮历史：Builder 只保留最近 5 条，朴素拼接则全量塞入。
    history = [
        Message(
            role="user" if index % 2 == 0 else "assistant",
            content=f"历史对话 {index + 1}：关于北京行程的讨论与微调。",
        )
        for index in range(20)
    ]
    built = builder.build("帮我规划北京两日游", history=history)

    print("\n[context] 六分区上下文：")
    print(built.text)
    print("\n[context] stats:")
    print(
        json.dumps(
            {
                "total_tokens": built.stats["total_tokens"],
                "sections": built.stats["sections"],
                "packets_selected": built.stats["packets_selected"],
                "dropped_relevance": built.stats["packets_dropped_relevance"],
                "dropped_budget": built.stats["packets_dropped_budget"],
            },
            ensure_ascii=False,
        )
    )

    # 3) 对照：朴素拼接（角色 + 输出 + 问题 + 全部资料 + 全部笔记 + 全部历史）。
    naive = "\n".join(
        [
            ROLE,
            OUTPUT,
            "帮我规划北京两日游",
            grep_output,
            "同行人：父母；偏好：少走路；预算：1500 元。",
            *(
                f"{note.title}\n{note.content}"
                for note in store.list()
            ),
            *(message.content for message in history),
        ]
    )
    naive_tokens = estimate_tokens(naive)
    saved = naive_tokens - built.stats["total_tokens"]
    ratio = saved / naive_tokens if naive_tokens else 0.0
    print(
        f"\n[tokens] 朴素拼接={naive_tokens}，上下文={built.stats['total_tokens']}，"
        f"节省={saved}（{ratio:.0%}）"
    )

    # 4) 跨会话：新 NoteStore 仍能读到 blocker。
    reopened = NoteStore(notes_root, user_id="demo")
    blockers = reopened.list(note_type="blocker")
    print(f"[notes] 新会话读取 blocker：{[note.title for note in blockers]}")

    base.mkdir(parents=True, exist_ok=True)
    (base / "context.json").write_text(
        json.dumps(
            {
                "built": built.to_dict(),
                "terminal_output": grep_output,
                "naive_tokens": naive_tokens,
                "saved_tokens": saved,
                "notes": [note.to_dict() for note in reopened.list()],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[output] {base / 'context.json'}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="上下文工程离线 Demo")
    parser.add_argument("--out", default=".cache/context_demo", help="输出目录")
    args = parser.parse_args()
    raise SystemExit(main(args.out))
