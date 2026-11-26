"""ContextBuilder GSSC 流水线测试：汇集、选择、分区、压缩与容错。"""

from datetime import datetime, timezone

from agentflow.context import (
    BuiltContext,
    ContextBuilder,
    ContextConfig,
    ContextPacket,
)
from agentflow.core.message import Message

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class ListSource:
    def __init__(self, name, packets):
        self.name = name
        self.packets = list(packets)
        self.calls = []

    def gather(self, query, *, limit, now):
        self.calls.append({"query": query, "limit": limit, "now": now})
        return self.packets[:limit]


class BrokenSource:
    name = "rag"

    def gather(self, query, *, limit, now):
        raise RuntimeError("boom")


def make_packet(content, *, relevance=1.0, section="context", source="memory", trusted=True, timestamp=NOW):
    return ContextPacket(
        content=content,
        timestamp=timestamp,
        relevance_score=relevance,
        source=source,
        section=section,
        trusted=trusted,
    )


def test_builder_includes_role_task_output():
    builder = ContextBuilder(
        role_prompt="你是旅行助手",
        output_rules="只输出行程",
        config=ContextConfig(max_tokens=1000),
    )
    built = builder.build("北京两日游", now=NOW)
    assert isinstance(built, BuiltContext)
    assert "# Role & Policies" in built.text
    assert "# Task" in built.text
    assert "# Output" in built.text
    assert built.stats["packets_selected"] == 3
    assert built.stats["packets_gathered"] == 3
    assert built.sections["task"] == "北京两日游"
    assert "# Task" in built.text
    assert "# Task" not in built.prompt_text
    assert "你是旅行助手" in built.prompt_text


def test_builder_ranks_by_relevance_and_filters():
    source = ListSource(
        "memory",
        [
            make_packet("上海外滩夜景", relevance=0.2),
            make_packet("故宫开放时间", relevance=1.0),
        ],
    )
    builder = ContextBuilder(
        [source], config=ContextConfig(max_tokens=1000), role_prompt=""
    )
    built = builder.build("故宫", now=NOW)
    assert built.sections["context"].startswith("故宫开放时间")
    memory_packets = [p for p in built.packets if p.source == "memory"]
    assert memory_packets[0].score > memory_packets[1].score
    assert source.calls[0]["limit"] == 10  # memory_limit


def test_builder_min_relevance_filter():
    source = ListSource(
        "memory", [make_packet("不相关内容", relevance=0.05, trusted=True)]
    )
    built = ContextBuilder(
        [source],
        config=ContextConfig(max_tokens=1000, min_relevance=0.1),
        role_prompt="",
        output_rules="",
    ).build("故宫", now=NOW)
    assert "不相关内容" not in built.text
    assert built.stats["packets_dropped_relevance"] == 1


def test_builder_uses_jaccard_for_untrusted_packets():
    source = ListSource(
        "memory",
        [
            make_packet("故宫开放时间", relevance=0.0, trusted=False),
            make_packet("上海外滩夜景", relevance=0.0, trusted=False),
        ],
    )
    built = ContextBuilder(
        [source], config=ContextConfig(max_tokens=1000), role_prompt=""
    ).build("故宫", now=NOW)
    assert "故宫开放时间" in built.text
    # 外滩与「故宫」无关键词重叠，低于 min_relevance 被过滤。
    assert "上海外滩夜景" not in built.text


def test_builder_greedy_budget_drops_large_packet():
    source = ListSource(
        "memory",
        [
            make_packet("北" * 200, relevance=1.0),
            make_packet("北京", relevance=0.9),
        ],
    )
    built = ContextBuilder(
        [source],
        config=ContextConfig(max_tokens=30, min_section_tokens=0),
        role_prompt="",
        output_rules="",
    ).build("问题", now=NOW)
    assert "北京" in built.text
    assert "北" * 200 not in built.text
    assert built.stats["packets_dropped_budget"] >= 1


def test_builder_source_failure_is_isolated():
    good = ListSource("memory", [make_packet("故宫开放时间", relevance=1.0)])
    builder = ContextBuilder(
        [BrokenSource(), good],
        config=ContextConfig(max_tokens=1000),
        role_prompt="",
    )
    built = builder.build("故宫", now=NOW)
    assert "故宫开放时间" in built.text
    assert "boom" in built.stats["source_errors"]["rag"]
    assert built.stats["source_errors"]["rag"].startswith("RuntimeError")


def test_builder_history_only_last_n():
    history = [Message(role="user", content=f"历史{i}") for i in range(5)]
    built = ContextBuilder(
        config=ContextConfig(max_tokens=1000, recent_history=2),
        role_prompt="",
    ).build("现在", history=history, now=NOW)
    assert "历史3" in built.text
    assert "历史4" in built.text
    assert "历史2" not in built.text


def test_builder_extra_packets_are_selected():
    extra = [make_packet("手工注入的约束", relevance=1.0, section="state")]
    built = ContextBuilder(
        config=ContextConfig(max_tokens=1000), role_prompt=""
    ).build("任务", extra_packets=extra, now=NOW)
    assert built.sections["state"] == "手工注入的约束"
    assert built.stats["packets_selected"] == 2


def test_builder_compresses_over_budget():
    builder = ContextBuilder(
        role_prompt="北" * 100,
        output_rules="南" * 20,
        config=ContextConfig(max_tokens=50, min_section_tokens=5),
    )
    built = builder.build("东" * 30, now=NOW)
    assert built.stats["compression"]
    assert built.stats["total_tokens"] <= 50
    assert built.stats["compression"]["truncated_sections"]


def test_builder_tokens_by_section_and_stats():
    source = ListSource("rag", [make_packet("证据", section="evidence", source="rag")])
    built = ContextBuilder(
        [source],
        config=ContextConfig(max_tokens=1000),
        role_prompt="角色",
        output_rules="输出",
    ).build("任务", now=NOW)
    stats = built.stats
    assert stats["max_tokens"] == 1000
    assert stats["system_tokens"] > 0
    assert stats["candidate_budget"] == 1000 - stats["system_tokens"]
    assert "evidence" in stats["tokens_by_section"]
    assert stats["sections"] == ["role", "task", "evidence", "output"]
    assert stats["now"].endswith("+00:00")
    assert isinstance(built.to_dict(), dict)


def test_builder_accepts_config_mapping():
    builder = ContextBuilder(config={"max_tokens": 123})
    assert builder.config.max_tokens == 123
