"""上下文数据模型测试：Packet / Config / BuiltContext / Template。"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from agentflow.context import (
    SECTIONS,
    BuiltContext,
    ContextConfig,
    ContextPacket,
    SectionTemplate,
)
from agentflow.core.agent import memory_text


def test_packet_auto_token_and_utc():
    packet = ContextPacket(content="故宫开放时间", timestamp=datetime(2026, 1, 1))
    assert packet.token_count > 0
    assert packet.timestamp.tzinfo is not None
    assert packet.section == "context"


def test_packet_invalid_section_falls_back():
    packet = ContextPacket(content="x", section="unknown")
    assert packet.section == "context"


def test_packet_clamps_relevance_and_serializes():
    packet = ContextPacket(content="记忆", relevance_score=2.0, trusted=True)
    data = packet.to_dict()
    assert packet.relevance_score == 1.0
    assert data["content"] == "记忆"
    assert data["timestamp"].endswith("+00:00")


def test_context_config_defaults():
    config = ContextConfig()
    assert config.max_tokens == 3000
    assert config.relevance_weight + config.recency_weight == pytest.approx(1.0)
    assert config.memory_limit == 10
    assert config.rag_limit == 5
    assert config.reserve_ratio == pytest.approx(0.2)


def test_context_config_rejects_weights_not_one():
    with pytest.raises(ValidationError):
        ContextConfig(relevance_weight=0.5, recency_weight=0.3)


def test_context_config_rejects_bad_budget():
    with pytest.raises(ValidationError):
        ContextConfig(max_tokens=0)
    with pytest.raises(ValidationError):
        ContextConfig(reserve_ratio=1.0)


def test_section_template_render_skips_empty():
    template = SectionTemplate()
    text = template.render({"role": "系统指令", "task": "任务", "evidence": ""})
    assert text.startswith("# Role & Policies\n系统指令")
    assert "# Task\n任务" in text
    assert "# Evidence" not in text
    assert not text.endswith("\n")


def test_section_template_rejects_unknown_section():
    with pytest.raises(Exception):
        SectionTemplate(order=("role", "unknown"))


def test_built_context_duck_types_memory_context():
    packet = ContextPacket(content="带父母，少走路", source="memory", section="context")
    built = BuiltContext(
        text="# Role & Policies\n系统\n\n# Task\n当前问题\n\n# Context\n带父母，少走路",
        prompt_text="# Role & Policies\n系统\n\n# Context\n带父母，少走路",
        sections={"context": "带父母，少走路"},
        packets=[packet],
        stats={"max_tokens": 3000},
    )
    assert built
    assert built.section("context") == "带父母，少走路"
    assert built.items[0]["content"] == "带父母，少走路"
    assert built.to_dict()["packets"][0]["source"] == "memory"
    # Agent 注入用 prompt_text（不含 Task，避免与自身输入重复）。
    assert memory_text(built) == built.prompt_text
    assert "# Task" not in memory_text(built)
    assert set(SECTIONS) == {"role", "task", "state", "evidence", "context", "output"}
