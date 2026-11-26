"""上下文工程数据模型：ContextPacket / ContextConfig / BuiltContext / 分区模板。

本模块只做纯数据承载与校验，不依赖 Memory、RAG 或 LLM，便于 Builder、
Source、Compressor 与测试共用同一套类型。时间统一使用 UTC。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from pydantic import BaseModel, Field, model_validator

from agentflow.core.exceptions import ContextError
from agentflow.memory.message_history import estimate_tokens

# 六分区：稳定骨架让人类和模型都容易定位问题，新信息源只需加新分区。
SECTIONS: tuple[str, ...] = (
    "role",
    "task",
    "state",
    "evidence",
    "context",
    "output",
)

DEFAULT_SECTION_HEADERS: dict[str, str] = {
    "role": "# Role & Policies",
    "task": "# Task",
    "state": "# State",
    "evidence": "# Evidence",
    "context": "# Context",
    "output": "# Output",
}


def utc_now() -> datetime:
    """返回当前 UTC 时间（带时区）。"""
    return datetime.now(timezone.utc)


def ensure_aware(value: datetime | None) -> datetime:
    """把可能为 naive 的时间统一为 UTC aware；None 返回当前时间。"""
    if value is None:
        return utc_now()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def clamp01(value: float) -> float:
    """把数值裁剪到 ``[0, 1]``。"""
    return max(0.0, min(1.0, float(value)))


@dataclass
class ContextPacket:
    """进入上下文窗口的一条基本信息单元。

    参数：
        content: 信息正文。
        timestamp: UTC 时间，用于新近性衰减。
        token_count: Token 估算；<=0 时自动计算。
        relevance_score: 原始相关性 ``[0, 1]``。
        source: 来源标识（memory / rag / note / history / terminal / custom）。
        section: 六分区之一；非法值回落 ``context``。
        metadata: 引用、类型、id 等附加信息。
        trusted: 为 True 时直接采用 ``relevance_score``，不再重算相关性。
        score: 选择阶段写入的综合分，供排序与统计使用。
    """

    content: str
    timestamp: datetime = field(default_factory=utc_now)
    token_count: int = 0
    relevance_score: float = 0.0
    source: str = ""
    section: str = "context"
    metadata: dict[str, Any] = field(default_factory=dict)
    trusted: bool = False
    score: float = 0.0

    def __post_init__(self) -> None:
        self.content = str(self.content or "").strip()
        self.timestamp = ensure_aware(self.timestamp)
        if self.token_count <= 0:
            self.token_count = estimate_tokens(self.content)
        if self.section not in SECTIONS:
            self.section = "context"
        self.relevance_score = clamp01(self.relevance_score)
        self.score = float(self.score)

    @classmethod
    def create(
        cls,
        content: str,
        *,
        timestamp: datetime | None = None,
        relevance_score: float = 0.0,
        source: str = "",
        section: str = "context",
        metadata: Mapping[str, Any] | None = None,
        trusted: bool = False,
    ) -> "ContextPacket":
        """便捷构造：自动估算 token_count。"""
        return cls(
            content=content,
            timestamp=ensure_aware(timestamp),
            relevance_score=relevance_score,
            source=source,
            section=section,
            metadata=dict(metadata or {}),
            trusted=trusted,
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(timespec="seconds"),
            "token_count": self.token_count,
            "relevance_score": self.relevance_score,
            "source": self.source,
            "section": self.section,
            "metadata": dict(self.metadata),
            "trusted": self.trusted,
            "score": self.score,
        }


class ContextConfig(BaseModel):
    """上下文构建配置。

    ``relevance_weight`` 与 ``recency_weight`` 必须和为 1；``reserve_ratio``
    只用于统计与提示，实际候选预算 = ``max_tokens - 系统指令 tokens``。
    """

    max_tokens: int = Field(default=3000, gt=0)
    reserve_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    min_relevance: float = Field(default=0.1, ge=0.0, le=1.0)
    relevance_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    recency_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    recent_history: int = Field(default=5, ge=0)
    memory_limit: int = Field(default=10, ge=1)
    rag_limit: int = Field(default=5, ge=1)
    note_limit: int = Field(default=10, ge=1)
    recency_half_life_hours: float = Field(default=72.0, gt=0)
    min_section_tokens: int = Field(default=50, ge=0)

    @model_validator(mode="after")
    def _validate_weights(self) -> "ContextConfig":
        total = self.relevance_weight + self.recency_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"relevance_weight + recency_weight 必须等于 1，当前为 {total}"
            )
        return self

    def reserved_tokens(self) -> int:
        """按比例计算给系统指令预留的 token 数（用于统计）。"""
        return int(self.max_tokens * self.reserve_ratio)

    @classmethod
    def from_env(cls) -> "ContextConfig":
        """从 ``.env`` / 环境变量读取上下文配置。"""
        from agentflow.core.env import env_float, env_int

        return cls(
            max_tokens=env_int("CONTEXT_MAX_TOKENS", 3000),
            reserve_ratio=env_float("CONTEXT_RESERVE_RATIO", 0.2),
            min_relevance=env_float("CONTEXT_MIN_RELEVANCE", 0.1),
            relevance_weight=env_float("CONTEXT_RELEVANCE_WEIGHT", 0.7),
            recency_weight=env_float("CONTEXT_RECENCY_WEIGHT", 0.3),
            recent_history=env_int("CONTEXT_RECENT_HISTORY", 5),
            memory_limit=env_int("CONTEXT_MEMORY_LIMIT", 10),
            rag_limit=env_int("CONTEXT_RAG_LIMIT", 5),
            note_limit=env_int("CONTEXT_NOTE_LIMIT", 10),
            recency_half_life_hours=env_float(
                "CONTEXT_RECENCY_HALF_LIFE_HOURS", 72.0
            ),
            min_section_tokens=env_int("CONTEXT_MIN_SECTION_TOKENS", 50),
        )


@dataclass(frozen=True)
class SectionTemplate:
    """六分区渲染模板：固定顺序 + 可替换标题。"""

    order: tuple[str, ...] = SECTIONS
    headers: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_SECTION_HEADERS)
    )

    def __post_init__(self) -> None:
        missing = [name for name in self.order if name not in SECTIONS]
        if missing:
            raise ContextError(f"未知分区：{missing}")
        if set(self.order) != set(SECTIONS):
            raise ContextError("SectionTemplate 必须包含全部六个分区且不重复")

    def render(self, sections: Mapping[str, str]) -> str:
        """按顺序渲染非空分区；空分区直接省略。"""
        parts: list[str] = []
        for name in self.order:
            body = str(sections.get(name, "") or "").strip()
            if not body:
                continue
            header = self.headers.get(name, f"# {name}")
            parts.append(f"{header}\n{body}")
        return "\n\n".join(parts)


@dataclass
class BuiltContext:
    """一次上下文构建的结果。

    为兼容 v1.0 的 ``MemoryContext``，暴露 ``text`` 与 ``items``；同时保留
    ``sections`` / ``packets`` / ``stats`` 供调试与评估使用。
    """

    text: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    packets: list[ContextPacket] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    task: str = ""
    enabled: bool = True
    # 供 Agent 注入：不含 Task 分区，避免与 Agent 自己的输入重复。
    prompt_text: str = ""
    kind: str = "built"

    @property
    def items(self) -> list[dict[str, Any]]:
        """与 MemoryContext.items 对齐的轻量视图。"""
        return [
            {
                "content": packet.content,
                "source": packet.source,
                "section": packet.section,
                "score": packet.score,
                "metadata": dict(packet.metadata),
            }
            for packet in self.packets
        ]

    def section(self, name: str) -> str:
        """返回指定分区正文；不存在时为空串。"""
        return self.sections.get(name, "")

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""
        return {
            "text": self.text,
            "sections": dict(self.sections),
            "packets": [packet.to_dict() for packet in self.packets],
            "stats": dict(self.stats),
            "task": self.task,
            "enabled": self.enabled,
            "prompt_text": self.prompt_text,
            "kind": self.kind,
        }

    def __bool__(self) -> bool:
        """有文本时视为真，便于调用方 ``if context:`` 判断。"""
        return bool(self.text)
