"""ContextBuilder：GSSC 四阶段流水线。

Gather 多源汇集 → Select 评分选择 → Structure 六分区 → Compress 分区兜底。
Memory / RAG / Notes / History 都只是原料来源；本模块是唯一的「装配车间」。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from agentflow.context.compress import PerSectionCompressor
from agentflow.context.scoring import (
    JaccardScorer,
    RelevanceScorer,
    combine_score,
    recency_score,
)
from agentflow.context.sources import (
    ContextSource,
    history_packets,
)
from agentflow.context.types import (
    SECTIONS,
    BuiltContext,
    ContextConfig,
    ContextPacket,
    SectionTemplate,
    ensure_aware,
    utc_now,
)
from agentflow.memory.message_history import estimate_tokens

# 来源名称 → 配置中的召回上限。
_SOURCE_LIMITS = {
    "memory": "memory_limit",
    "rag": "rag_limit",
    "note": "note_limit",
}


class ContextBuilder:
    """按 GSSC 流水线构建一次上下文。

    参数：
        sources: 信息源列表；每个 Source 独立失败。
        config: 预算、权重、召回上限等。
        scorer: 相关性打分器；默认 Jaccard。
        compressor: 压缩器；默认按 ``min_section_tokens`` 分区截断。
        template: 六分区模板。
        role_prompt: 系统指令（Role & Policies），始终保留。
        output_rules: 输出约束，始终保留。
    """

    def __init__(
        self,
        sources: Sequence[ContextSource] = (),
        config: ContextConfig | Mapping[str, Any] | None = None,
        *,
        scorer: RelevanceScorer | None = None,
        compressor: PerSectionCompressor | None = None,
        template: SectionTemplate | None = None,
        role_prompt: str = "",
        output_rules: str = "",
    ) -> None:
        if isinstance(config, ContextConfig):
            self.config = config
        else:
            self.config = ContextConfig.model_validate(dict(config or {}))
        self.sources = list(sources)
        self.scorer = scorer or JaccardScorer()
        self.compressor = compressor or PerSectionCompressor(
            self.config.min_section_tokens
        )
        self.template = template or SectionTemplate()
        self.role_prompt = str(role_prompt or "").strip()
        self.output_rules = str(output_rules or "").strip()

    def build(
        self,
        task: str,
        *,
        history: Any | None = None,
        extra_packets: Sequence[ContextPacket] | None = None,
        now: datetime | None = None,
    ) -> BuiltContext:
        """构建上下文；任何单源失败都不会阻断主流程。"""
        task_text = str(task or "").strip()
        current = ensure_aware(now or utc_now())

        always_packets = self._always_packets(task_text, current)
        gathered: list[ContextPacket] = []
        source_errors: dict[str, str] = {}

        for source in self.sources:
            name = str(getattr(source, "name", "") or source.__class__.__name__)
            try:
                limit = self._limit_for(name)
                gathered.extend(
                    source.gather(task_text, limit=limit, now=current) or []
                )
            except Exception as exc:  # 单源失败不拖垮整体
                source_errors[name] = f"{type(exc).__name__}: {exc}"

        gathered.extend(
            history_packets(
                history, limit=self.config.recent_history, now=current
            )
        )
        if extra_packets:
            gathered.extend(list(extra_packets))

        system_tokens = sum(packet.token_count for packet in always_packets)
        selected, dropped_relevance, dropped_budget, candidate_tokens = self._select(
            task_text, gathered, current, system_tokens=system_tokens
        )
        candidate_budget = max(0, self.config.max_tokens - system_tokens)

        grouped = self._group(always_packets + selected)
        sections = {
            name: "\n\n".join(
                packet.content for packet in grouped.get(name, []) if packet.content
            ).strip()
            for name in SECTIONS
        }
        text = self.template.render(sections)

        compression_stats: dict[str, Any] = {}
        if estimate_tokens(text) > self.config.max_tokens:
            sections, compression_stats = self.compressor.compress(
                sections,
                self.config.max_tokens,
                headers=self.template.headers,
            )
            text = self.template.render(sections)

        packets = [
            packet
            for name in SECTIONS
            for packet in grouped.get(name, [])
        ]
        stats = {
            "max_tokens": self.config.max_tokens,
            "reserved_tokens": self.config.reserved_tokens(),
            "system_tokens": system_tokens,
            "candidate_budget": candidate_budget,
            "candidate_tokens": candidate_tokens,
            "total_tokens": estimate_tokens(text),
            "tokens_by_section": {
                name: estimate_tokens(
                    f"{self.template.headers.get(name, '')}\n{sections.get(name, '')}"
                )
                for name in SECTIONS
                if sections.get(name)
            },
            "sections": [name for name in SECTIONS if sections.get(name)],
            "packets_gathered": len(always_packets) + len(gathered),
            "packets_selected": len(always_packets) + len(selected),
            "packets_dropped_relevance": dropped_relevance,
            "packets_dropped_budget": dropped_budget,
            "source_errors": source_errors,
            "compression": compression_stats,
            "now": current.isoformat(timespec="seconds"),
        }
        return BuiltContext(
            text=text,
            sections=sections,
            packets=packets,
            stats=stats,
            task=task_text,
            enabled=True,
        )

    def _always_packets(
        self, task: str, now: datetime
    ) -> list[ContextPacket]:
        """系统指令 / Task / Output：始终保留，不参与相关性过滤。"""
        packets: list[ContextPacket] = []
        if self.role_prompt:
            packets.append(
                ContextPacket(
                    content=self.role_prompt,
                    timestamp=now,
                    relevance_score=1.0,
                    source="system",
                    section="role",
                    metadata={"kind": "role"},
                    trusted=True,
                )
            )
        if task:
            packets.append(
                ContextPacket(
                    content=task,
                    timestamp=now,
                    relevance_score=1.0,
                    source="task",
                    section="task",
                    metadata={"kind": "task"},
                    trusted=True,
                )
            )
        if self.output_rules:
            packets.append(
                ContextPacket(
                    content=self.output_rules,
                    timestamp=now,
                    relevance_score=1.0,
                    source="system",
                    section="output",
                    metadata={"kind": "output"},
                    trusted=True,
                )
            )
        return packets

    def _limit_for(self, source_name: str) -> int:
        """按来源名称取配置上限；未知来源给保守值 10。"""
        attribute = _SOURCE_LIMITS.get(source_name)
        if attribute is None:
            return 10
        return int(getattr(self.config, attribute))

    def _select(
        self,
        task: str,
        packets: Sequence[ContextPacket],
        now: datetime,
        *,
        system_tokens: int,
    ) -> tuple[list[ContextPacket], int, int, int]:
        """评分 → 过滤 → 贪心装填。"""
        scored: list[ContextPacket] = []
        dropped_relevance = 0
        for packet in packets:
            if not packet.content:
                continue
            base = (
                packet.relevance_score
                if packet.trusted
                else self.scorer.score(task, packet.content)
            )
            if base < self.config.min_relevance:
                dropped_relevance += 1
                continue
            recency = recency_score(
                packet.timestamp,
                now=now,
                half_life_hours=self.config.recency_half_life_hours,
            )
            packet.score = combine_score(base, recency, self.config)
            scored.append(packet)

        scored.sort(
            key=lambda packet: (-packet.score, -packet.timestamp.timestamp())
        )

        budget = max(0, self.config.max_tokens - system_tokens)
        selected: list[ContextPacket] = []
        used = 0
        dropped_budget = 0
        for packet in scored:
            if used + packet.token_count <= budget:
                selected.append(packet)
                used += packet.token_count
            else:
                dropped_budget += 1
        return selected, dropped_relevance, dropped_budget, used

    @staticmethod
    def _group(
        packets: Sequence[ContextPacket],
    ) -> dict[str, list[ContextPacket]]:
        """按分区归位，分区内按综合分降序。"""
        grouped: dict[str, list[ContextPacket]] = {name: [] for name in SECTIONS}
        for packet in packets:
            grouped.setdefault(packet.section, []).append(packet)
        for name in grouped:
            grouped[name].sort(
                key=lambda packet: (-packet.score, -packet.timestamp.timestamp())
            )
        return grouped
