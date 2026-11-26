"""agentflow.context —— 上下文工程：GSSC 流水线、六分区、来源适配与压缩。"""

from agentflow.context.builder import ContextBuilder
from agentflow.context.compress import PerSectionCompressor, truncate_to_tokens
from agentflow.context.scoring import (
    JaccardScorer,
    RelevanceScorer,
    combine_score,
    jaccard_similarity,
    recency_score,
)
from agentflow.context.sources import (
    ContextSource,
    HistoryContextSource,
    MemoryContextSource,
    RetrievalContextSource,
    StaticContextSource,
    history_packets,
    parse_timestamp,
)
from agentflow.context.types import (
    SECTIONS,
    BuiltContext,
    ContextConfig,
    ContextPacket,
    SectionTemplate,
    clamp01,
    ensure_aware,
    utc_now,
)

__all__ = [
    "SECTIONS",
    "BuiltContext",
    "ContextBuilder",
    "ContextConfig",
    "ContextPacket",
    "ContextSource",
    "HistoryContextSource",
    "JaccardScorer",
    "MemoryContextSource",
    "PerSectionCompressor",
    "RelevanceScorer",
    "RetrievalContextSource",
    "SectionTemplate",
    "StaticContextSource",
    "clamp01",
    "combine_score",
    "ensure_aware",
    "history_packets",
    "jaccard_similarity",
    "parse_timestamp",
    "recency_score",
    "truncate_to_tokens",
    "utc_now",
]
