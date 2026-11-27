"""评估数据模型：EvalCase / EvalResult / EvalReport。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCase:
    """一条评测样本。"""

    id: str
    suite: str
    category: str = ""
    question: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    ground_truth: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, suite: str, index: int) -> "EvalCase":
        metadata = dict(data.get("metadata", {}) or {})
        # 兼容 M8 的 travel_qa.jsonl 字段（gold_answer / gold_chunk_ids / city）。
        metadata.setdefault("gold_chunk_ids", list(data.get("gold_chunk_ids", []) or []))
        metadata.setdefault("city", str(data.get("city", "") or ""))
        metadata.setdefault("tags", list(data.get("tags", []) or []))
        metadata.setdefault("loose_fields", list(data.get("loose_fields", []) or []))
        ground_truth = data.get("ground_truth")
        if ground_truth is None:
            ground_truth = data.get("gold_answer")
        return cls(
            id=str(data.get("id") or f"{suite}_{index:03d}"),
            suite=suite,
            category=str(data.get("category", "") or ""),
            question=str(data.get("question", "") or ""),
            payload=dict(data.get("payload", {}) or {}),
            ground_truth=ground_truth,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "suite": self.suite,
            "category": self.category,
            "question": self.question,
            "payload": dict(self.payload),
            "ground_truth": self.ground_truth,
            "metadata": dict(self.metadata),
        }


@dataclass
class EvalResult:
    """单条样本的评测结果。"""

    case_id: str
    passed: bool
    score: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    error: str = ""
    latency_ms: float = 0.0
    tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "score": self.score,
            "metrics": dict(self.metrics),
            "output": self.output,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalResult":
        """从报告 JSON 还原单条结果。"""
        return cls(
            case_id=str(data.get("case_id", "")),
            passed=bool(data.get("passed")),
            score=float(data.get("score", 0.0) or 0.0),
            metrics=dict(data.get("metrics", {}) or {}),
            output=str(data.get("output", "") or ""),
            error=str(data.get("error", "") or ""),
            latency_ms=float(data.get("latency_ms", 0.0) or 0.0),
            tokens=int(data.get("tokens", 0) or 0),
        )


@dataclass
class EvalReport:
    """一套基准的完整报告。"""

    suite: str
    system: str
    cases: list[EvalResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "system": self.system,
            "cases": [case.to_dict() for case in self.cases],
            "metrics": dict(self.metrics),
            "manifest": dict(self.manifest),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalReport":
        """从报告 JSON 还原；用于多次采样聚合。"""
        return cls(
            suite=str(data.get("suite", "")),
            system=str(data.get("system", "")),
            cases=[
                EvalResult.from_dict(item)
                for item in (data.get("cases", []) or [])
                if isinstance(item, dict)
            ],
            metrics=dict(data.get("metrics", {}) or {}),
            manifest=dict(data.get("manifest", {}) or {}),
        )
