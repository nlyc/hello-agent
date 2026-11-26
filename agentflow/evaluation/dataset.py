"""评测集加载：JSONL + 数据集哈希 + 字段校验。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from agentflow.core.exceptions import EvaluationError
from agentflow.evaluation.types import EvalCase


@dataclass
class EvalDataset:
    """一份 JSONL 评测集。"""

    name: str
    path: str
    cases: list[EvalCase] = field(default_factory=list)
    sha256: str = ""

    @classmethod
    def load(cls, path: str | Path, *, name: str | None = None) -> "EvalDataset":
        """读取 JSONL；每行一个对象，缺失 ``question`` 时报错。"""
        target = Path(path)
        if not target.exists():
            raise EvaluationError(f"评测集不存在：{target}")
        suite = name or target.stem
        cases: list[EvalCase] = []
        for line_no, item in enumerate(iter_jsonl(target), start=1):
            if not isinstance(item, dict):
                raise EvaluationError(f"{target}:{line_no} 必须是 JSON 对象")
            if "question" not in item:
                raise EvaluationError(f"{target}:{line_no} 缺少 question")
            cases.append(
                EvalCase.from_dict(item, suite=suite, index=len(cases) + 1)
            )
        if not cases:
            raise EvaluationError(f"评测集为空：{target}")
        return cls(
            name=suite,
            path=str(target),
            cases=cases,
            sha256=sha256_file(target),
        )

    def filtered(
        self,
        *,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[EvalCase]:
        """按类别过滤并截断。"""
        result = [
            case
            for case in self.cases
            if not category or case.category == category
        ]
        if limit is not None and limit >= 0:
            result = result[:limit]
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "sha256": self.sha256,
            "cases": len(self.cases),
        }


def iter_jsonl(path: str | Path) -> Iterator[Any]:
    """逐行读取 JSONL，跳过空行。"""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except ValueError as exc:
                raise EvaluationError(f"{path} 存在非法 JSON：{exc}") from exc


def sha256_file(path: str | Path) -> str:
    """计算文件 SHA-256（用于复现清单）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """写出 JSONL（评测数据生成用）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
