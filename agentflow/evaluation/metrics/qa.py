"""QA 评估：准精确匹配、包含匹配、引用覆盖与答案提取。"""

from __future__ import annotations

import re
from typing import Any, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = re.compile(r"[.,;:!?，。；：！？、]+$")
_CURRENCY = re.compile(r"[¥$€£]")
_UNITS = re.compile(
    r"(人民币|美元|元|公里|千米|kg|千克|米|小时|分钟|分钟|分钟|h|min|%)\s*$",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def extract_final_answer(text: str) -> str:
    """按 GAIA 纪律提取答案：FINAL ANSWER → 答案: → 最后一个非空行。"""
    source = str(text or "").strip()
    if not source:
        return ""
    for pattern in (
        r"FINAL\s*ANSWER\s*[:：]\s*(.+)",
        r"最终答案\s*[:：]\s*(.+)",
        r"答案\s*[:：]\s*(.+)",
    ):
        match = re.search(pattern, source, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip().splitlines()[0].strip()
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def as_number(value: Any) -> float | None:
    """把 ``$1,234.56`` / ``60元`` / ``50%`` 解析为数字；失败返回 None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    text = _CURRENCY.sub("", text).replace(",", "").replace("，", "").strip()
    text = _UNITS.sub("", text).strip()
    return float(text) if _NUMBER.match(text) else None


def normalize_string(text: Any) -> str:
    """小写、去冠词、去末尾标点、压缩空白。"""
    value = str(text or "").strip().lower()
    value = _ARTICLES.sub(" ", value)
    value = _PUNCT.sub("", value)
    return " ".join(value.split())


def normalize_list(text: Any) -> list[str] | None:
    """逗号/顿号/分号分隔的列表：逐项归一化后按字母排序。"""
    raw = str(text or "").strip()
    if not re.search(r"[,，、;；]", raw):
        return None
    items = [
        normalize_string(item)
        for item in re.split(r"[,，、;；]", raw)
        if item.strip()
    ]
    if len(items) < 2:
        return None
    return sorted(items)


def quasi_exact_match(prediction: Any, gold: Any) -> bool:
    """准精确匹配：数字按数值、列表按排序、字符串按归一化。"""
    left_number = as_number(prediction)
    right_number = as_number(gold)
    if left_number is not None and right_number is not None:
        return abs(left_number - right_number) <= 1e-9
    left_list = normalize_list(prediction)
    right_list = normalize_list(gold)
    if left_list is not None and right_list is not None:
        return left_list == right_list
    return normalize_string(prediction) == normalize_string(gold)


def containment_match(prediction: Any, gold: Any) -> bool:
    """归一化后的 gold 是否被 prediction 包含。"""
    left = normalize_string(prediction)
    right = normalize_string(gold)
    return bool(right) and right in left


def citation_coverage(
    citations: Sequence[str] | None, expected: Sequence[str] | None
) -> float:
    """期望引用被实际引用覆盖的比例；无期望时记 1.0。"""
    targets = [str(item).strip() for item in (expected or []) if str(item).strip()]
    if not targets:
        return 1.0
    actual = [str(item).strip() for item in (citations or []) if str(item).strip()]
    covered = sum(
        1 for target in targets if any(target in citation for citation in actual)
    )
    return covered / len(targets)


def evaluate_qa(
    prediction: str,
    gold: str,
    *,
    citations: Sequence[str] | None = None,
    expected_citations: Sequence[str] | None = None,
) -> dict[str, Any]:
    """单题 QA 指标：QEM / 包含 / 引用覆盖 / 是否通过。"""
    answer = extract_final_answer(prediction)
    exact = quasi_exact_match(answer, gold)
    contains = containment_match(answer, gold)
    coverage = citation_coverage(citations, expected_citations)
    passed = (exact or contains) and coverage >= 1.0
    return {
        "answer": answer,
        "exact_match": exact,
        "contains": contains,
        "citation_coverage": coverage,
        "passed": passed,
    }
