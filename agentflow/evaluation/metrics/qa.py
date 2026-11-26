"""QA 评估：准精确匹配、包含匹配、引用覆盖与答案提取。"""

from __future__ import annotations

import re
from typing import Any, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = re.compile(r"[.,;:!?，。；：！？、]+$")
_MARKDOWN = re.compile(r"[*_`~]+")
_CURRENCY = re.compile(r"[¥$€£]")
_UNITS = re.compile(
    r"(人民币|美元|元|公里|千米|kg|千克|米|小时|分钟|h|min|%)\s*$",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_CN_NUMBER = re.compile(r"[零一二两三四五六七八九十]{1,3}(?=[点时号日天])")
_CN_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def _cn_to_int(text: str) -> int | None:
    """把 ``七`` / ``十七`` / ``二十`` 等简单中文数字转成整数。"""
    if not text:
        return None
    if text == "十":
        return 10
    if text.startswith("十") and len(text) == 2:
        return 10 + _CN_DIGITS.get(text[1], 0)
    if "十" in text:
        left, _, right = text.partition("十")
        tens = _CN_DIGITS.get(left, 1) if left else 1
        ones = _CN_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    return _CN_DIGITS.get(text)


def extract_final_answer(text: str) -> str:
    """按 GAIA 纪律提取答案：FINAL ANSWER → 答案: → 去掉引用块的正文。"""
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
    # RAG 回答常在末尾追加「依据：…」；它不是答案，先剥离。
    citation = re.search(r"\n\s*(?:依据|参考来源|参考引用)\s*[:：]", source)
    if citation:
        source = source[: citation.start()].strip()
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
    """小写、去 Markdown、去冠词、中文时间数字转阿拉伯、压缩空白。"""
    value = str(text or "").strip().lower()
    value = _MARKDOWN.sub("", value)
    value = _ARTICLES.sub(" ", value)
    value = _CN_NUMBER.sub(
        lambda match: str(_cn_to_int(match.group(0)) or match.group(0)), value
    )
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


def _citation_norm(text: Any) -> str:
    """引用比较归一化：去空白、斜杠、标点，避免排版差异误判。"""
    return re.sub(r"[\s/\\|,，、;；:：]+", "", str(text or "").lower())


def citation_coverage(
    citations: Sequence[str] | None, expected: Sequence[str] | None
) -> float | None:
    """期望引用被实际引用覆盖的比例；没有期望时返回 ``None``（不参与聚合）。"""
    targets = [str(item).strip() for item in (expected or []) if str(item).strip()]
    if not targets:
        return None
    actual = [str(item).strip() for item in (citations or []) if str(item).strip()]
    actual_norm = [_citation_norm(item) for item in actual]
    covered = sum(
        1
        for target in targets
        if any(_citation_norm(target) and _citation_norm(target) in citation for citation in actual_norm)
    )
    return covered / len(targets)


def evaluate_qa(
    prediction: str,
    gold: str,
    *,
    citations: Sequence[str] | None = None,
    expected_citations: Sequence[str] | None = None,
) -> dict[str, Any]:
    """单题 QA 指标：QEM / 包含 / 引用覆盖。

    引用覆盖是独立诊断指标，不参与 ``passed``；答案正确性只看 QEM/包含。
    """
    answer = extract_final_answer(prediction)
    exact = quasi_exact_match(answer, gold)
    contains = containment_match(answer, gold)
    coverage = citation_coverage(citations, expected_citations)
    passed = exact or contains
    return {
        "answer": answer,
        "exact_match": exact,
        "contains": contains,
        "citation_coverage": coverage,
        "passed": passed,
    }
