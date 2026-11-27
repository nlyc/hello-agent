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
_CN_NUMBER = re.compile(r"[零一二两三四五六七八九十]{1,3}(?=[点时号日天月])")
_CN_DECIMAL = re.compile(r"([零一二两三四五六七八九])点([零一二三四五六七八九])")
_CN_MEASURE = re.compile(
    r"([零一二两三四五六七八九十]{1,3})(?=(?:元|公里|千米|小时|分钟|岁|人|次|个|只|座|层|米))"
)
_TIME_COLON = re.compile(r"(\d{1,2})\s*:\s*(\d{2})")
_NIGHT_RANGE = re.compile(
    r"(?:晚上|傍晚|夜里)\s*"
    r"(\d{1,2}|[零一二两三四五六七八九十]{1,3})\s*点\s*"
    r"(?:至|到|[-–—~～])\s*"
    r"(\d{1,2}|[零一二两三四五六七八九十]{1,3})\s*点"
)
_DASH = re.compile(r"[–—~～]")
_LIST_SEPARATOR = re.compile(r"[,，、;；]|(?<=.)和(?=.)|(?<=.)及(?=.)")
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


def _cn_or_int(text: str) -> int | None:
    """``七`` / ``十七`` / ``7`` 统一转整数。"""
    if text.isdigit():
        return int(text)
    return _cn_to_int(text)


def _night_range(match: re.Match[str]) -> str:
    """``晚上七点至九点`` / ``晚上7点至9点`` → ``19点-21点``。"""
    start = _cn_or_int(match.group(1))
    end = _cn_or_int(match.group(2))
    if start is None or end is None:
        return match.group(0)
    if start < 12:
        start += 12
    if end < 12:
        end += 12
    return f"{start}点-{end}点"


def strip_citations(text: str) -> str:
    """去掉末尾「依据：/参考来源：」引用块，保留正文。"""
    source = str(text or "").strip()
    citation = re.search(r"\n\s*(?:依据|参考来源|参考引用)\s*[:：]", source)
    if citation:
        source = source[: citation.start()].strip()
    return source


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
    source = strip_citations(source)
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
    """小写、去 Markdown、去冠词、中文数字/时间转阿拉伯、压缩空白。"""
    value = str(text or "").strip().lower()
    value = _MARKDOWN.sub("", value)
    value = _ARTICLES.sub(" ", value)
    # 晚上七点至九点 / 晚上7点至9点 → 19点-21点，与 19:00–21:00 对齐。
    value = _NIGHT_RANGE.sub(_night_range, value)
    value = _DASH.sub("-", value)
    # 8:30 → 8点30分；八点半 → 八点30分，随后统一中文数字。
    value = _TIME_COLON.sub(
        lambda match: f"{int(match.group(1))}点{int(match.group(2))}分", value
    )
    value = value.replace("点半", "点30分")
    value = _CN_DECIMAL.sub(
        lambda match: f"{_CN_DIGITS.get(match.group(1), 0)}."
        f"{_CN_DIGITS.get(match.group(2), 0)}",
        value,
    )
    value = _CN_NUMBER.sub(
        lambda match: str(_cn_to_int(match.group(0)) or match.group(0)), value
    )
    value = re.sub(r"(\d+)点0分", r"\1点", value)
    value = _CN_MEASURE.sub(
        lambda match: str(_cn_to_int(match.group(0)) or match.group(0)), value
    )
    value = _PUNCT.sub("", value)
    # 中文/数字之间的排版空格不承载语义：`1.5 公里` ≡ `1.5公里`。
    value = re.sub(r"(?<=[\d一-鿿])\s+(?=[\d一-鿿])", "", value)
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
    """归一化后的 gold 是否被 prediction 包含；列表项支持顺序无关。"""
    left = normalize_string(prediction)
    right = normalize_string(gold)
    if bool(right) and right in left:
        return True
    # 「A、B 和 C」只要各项都出现即可，避免分隔符差异导致假阴性。
    raw_items = [
        item.strip()
        for item in _LIST_SEPARATOR.split(str(gold or ""))
        if item.strip()
    ]
    if len(raw_items) >= 2 and all(len(item) >= 2 for item in raw_items):
        items = [normalize_string(item) for item in raw_items]
        return all(item and item in left for item in items)
    return False


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
    body = strip_citations(prediction)
    exact = quasi_exact_match(answer, gold)
    # 多行 RAG 回答常在正文列出事实、末行才给结论；
    # 包含判定同时看正文，避免因排版差异把正确答案判错。
    contains = containment_match(answer, gold) or containment_match(body, gold)
    coverage = citation_coverage(citations, expected_citations)
    passed = exact or contains
    return {
        "answer": answer,
        "exact_match": exact,
        "contains": contains,
        "citation_coverage": coverage,
        "passed": passed,
    }
