"""行程程序化约束校验：天数、预算、必含/避开项。

这些检查不依赖 LLM，作为 Judge 的硬约束补充。
"""

from __future__ import annotations

import re
from typing import Any

_NEGATIONS = (
    "不",
    "未",
    "别",
    "无需",
    "避开",
    "排除",
    "不建议",
    "不安排",
    "不去",
    "少",
    "非",
    "免",
    "休息",
    "闭馆",
    "除外",
)

_LOCAL_NEGATIONS = ("不", "未", "无", "非", "免", "别", "没")


def check_itinerary(
    text: str, constraints: dict[str, Any] | None = None
) -> dict[str, Any]:
    """检查行程是否满足显式约束。

    ``must_include`` 要求正向提及（避免「不安排故宫」被判为包含）；
    ``must_avoid`` 只在正向安排时判违规（「避开周一」「周一闭馆」不算）。
    """
    source = str(text or "")
    rules = dict(constraints or {})
    checks: dict[str, bool] = {}

    days = rules.get("days")
    if days is not None:
        labels = {int(match) for match in re.findall(r"D(\d+)", source, re.I)}
        checks["days"] = bool(labels) and max(labels) >= int(days)

    budget = rules.get("budget")
    if budget is not None:
        checks["budget"] = _budget_ok(source, float(budget))

    must_include = [str(item) for item in rules.get("must_include", []) or []]
    if must_include:
        checks["must_include"] = all(
            _included(source, item) for item in must_include
        )

    must_avoid = [str(item) for item in rules.get("must_avoid", []) or []]
    if must_avoid:
        checks["must_avoid"] = all(
            not _violated(source, item) for item in must_avoid
        )

    companions = [str(item) for item in rules.get("companions", []) or []]
    if companions:
        checks["companions"] = all(
            _included(source, item) for item in companions
        )

    return {
        "checks": checks,
        "passed": all(checks.values()) if checks else True,
    }


def _occurrences(text: str, item: str):
    """迭代 item 在 text 中的所有位置。"""
    start = 0
    while item and start <= len(text):
        index = text.find(item, start)
        if index < 0:
            return
        yield index
        start = index + len(item)


def _included(text: str, item: str) -> bool:
    """是否被正向提及：只看紧邻的局部否定词，避免跨句误伤。"""
    for index in _occurrences(text, item):
        prefix = text[max(0, index - 4) : index]
        if not any(negation in prefix for negation in _LOCAL_NEGATIONS):
            return True
    return False


_WEEKDAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
_WEEKDAY_NEGATIONS = (
    "避开",
    "不",
    "非",
    "若",
    "如遇",
    "逢",
    "免",
    "休息",
    "闭馆",
    "除外",
    "对调",
)
_SCHEDULE_VERBS = ("去", "游", "安排", "参观", "行程", "逛", "玩")


def _budget_ok(text: str, budget: float) -> bool:
    """预算块内取最大金额与预算比较。

    优先「预算明细 / 总预算 / 预算控制 / 合计」块；这些块里取最小的大额数字，
    避免「多人同行会涨到 1000–1100」之类的调整段落把实际预算拉高。
    """
    blocks = re.split(r"\n\s*\n", str(text or ""))
    budget_blocks = [
        block
        for block in blocks
        if any(key in block for key in ("预算", "合计", "总价", "花费"))
    ]
    if not budget_blocks:
        return False
    preferred = [
        block
        for block in budget_blocks
        if any(key in block for key in ("预算明细", "总预算", "预算控制", "合计"))
    ]
    maxima: list[float] = []
    for block in preferred or budget_blocks:
        numbers = [
            float(value)
            for value in re.findall(r"\d+(?:\.\d+)?", block)
            if float(value) < 1_000_000
        ]
        if numbers:
            maxima.append(max(numbers))
    return bool(maxima) and min(maxima) <= budget


def _violated(text: str, item: str) -> bool:
    """是否被正向安排：否定、闭馆、避开或条件句都不算违规。"""
    if item in _WEEKDAYS:
        for match in re.finditer(item, text):
            prefix = text[max(0, match.start() - 15) : match.start()]
            suffix = text[match.end() : match.end() + 8]
            if any(negation in prefix for negation in _WEEKDAY_NEGATIONS):
                continue
            if re.search(r"(D\d|第[一二三四五六七八九十]+天)", prefix) and any(
                verb in f"{prefix}{suffix}" for verb in _SCHEDULE_VERBS
            ):
                return True
        return False

    # 非周几项（如“爬坡”）：只在“行程/安排/参观”句子内判断；
    # 引用资料或背景说明（“登顶需要连续爬坡”）不算安排。
    for sentence in re.split(r"[。；;！？\n]+", text):
        if not re.search(r"(D\d|第[一二三四五六七八九十]+天|安排|行程|参观|游览)", sentence):
            continue
        for match in re.finditer(re.escape(item), sentence):
            prefix = sentence[max(0, match.start() - 40) : match.start()]
            if any(negation in prefix for negation in _NEGATIONS):
                continue
            return True
    return False
