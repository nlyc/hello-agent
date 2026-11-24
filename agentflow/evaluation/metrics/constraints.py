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
    "非",
    "免",
    "休息",
    "闭馆",
    "除外",
)

_LOCAL_NEGATIONS = ("不", "未", "无", "非", "免", "别", "没")

_CONDITIONALS = ("若", "如果", "恰逢", "万一", "备选", "对调", "改为")


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
        candidates = [
            float(value)
            for value in re.findall(r"(?:预算|花费|总价)[^\d]{0,6}(\d+(?:\.\d+)?)", source)
        ]
        checks["budget"] = bool(candidates) and max(candidates) <= float(budget)

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


def _violated(text: str, item: str) -> bool:
    """是否被正向安排：否定、闭馆、避开或条件句都不算违规。"""
    for index in _occurrences(text, item):
        prefix = text[max(0, index - 10) : index]
        suffix = text[index + len(item) : index + len(item) + 8]
        context = f"{prefix}{suffix}"
        if any(negation in context for negation in _NEGATIONS):
            continue
        if any(marker in prefix for marker in _CONDITIONALS):
            continue
        return True
    return False
