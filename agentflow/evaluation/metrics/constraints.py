"""行程程序化约束校验：天数、预算、必含/避开项。

这些检查不依赖 LLM，作为 Judge 的硬约束补充。
"""

from __future__ import annotations

import re
from typing import Any


def check_itinerary(
    text: str, constraints: dict[str, Any] | None = None
) -> dict[str, Any]:
    """检查行程是否满足显式约束。"""
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
        checks["must_include"] = all(item in source for item in must_include)

    must_avoid = [str(item) for item in rules.get("must_avoid", []) or []]
    if must_avoid:
        checks["must_avoid"] = all(item not in source for item in must_avoid)

    companions = [str(item) for item in rules.get("companions", []) or []]
    if companions:
        checks["companions"] = all(item in source for item in companions)

    return {
        "checks": checks,
        "passed": all(checks.values()) if checks else True,
    }
