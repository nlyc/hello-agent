"""上下文压缩：超预算时的分区级截断。

结构化数据（代码、数字、引用）优先截断而不是摘要，避免 LLM 摘要编造；
叙事性历史可在 P1 换成 LLM 摘要。本模块只依赖 token 估算。
"""

from __future__ import annotations

from typing import Mapping

from agentflow.memory.message_history import estimate_tokens


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """按 token 估算二分截断文本。"""
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip()


class PerSectionCompressor:
    """按分区分配预算并截断，保证每区至少保留 ``min_section_tokens``。

    传入的 ``sections`` 是「分区正文」（不含标题）；``headers`` 用于扣除标题
    开销。返回新的正文字典与统计，调用方负责重新渲染。
    """

    def __init__(self, min_section_tokens: int = 50) -> None:
        self.min_section_tokens = max(0, int(min_section_tokens))

    def compress(
        self,
        sections: Mapping[str, str],
        max_tokens: int,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[dict[str, str], dict]:
        """压缩分区正文，返回 ``(新正文, 统计)``。"""
        headers = headers or {}
        bodies = {
            str(name): str(body or "").strip()
            for name, body in sections.items()
        }
        non_empty = {name: body for name, body in bodies.items() if body}
        lengths = {name: estimate_tokens(body) for name, body in non_empty.items()}
        # 只计算非空分区的标题开销。
        header_tokens = sum(
            estimate_tokens(str(headers.get(name, "")))
            for name in non_empty
        )
        total_before = sum(lengths.values()) + header_tokens
        stats: dict = {
            "before_tokens": total_before,
            "header_tokens": header_tokens,
            "max_tokens": int(max_tokens),
            "truncated_sections": [],
            "allocations": {},
        }
        if max_tokens <= 0:
            stats["after_tokens"] = 0
            stats["truncated_sections"] = list(non_empty)
            return {name: "" for name in bodies}, stats

        available = max_tokens - header_tokens
        if available <= 0:
            stats["after_tokens"] = 0
            stats["truncated_sections"] = list(non_empty)
            return {name: "" for name in bodies}, stats

        total_body = sum(lengths.values())
        if total_body <= available:
            stats["after_tokens"] = total_before
            return bodies, stats

        allocations = self._allocate(lengths, available)
        compressed: dict[str, str] = {}
        for name, body in bodies.items():
            if not body:
                compressed[name] = ""
                continue
            allocation = allocations.get(name, 0)
            new_body = truncate_to_tokens(body, allocation)
            compressed[name] = new_body
            stats["allocations"][name] = allocation
            if estimate_tokens(new_body) < lengths[name]:
                stats["truncated_sections"].append(name)
        stats["after_tokens"] = (
            sum(estimate_tokens(body) for body in compressed.values())
            + header_tokens
        )
        return compressed, stats

    def _allocate(self, lengths: Mapping[str, int], available: int) -> dict[str, int]:
        """按最小保留 + 剩余比例分配。"""
        total = sum(lengths.values())
        if total <= 0:
            return {name: 0 for name in lengths}

        minimums = {
            name: min(self.min_section_tokens, length)
            for name, length in lengths.items()
        }
        min_total = sum(minimums.values())
        allocations: dict[str, int] = {}

        if min_total >= available:
            # 预算连最小保留都放不下：按长度比例切分。
            for name, length in lengths.items():
                allocations[name] = int(available * length / total)
        else:
            remaining = available - min_total
            extra_pool = sum(length - minimums[name] for name, length in lengths.items())
            for name, length in lengths.items():
                share = 0
                if extra_pool > 0:
                    share = int(remaining * (length - minimums[name]) / extra_pool)
                allocations[name] = minimums[name] + share

        # 把取整后的剩余预算分给还有增长空间的分区。
        leftover = available - sum(allocations.values())
        while leftover > 0:
            progressed = False
            for name, length in sorted(
                lengths.items(), key=lambda item: allocations.get(item[0], 0), reverse=True
            ):
                if allocations.get(name, 0) < length and leftover > 0:
                    allocations[name] = allocations.get(name, 0) + 1
                    leftover -= 1
                    progressed = True
            if not progressed:
                break
        return allocations
