"""BaseMemory 抽象：remember / recall，可 as_tool() 回归工具形态。"""

from abc import ABC, abstractmethod

from agentflow.tools.base import Tool, ToolParameter


def _format_recall(items, query: str) -> str:
    """把 (key, value) 序列按 query 做子串匹配，最近优先，拼成文本。"""
    q = (query or "").lower().strip()
    matched = []
    for k, v in reversed(list(items)):
        if not q or q in f"{k} {v}".lower():
            matched.append(f"{k}: {v}")
    return "\n".join(matched)


class BaseMemory(ABC):
    """记忆后端统一协议。remember 写入、recall 检索、clear 清空。"""

    @abstractmethod
    def remember(self, key: str, value: str) -> None:
        ...

    @abstractmethod
    def recall(self, query: str) -> str:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    def as_tool(self, name: str = "memory") -> Tool:
        """回归「万物皆为工具」：把记忆包装成 Tool，供 Agent 显式存/取。"""
        memory = self

        class MemoryTool(Tool):
            def __init__(self):
                super().__init__(
                    name=name,
                    description="读写长期记忆。action=remember 存储 key/value；action=recall 按 query 检索。",
                )

            def run(self, parameters):
                action = parameters.get("action", "recall")
                if action == "remember":
                    key = parameters.get("key", "")
                    value = parameters.get("value", "")
                    if not key:
                        return "错误: remember 需要 key"
                    memory.remember(key, value)
                    return f"已记住 {key}"
                return memory.recall(parameters.get("query", "")) or "（无相关记忆）"

            def get_parameters(self):
                return [
                    ToolParameter("action", "string", "remember 或 recall", True),
                    ToolParameter("key", "string", "记忆键（remember 用）"),
                    ToolParameter("value", "string", "记忆值（remember 用）"),
                    ToolParameter("query", "string", "检索词（recall 用）"),
                ]

        return MemoryTool()
