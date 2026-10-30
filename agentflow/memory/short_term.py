"""ShortTermMemory：进程内滑动窗口记忆（不持久化）。"""

from agentflow.memory.base import BaseMemory, _format_recall


class ShortTermMemory(BaseMemory):
    """滑动窗口记忆：只保留最近 max_items 条，重启即失。"""

    def __init__(self, max_items: int = 50):
        self.max_items = max_items
        self._items: list[tuple[str, str]] = []

    def remember(self, key: str, value: str) -> None:
        self._items.append((key, value))
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items :]

    def recall(self, query: str) -> str:
        return _format_recall(self._items, query)

    def clear(self) -> None:
        self._items.clear()
