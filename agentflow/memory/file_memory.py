"""FileMemory：JSON 文件持久化记忆（进程重启后仍在）。"""

import json
import os

from agentflow.memory.base import BaseMemory, _format_recall


class FileMemory(BaseMemory):
    """把记忆落盘为 JSON 文件，跨进程持久。"""

    def __init__(self, path: str):
        self.path = path
        self._items: list[tuple[str, str]] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self._items = [(d["key"], d["value"]) for d in data]

    def _save(self) -> None:
        data = [{"key": k, "value": v} for k, v in self._items]
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def remember(self, key: str, value: str) -> None:
        self._items.append((key, value))
        self._save()

    def recall(self, query: str) -> str:
        return _format_recall(self._items, query)

    def clear(self) -> None:
        self._items = []
        if os.path.exists(self.path):
            os.remove(self.path)
