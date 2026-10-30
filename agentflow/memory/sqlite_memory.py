"""SQLiteMemory：SQLite 持久化记忆（key 唯一，重复 remember 覆盖）。"""

import sqlite3

from agentflow.memory.base import BaseMemory, _format_recall


class SQLiteMemory(BaseMemory):
    """把记忆存入 SQLite，key 唯一（重复写入覆盖旧值），跨进程持久。"""

    def __init__(self, path: str):
        self._conn = sqlite3.connect(path)
        self._conn.execute("CREATE TABLE IF NOT EXISTS memory (key TEXT PRIMARY KEY, value TEXT)")
        self._conn.commit()

    def remember(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO memory (key, value) VALUES (?, ?)", (key, value)
        )
        self._conn.commit()

    def recall(self, query: str) -> str:
        rows = self._conn.execute("SELECT key, value FROM memory ORDER BY rowid DESC").fetchall()
        return _format_recall(list(rows), query)

    def clear(self) -> None:
        self._conn.execute("DELETE FROM memory")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
