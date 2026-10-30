"""MessageHistory：会话历史，追加 + 按 token 预算从尾部裁剪 + 复制访问。"""

import re

from agentflow.core.message import Message


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：CJK 每字约 1 token，其余按词近似。仅用于防超限，非精确计数。"""
    if not text:
        return 0
    cjk = len(re.findall(r"[一-鿿]", text))
    rest = re.sub(r"[一-鿿]", " ", text)
    words = len(re.findall(r"\S+", rest))
    return cjk + words


class MessageHistory:
    """会话历史的容器：追加消息、超长自动裁尾、按 token 预算截断、复制访问。"""

    def __init__(self, max_length: int = 100):
        self._messages: list[Message] = []
        self.max_length = max_length

    def add(self, message: Message) -> None:
        self._messages.append(message)
        if len(self._messages) > self.max_length:
            self._messages = self._messages[-self.max_length :]

    def get(self) -> list[Message]:
        return self._messages.copy()

    def get_recent(self, n: int) -> list[Message]:
        return self._messages[-n:].copy()

    def clear(self) -> None:
        self._messages.clear()

    def truncate(self, max_tokens: int) -> list[Message]:
        """从尾部向前保留，直到 token 预算用尽（至少保留最近一条）。"""
        kept: list[Message] = []
        total = 0
        for msg in reversed(self._messages):
            tokens = estimate_tokens(msg.content)
            if kept and total + tokens > max_tokens:
                break
            kept.append(msg)
            total += tokens
        kept.reverse()
        return kept

    def to_dicts(self) -> list[dict]:
        return [m.to_dict() for m in self._messages]

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)

    def __contains__(self, message: Message) -> bool:
        return message in self._messages
