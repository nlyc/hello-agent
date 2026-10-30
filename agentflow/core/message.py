"""消息系统：对内丰富，对外兼容 OpenAI 字典。"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """一条对话消息。

    role 严格四值，对齐 OpenAI 规范；timestamp / metadata 预留扩展；
    name 供 tool 角色携带工具名（v1.0 文本范式暂不产出，为原生 FC 预留）。
    """

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """转成 OpenAI API 的消息字典。"""
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            data["name"] = self.name
        return data

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
