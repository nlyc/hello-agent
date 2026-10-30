"""工具系统地基：Tool 抽象 + ToolParameter 自描述 + 函数工具包装。"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional


@dataclass
class ToolParameter:
    """工具参数自描述（自动化文档 / schema 生成 / 参数验证的基础）。"""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    default: Any = None


class Tool(ABC):
    """工具基类：run(parameters) -> str 统一执行 + get_parameters() 自描述。"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: dict[str, Any]) -> str:
        """执行工具，返回字符串结果。"""

    def get_parameters(self) -> list[ToolParameter]:
        """工具参数定义；默认无参。"""
        return []

    def validate_parameters(self, parameters: dict[str, Any]) -> list[str]:
        """返回缺失的必填参数名列表。"""
        return [p.name for p in self.get_parameters() if p.required and p.name not in parameters]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [asdict(p) for p in self.get_parameters()],
        }

    def to_openai_schema(self) -> dict[str, Any]:
        """转成 OpenAI 原生 function calling 的 tool schema。"""
        params = self.get_parameters()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        p.name: {"type": p.type, "description": p.description} for p in params
                    },
                    "required": [p.name for p in params if p.required],
                },
            },
        }

    def __str__(self) -> str:
        return f"Tool(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()


class FunctionTool(Tool):
    """把普通函数包装成 Tool，供 register_function 一行注册。"""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters: Optional[list[ToolParameter]] = None,
    ):
        super().__init__(name, description)
        self._func = func
        self._parameters = parameters or [
            ToolParameter(name="input", type="string", required=True)
        ]

    def run(self, parameters: dict[str, Any]) -> str:
        return str(self._func(**parameters))

    def get_parameters(self) -> list[ToolParameter]:
        return self._parameters
