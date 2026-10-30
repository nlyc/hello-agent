"""工具注册表：双注册 + 热插拔 + 线程安全。"""

import threading
from typing import Any, Callable, Optional

from agentflow.core.exceptions import ToolError
from agentflow.tools.base import FunctionTool, Tool, ToolParameter


class ToolRegistry:
    """统一工具注册表。运行中可增删替换（热插拔），线程安全。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._lock = threading.RLock()

    def register_tool(self, tool: Tool) -> None:
        with self._lock:
            self._tools[tool.name] = tool

    def register_function(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters: Optional[list[ToolParameter]] = None,
    ) -> None:
        """一行注册现成函数。func 的形参名须与 parameters 的 name 一致（缺省为 input）。"""
        self.register_tool(FunctionTool(name, description, func, parameters))

    def unregister(self, name: str) -> bool:
        """注销工具；返回是否确实存在。"""
        with self._lock:
            return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Tool:
        with self._lock:
            if name not in self._tools:
                raise ToolError(f"未知工具 '{name}'，可用：{sorted(self._tools)}")
            return self._tools[name]

    def execute(self, name: str, parameters: dict[str, Any]) -> str:
        tool = self.get(name)
        try:
            return tool.run(parameters)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"执行工具 '{name}' 失败: {e}") from e

    def list_tools(self) -> list[Tool]:
        with self._lock:
            return list(self._tools.values())

    def names(self) -> list[str]:
        return [t.name for t in self.list_tools()]

    def get_tools_description(self) -> str:
        """把全部工具拼成提示词片段（description 依旧是灵魂）。"""
        tools = self.list_tools()
        if not tools:
            return "（无可用工具）"
        lines = []
        for t in tools:
            params = ", ".join(f"{p.name}: {p.type}" for p in t.get_parameters())
            lines.append(f"- {t.name}({params}): {t.description}")
        return "\n".join(lines)

    def to_openai_schema(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self.list_tools()]

    def clear(self) -> None:
        with self._lock:
            self._tools.clear()

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._tools

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)
