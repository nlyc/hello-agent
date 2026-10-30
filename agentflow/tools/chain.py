"""工具链：线性串联多个工具，context 字典承载中间结果，模板变量传值。"""

from typing import Any, Optional

from agentflow.core.exceptions import ToolError
from agentflow.tools.base import Tool


class ToolChain:
    """线性工具链。add_step(tool, input_template, output_key) 串步骤。"""

    def __init__(self, name: str = "", description: str = ""):
        self.name = name
        self.description = description
        self._steps: list[tuple[Tool, str, str]] = []
        self._context: dict[str, Any] = {}

    def add_step(
        self,
        tool: Tool,
        input_template: str = "{input}",
        output_key: Optional[str] = None,
    ) -> "ToolChain":
        """添加一步。input_template 用 context 变量替换后作为工具的 input 参数。"""
        output_key = output_key or f"step_{len(self._steps)}_result"
        self._steps.append((tool, input_template, output_key))
        return self

    def run(self, input_text: str, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """顺序执行：每步渲染模板 → 执行工具 → 结果写入 context；返回完整 context。"""
        ctx = dict(context or {})
        ctx["input"] = input_text
        for tool, template, output_key in self._steps:
            try:
                rendered = template.format(**ctx)
            except KeyError as e:
                raise ToolError(f"模板变量缺失: {e}") from e
            ctx[output_key] = tool.run({"input": rendered})
        self._context = ctx
        return ctx

    @property
    def steps(self) -> int:
        return len(self._steps)


class ToolChainManager:
    """工具链管理器：统一注册与执行多条链。"""

    def __init__(self):
        self._chains: dict[str, ToolChain] = {}

    def register(self, chain: ToolChain) -> None:
        self._chains[chain.name] = chain

    def execute(self, name: str, input_text: str, context: Optional[dict] = None) -> dict:
        if name not in self._chains:
            raise ToolError(f"工具链 '{name}' 不存在")
        return self._chains[name].run(input_text, context)

    def list_chains(self) -> list[str]:
        return list(self._chains)
