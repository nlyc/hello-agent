"""计算器工具：ast 解析 + 运算符/函数白名单，绝不 eval。"""

import ast
import math
import operator
from typing import Any

from agentflow.core.exceptions import ToolError
from agentflow.tools.base import Tool, ToolParameter

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "max": max,
    "min": min,
    "sum": sum,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
}


class CalculatorTool(Tool):
    """安全计算器：只执行白名单内的运算符与函数，LLM 输出不可信。"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="执行数学计算，支持 + - * / ** 与 sqrt/abs/round/max/min/sum/sin/cos/tan/log/exp 等。",
        )

    def run(self, parameters: dict[str, Any]) -> str:
        expression = parameters.get("expression", "") or parameters.get("input", "")
        if not expression:
            raise ToolError("计算表达式不能为空")
        try:
            tree = ast.parse(str(expression), mode="eval")
            return str(self._eval(tree.body))
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"计算失败: {e}") from e

    def _eval(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            op = _OPERATORS.get(type(node.op))
            if op is None:
                raise ToolError(f"不支持的运算符: {type(node.op).__name__}")
            return op(self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _OPERATORS.get(type(node.op))
            if op is None:
                raise ToolError(f"不支持的运算符: {type(node.op).__name__}")
            return op(self._eval(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
                raise ToolError(f"不支持的函数: {getattr(node.func, 'id', '?')}")
            args = [self._eval(a) for a in node.args]
            return _FUNCTIONS[node.func.id](*args)
        if isinstance(node, ast.Name):
            if node.id not in _FUNCTIONS:
                raise ToolError(f"未定义的变量: {node.id}")
            return _FUNCTIONS[node.id]
        raise ToolError(f"不支持的表达式: {type(node).__name__}")

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="要计算的数学表达式",
                required=True,
            )
        ]
