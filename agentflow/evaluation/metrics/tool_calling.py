"""工具调用评估：AST 匹配（函数名精确 + 参数无序 + 值语义等价）。"""

from __future__ import annotations

import ast
import json
import operator
from dataclasses import dataclass, field
from typing import Any, Sequence

from agentflow.core.exceptions import EvaluationError

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


@dataclass
class ToolCall:
    """一次结构化工具调用。"""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    positional: list[Any] = field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> "ToolCall":
        """解析 ``func(a=1, b='x')``；只允许字面量与四则运算。"""
        try:
            node = ast.parse(str(text).strip(), mode="eval").body
        except SyntaxError as exc:
            raise EvaluationError(f"无法解析工具调用：{text}") from exc
        if not isinstance(node, ast.Call):
            raise EvaluationError(f"不是函数调用：{text}")
        name = _call_name(node.func)
        arguments: dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                raise EvaluationError("不支持 **kwargs 形式的工具调用")
            arguments[keyword.arg] = safe_eval(keyword.value)
        positional = [safe_eval(arg) for arg in node.args]
        return cls(name=name, arguments=arguments, positional=positional)

    @classmethod
    def from_dict(cls, data: Any) -> "ToolCall":
        """从 dict / JSON 字符串 / 文本解析。"""
        if isinstance(data, ToolCall):
            return data
        if isinstance(data, str):
            text = data.strip()
            if text.startswith("{"):
                try:
                    data = json.loads(text)
                except ValueError:
                    return cls.from_text(text)
            else:
                return cls.from_text(text)
        if not isinstance(data, dict):
            raise EvaluationError(f"工具调用必须是对象：{data!r}")
        name = str(
            data.get("name")
            or data.get("tool")
            or data.get("function")
            or ""
        ).strip()
        if not name:
            raise EvaluationError(f"工具调用缺少 name：{data!r}")
        arguments = (
            data.get("arguments")
            or data.get("params")
            or data.get("kwargs")
            or {}
        )
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError as exc:
                raise EvaluationError(f"arguments 不是 JSON：{arguments!r}") from exc
        if not isinstance(arguments, dict):
            raise EvaluationError(f"arguments 必须是对象：{arguments!r}")
        return cls(
            name=name,
            arguments=dict(arguments),
            positional=list(data.get("positional", []) or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": dict(self.arguments),
            "positional": list(self.positional),
        }


def _call_name(node: ast.AST) -> str:
    """取函数名；支持 ``module.func``。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_call_name(node.value)}.{node.attr}"
    raise EvaluationError("工具调用必须是函数名或属性路径")


def safe_eval(node: ast.AST) -> Any:
    """安全求值：仅允许字面量、容器与四则运算，绝不 ``eval``。"""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = safe_eval(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](safe_eval(node.left), safe_eval(node.right))
    if isinstance(node, ast.List):
        return [safe_eval(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(safe_eval(item) for item in node.elts)
    if isinstance(node, ast.Set):
        return {safe_eval(item) for item in node.elts}
    if isinstance(node, ast.Dict):
        return {
            safe_eval(key): safe_eval(value)
            for key, value in zip(node.keys, node.values)
        }
    raise EvaluationError(f"不安全的表达式节点：{type(node).__name__}")


def values_equal(left: Any, right: Any) -> bool:
    """参数值语义等价：2+3 ≡ 5、'x' ≡ "x"、bool 不与数字混同。"""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 1e-9
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            values_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, str) and isinstance(right, str):
        return left.strip() == right.strip()
    return left == right


def calls_equal(expected: ToolCall, actual: ToolCall) -> bool:
    """函数名精确一致 + 参数集合无序 + 值语义等价。"""
    if expected.name != actual.name:
        return False
    if set(expected.arguments) != set(actual.arguments):
        return False
    if not all(
        values_equal(expected.arguments[key], actual.arguments[key])
        for key in expected.arguments
    ):
        return False
    return all(
        values_equal(a, b) for a, b in zip(expected.positional, actual.positional)
    ) and len(expected.positional) == len(actual.positional)


def match_calls(
    expected: Sequence[ToolCall],
    actual: Sequence[ToolCall],
) -> dict[str, Any]:
    """多重集合匹配；返回是否完全一致与匹配数。"""
    remaining = list(actual)
    matched = 0
    for item in expected:
        for index, candidate in enumerate(remaining):
            if calls_equal(item, candidate):
                remaining.pop(index)
                matched += 1
                break
    exact = (
        matched == len(expected)
        and len(actual) == len(expected)
        and not remaining
    )
    return {
        "expected_count": len(expected),
        "actual_count": len(actual),
        "matched": matched,
        "exact_match": exact,
    }


def evaluate_tool_calls(
    expected: Sequence[ToolCall],
    actual: Sequence[ToolCall],
    *,
    category: str = "",
) -> dict[str, Any]:
    """按类别评估：irrelevance 看是否克制，其余看调用集合是否一致。"""
    if category == "irrelevance":
        passed = len(actual) == 0
        return {
            "passed": passed,
            "exact_match": passed,
            "expected_count": 0,
            "actual_count": len(actual),
            "matched": 0,
            "false_positive": not passed,
        }
    result = match_calls(expected, actual)
    result["passed"] = bool(result["exact_match"])
    result["false_positive"] = bool(
        category == "irrelevance" and actual
    )
    return result
