"""CalculatorTool（ast 白名单求值，绝不 eval）测试。"""

import pytest

from agentflow.core.exceptions import ToolError
from agentflow.tools.builtin import CalculatorTool


def calc(expr):
    return CalculatorTool().run({"expression": expr})


def test_basic_arithmetic():
    assert calc("2 + 3 * 4") == "14"
    assert calc("2 ** 8") == "256"
    assert calc("sqrt(16)") == "4.0"


def test_functions():
    assert calc("abs(-5)") == "5"
    assert calc("max(1, 5, 3)") == "5"
    assert calc("sin(0)") == "0.0"


def test_rejects_import():
    with pytest.raises(ToolError):
        calc("__import__('os').system('whoami')")


def test_rejects_attribute_access():
    with pytest.raises(ToolError):
        calc("(1).__class__")


def test_accepts_input_key_alias():
    assert CalculatorTool().run({"input": "1 + 1"}) == "2"


def test_empty_expression_raises():
    with pytest.raises(ToolError):
        calc("")
