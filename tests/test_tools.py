"""Tool 基类与 ToolRegistry（热插拔、线程安全）测试。"""

import threading

import pytest

from agentflow.core.exceptions import ToolError
from agentflow.tools import Tool, ToolParameter, ToolRegistry


class AddTool(Tool):
    def __init__(self):
        super().__init__(name="add", description="两个数相加")
        self._params = [
            ToolParameter(name="a", type="number", description="加数", required=True),
            ToolParameter(name="b", type="number", description="被加数", required=True),
        ]

    def run(self, parameters):
        return str(int(parameters["a"]) + int(parameters["b"]))

    def get_parameters(self):
        return self._params


def test_register_tool_and_execute():
    r = ToolRegistry()
    r.register_tool(AddTool())
    assert r.execute("add", {"a": 2, "b": 3}) == "5"


def test_register_function_single_input():
    r = ToolRegistry()
    r.register_function("upper", "转大写", lambda input: input.upper())
    assert r.execute("upper", {"input": "hi"}) == "HI"


def test_register_function_explicit_params():
    r = ToolRegistry()
    r.register_function(
        "mul",
        "相乘",
        lambda a, b: a * b,
        [ToolParameter("a", "number", "", True), ToolParameter("b", "number", "", True)],
    )
    assert r.execute("mul", {"a": 3, "b": 4}) == "12"


def test_unknown_tool_raises():
    with pytest.raises(ToolError):
        ToolRegistry().execute("nope", {})


def test_hotplug_unregister():
    r = ToolRegistry()
    r.register_function("f", "d", lambda input: input)
    assert "f" in r
    assert r.unregister("f") is True
    assert "f" not in r
    assert r.unregister("f") is False


def test_execute_wraps_tool_error():
    r = ToolRegistry()

    def boom(input):
        raise RuntimeError("内部错误")

    r.register_function("boom", "d", boom)
    with pytest.raises(ToolError):
        r.execute("boom", {"input": "x"})


def test_get_tools_description_and_schema():
    r = ToolRegistry()
    r.register_tool(AddTool())
    desc = r.get_tools_description()
    assert "add" in desc and "相加" in desc
    schema = r.to_openai_schema()
    assert schema[0]["function"]["name"] == "add"
    assert "a" in schema[0]["function"]["parameters"]["properties"]


def test_thread_safe_concurrent_register():
    r = ToolRegistry()

    def register(n):
        for i in range(50):
            r.register_function(f"t{n}_{i}", "d", lambda input: input)

    threads = [threading.Thread(target=register, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(r) == 200


def test_validate_parameters():
    t = AddTool()
    assert t.validate_parameters({"a": 1}) == ["b"]
    assert t.validate_parameters({"a": 1, "b": 2}) == []
