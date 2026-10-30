"""ToolChain（context 模板传值）测试。"""

import pytest

from agentflow.core.exceptions import ToolError
from agentflow.tools import Tool, ToolChain


class EchoTool(Tool):
    def __init__(self, name):
        super().__init__(name=name, description="")

    def run(self, parameters):
        return f"[{self.name}]" + parameters.get("input", "")


def test_chain_passes_context():
    chain = ToolChain("demo")
    chain.add_step(EchoTool("first"), "{input}", "a").add_step(EchoTool("second"), "{a}", "b")
    ctx = chain.run("hello")
    assert ctx["a"] == "[first]hello"
    assert ctx["b"] == "[second][first]hello"
    assert chain.steps == 2


def test_chain_missing_variable_raises():
    chain = ToolChain("demo").add_step(EchoTool("x"), "{nope}", "out")
    with pytest.raises(ToolError):
        chain.run("hi")


def test_chain_manager():
    from agentflow.tools import ToolChainManager

    manager = ToolChainManager()
    manager.register(ToolChain("c1").add_step(EchoTool("e"), "{input}", "o"))
    assert manager.list_chains() == ["c1"]
    assert manager.execute("c1", "hi")["o"] == "[e]hi"
    with pytest.raises(ToolError):
        manager.execute("missing", "hi")
