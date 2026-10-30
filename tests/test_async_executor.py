"""AsyncToolExecutor（线程池并行执行）测试。"""

from agentflow.tools import AsyncToolExecutor, Tool


class DoubleTool(Tool):
    def __init__(self):
        super().__init__(name="double", description="")

    def run(self, parameters):
        return str(int(parameters["input"]) * 2)


def test_run_parallel():
    tool = DoubleTool()
    calls = [(tool, {"input": "1"}), (tool, {"input": "2"}), (tool, {"input": "3"})]
    with AsyncToolExecutor(max_workers=3) as ex:
        results = ex.run_parallel_sync(calls)
    assert results == ["2", "4", "6"]
