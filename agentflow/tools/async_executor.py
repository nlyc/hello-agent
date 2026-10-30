"""异步工具执行器：ThreadPoolExecutor + asyncio.gather 并行跑多工具。"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from agentflow.tools.base import Tool


class AsyncToolExecutor:
    """把同步工具放到线程池并行执行（等待 IO 时别闲着）。"""

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def run_parallel(self, calls: Sequence[tuple[Tool, dict[str, Any]]]) -> list[str]:
        """并行执行多个 (tool, parameters)，按输入顺序返回结果。"""
        loop = asyncio.get_running_loop()

        def _run(tool: Tool, parameters: dict[str, Any]) -> str:
            return tool.run(parameters)

        futures = [
            loop.run_in_executor(self._executor, _run, tool, params) for tool, params in calls
        ]
        return await asyncio.gather(*futures)

    def run_parallel_sync(self, calls: Sequence[tuple[Tool, dict[str, Any]]]) -> list[str]:
        return asyncio.run(self.run_parallel(calls))

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "AsyncToolExecutor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
