"""搜索工具：多源降级链（最优源 → 备选 → 明确报错，不静默失败）。"""

import os
from typing import Any, Callable, Optional, Sequence

from agentflow.core.exceptions import ToolError
from agentflow.tools.base import Tool, ToolParameter


def _format_tavily(response: dict) -> str:
    lines = [f"搜索答案：{response.get('answer') or '未找到直接答案'}"]
    for i, item in enumerate(response.get("results", [])[:3], 1):
        lines.append(f"[{i}] {item.get('title', '')} — {item.get('url', '')}")
        lines.append(f"    {(item.get('content', '') or '')[:200]}")
    return "\n".join(lines)


def _format_serpapi(response: dict) -> str:
    lines = []
    answer_box = response.get("answer_box", {})
    if answer_box.get("answer"):
        lines.append(f"直接答案：{answer_box['answer']}")
    kg = response.get("knowledge_graph", {})
    if kg.get("description"):
        lines.append(f"知识图谱：{kg['description']}")
    for i, res in enumerate(response.get("organic_results", [])[:3], 1):
        lines.append(f"[{i}] {res.get('title', '')} — {res.get('link', '')}")
        lines.append(f"    {(res.get('snippet', '') or '')[:200]}")
    return "\n".join(lines) if lines else "未找到相关信息"


class SearchTool(Tool):
    """多源搜索：按后端优先级降级，全部失败则明确报错。"""

    def __init__(self, backends: Optional[Sequence[tuple[str, Callable[[str], str]]]] = None):
        super().__init__(
            name="search",
            description="网页搜索，按可用性自动选择后端。回答需要实时信息的问题时使用。",
        )
        self._backends = list(backends) if backends is not None else self._default_backends()

    def _default_backends(self) -> list[tuple[str, Callable[[str], str]]]:
        backends: list[tuple[str, Callable[[str], str]]] = []
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            try:
                from tavily import TavilyClient

                client = TavilyClient(api_key=tavily_key)
                backends.append(
                    (
                        "tavily",
                        lambda q: _format_tavily(
                            client.search(query=q, search_depth="basic", include_answer=True, max_results=3)
                        ),
                    )
                )
            except ImportError:
                pass
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        if serpapi_key:
            try:
                from serpapi import SerpApiClient

                backends.append(
                    (
                        "serpapi",
                        lambda q: _format_serpapi(
                            SerpApiClient({"engine": "google", "q": q, "api_key": serpapi_key}).get_dict()
                        ),
                    )
                )
            except ImportError:
                pass
        return backends

    def run(self, parameters: dict[str, Any]) -> str:
        query = str(parameters.get("input", "") or parameters.get("query", "")).strip()
        if not query:
            raise ToolError("搜索查询不能为空")
        if not self._backends:
            raise ToolError("未配置任何搜索后端（请设置 TAVILY_API_KEY 或 SERPAPI_API_KEY）")
        errors = []
        for name, backend in self._backends:
            try:
                return backend(query)
            except Exception as e:
                errors.append(f"{name}: {e}")
        raise ToolError("所有搜索后端均失败：" + "；".join(errors))

    def get_parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="input", type="string", description="搜索关键词", required=True)]
