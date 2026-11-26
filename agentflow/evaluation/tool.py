"""EvaluationTool：把评估运行器包装成 Agent 可调用的工具。"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from agentflow.core.exceptions import EvaluationError
from agentflow.evaluation.report import render_report
from agentflow.evaluation.types import EvalReport
from agentflow.tools.base import Tool, ToolParameter

_ACTIONS = ("list", "run", "report")


class EvaluationTool(Tool):
    """评估工具。

    参数：
        runner: ``runner(suite, limit, category) -> EvalReport``。
        suites: 可用基准名列表。
        name: 工具名。
    """

    def __init__(
        self,
        runner: Callable[..., EvalReport],
        suites: Sequence[str] = ("tool_calling", "qa", "quality"),
        *,
        name: str = "evaluation",
    ) -> None:
        super().__init__(
            name=name,
            description=(
                "运行 v2.0 评估基准。action=list 查看基准；"
                "action=run 运行并返回汇总；action=report 返回 Markdown。"
            ),
        )
        self.runner = runner
        self.suites = list(suites)
        self.last_report: EvalReport | None = None

    def run(self, parameters: dict[str, Any]) -> str:
        params = parameters or {}
        action = str(params.get("action", "list")).strip().lower()
        if action not in _ACTIONS:
            return f"错误: 未知 action '{action}'，可用：{' / '.join(_ACTIONS)}"
        if action == "list":
            return "可用基准：" + " / ".join(self.suites)
        suite = str(params.get("suite", "tool_calling")).strip()
        if suite not in self.suites:
            return f"错误: 未知 suite '{suite}'，可用：{' / '.join(self.suites)}"
        try:
            limit = int(params.get("limit", 5))
            limit = max(1, limit)
        except (TypeError, ValueError):
            return "错误: limit 必须是正整数"
        category = params.get("category")
        try:
            report = self.runner(
                suite,
                limit,
                str(category).strip() if category else None,
            )
        except EvaluationError as exc:
            return f"错误: {exc}"
        except Exception as exc:
            return f"错误: 评估失败：{exc}"
        self.last_report = report
        if action == "report":
            return render_report(report)
        return json.dumps(report.metrics, ensure_ascii=False, indent=2)

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("action", "string", " / ".join(_ACTIONS), True),
            ToolParameter("suite", "string", "tool_calling / qa / quality"),
            ToolParameter("limit", "integer", "样本上限，默认 5"),
            ToolParameter("category", "string", "按类别过滤（可选）"),
        ]
