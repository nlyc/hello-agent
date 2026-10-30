"""PlanAndSolveAgent（规划列表 + 执行状态管理）测试。"""

import pytest

from agentflow.agents import PlanAndSolveAgent
from agentflow.core.exceptions import AgentError
from tests.fakes import ScriptedLLM


def test_plan_and_solve():
    llm = ScriptedLLM(
        [
            '```python\n["第一步", "第二步"]\n```',  # planner
            "结果1",  # executor 步骤 1
            "结果2",  # executor 步骤 2
        ]
    )
    agent = PlanAndSolveAgent("solver", llm)
    result = agent.run("一个问题")
    assert result == "结果2"
    assert len(llm.calls) == 3
    # 状态管理：第二步的 prompt 必须带上第一步的结果
    step2_prompt = llm.calls[2]["messages"][0]["content"]
    assert "步骤 1" in step2_prompt and "结果1" in step2_prompt


def test_plan_invalid_raises():
    llm = ScriptedLLM(["这不是列表"])
    agent = PlanAndSolveAgent("solver", llm)
    with pytest.raises(AgentError):
        agent.run("x")
