"""Plan-and-Solve 智能体：先规划（强制 Python 列表），再逐步执行（状态管理）。"""

import ast
from typing import Optional

from agentflow.core.agent import Agent, prepend_memory_text
from agentflow.core.exceptions import AgentError

PLANNER_TEMPLATE = """你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划，```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```"""

EXECUTOR_TEMPLATE = """你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:"""


class PlanAndSolveAgent(Agent):
    """Plan-and-Solve 范式：三思而后行，计划静态、执行时靠 history 做状态管理。"""

    def __init__(
        self,
        name: str,
        llm,
        system_prompt: Optional[str] = None,
        config=None,
        custom_prompts: Optional[dict] = None,
        memory=None,
        tracer=None,
        memory_enabled: Optional[bool] = None,
        context_builder=None,
    ):
        super().__init__(
            name,
            llm,
            system_prompt,
            config,
            memory,
            tracer,
            memory_enabled,
            context_builder=context_builder,
        )
        self._templates = {"planner": PLANNER_TEMPLATE, "executor": EXECUTOR_TEMPLATE}
        if custom_prompts:
            self._templates.update(custom_prompts)

    def _chat(self, prompt: str, step: int = 0) -> str:
        return self._call_llm([{"role": "user", "content": prompt}], step=step).strip()

    def _execute(self, input_text: str, memory_context=None, **kwargs) -> str:
        # 记忆文本同时前置到 planner 与 executor 的 question
        question = prepend_memory_text(input_text, memory_context)
        plan = self._plan(question)
        if not plan:
            raise AgentError("规划器未能生成有效计划")
        return self._execute_plan(question, plan)

    def _plan(self, question: str) -> list[str]:
        prompt = self._templates["planner"].format(question=question)
        response = self._chat(prompt, step=1)
        try:
            plan_str = response.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            raise AgentError(f"解析计划失败: {e}") from e

    def _execute_plan(self, question: str, plan: list[str]) -> str:
        history = ""
        result = ""
        for i, step in enumerate(plan):
            prompt = self._templates["executor"].format(
                question=question, plan=plan, history=history or "无", current_step=step
            )
            result = self._chat(prompt, step=i + 2)
            history += f"步骤 {i+1}: {step}\n结果: {result}\n\n"
        return result
