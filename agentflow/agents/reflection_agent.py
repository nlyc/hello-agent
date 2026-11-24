"""Reflection 智能体：执行 → 反思 → 优化循环，「无需改进」+ 轮次上限双终止。"""

from typing import Optional

from agentflow.core.agent import Agent, prepend_memory_text

INITIAL_TEMPLATE = """你是一位{role}。请根据以下要求完成任务。

要求: {task}

请直接输出结果，不要包含任何额外的解释。"""

REFLECT_TEMPLATE = """你是一位{reviewer_role}。
你的任务是审查以下结果，找出其中的事实性错误、逻辑漏洞、效率问题或遗漏信息。

# 原始任务:
{task}

# 待审查的结果:
{result}

请分析该结果是否存在明显不足，并提出具体的、可操作的改进建议。
只有在结果已经达到最优、确实没有可改进之处时，才能回答"无需改进"。

请直接输出你的反馈，不要包含任何额外的解释。"""

REFINE_TEMPLATE = """你是一位{role}。你正在根据评审专家的反馈优化你的结果。

# 原始任务:
{task}

# 你上一轮的结果:
{last_result}

# 评审员反馈:
{feedback}

请根据反馈生成一个优化后的新版本结果。
请直接输出优化后的结果，不要包含任何额外的解释。"""


class ReflectionAgent(Agent):
    """Reflection 范式：事后自校循环，以成本换质量。"""

    def __init__(
        self,
        name: str,
        llm,
        system_prompt: Optional[str] = None,
        config=None,
        max_iterations: int = 3,
        role: str = "资深专家",
        reviewer_role: str = "极其严格的评审专家",
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
        self.max_iterations = max_iterations
        self.role = role
        self.reviewer_role = reviewer_role
        self._templates = {
            "initial": INITIAL_TEMPLATE,
            "reflect": REFLECT_TEMPLATE,
            "refine": REFINE_TEMPLATE,
        }
        if custom_prompts:
            self._templates.update(custom_prompts)
        self._records: list[dict] = []

    def _chat(self, prompt: str, step: int = 0) -> str:
        return self._call_llm([{"role": "user", "content": prompt}], step=step).strip()

    def _execute(self, input_text: str, memory_context=None, **kwargs) -> str:
        self._records = []
        # 记忆文本前置到任务，initial / reflect / refine 三段共享同一份任务
        task = prepend_memory_text(input_text, memory_context)
        step = 1
        result = self._chat(
            self._templates["initial"].format(role=self.role, task=task), step=step
        )
        self._records.append({"type": "execution", "content": result})

        for _ in range(self.max_iterations):
            step += 1
            feedback = self._chat(
                self._templates["reflect"].format(
                    reviewer_role=self.reviewer_role, task=task, result=result
                ),
                step=step,
            )
            self._records.append({"type": "reflection", "content": feedback})
            if "无需改进" in feedback:
                break
            step += 1
            result = self._chat(
                self._templates["refine"].format(
                    role=self.role, task=task, last_result=result, feedback=feedback
                ),
                step=step,
            )
            self._records.append({"type": "execution", "content": result})
        return result

    def get_trajectory(self) -> str:
        """把执行/反思轨迹序列化成文本，供提示词注入或轨迹分析。"""
        parts = []
        for r in self._records:
            if r["type"] == "execution":
                parts.append(f"--- 尝试 ---\n{r['content']}")
            else:
                parts.append(f"--- 反馈 ---\n{r['content']}")
        return "\n\n".join(parts)
