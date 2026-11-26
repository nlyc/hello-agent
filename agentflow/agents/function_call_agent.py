"""FunctionCallAgent：原生 Function Calling 范式。

与 ReAct 文本协议的区别：
- 工具 schema 由 ``ToolRegistry.to_openai_schema()`` 生成，直接交给模型；
- 模型返回结构化 ``tool_calls``，无需正则解析；
- 一次响应可并行返回多个工具调用。

用于评估时更贴近 BFCL 的 multiple / parallel 口径。
"""

from __future__ import annotations

from typing import Optional

from agentflow.core.agent import Agent, memory_text
from agentflow.core.exceptions import AgentError, ToolError
from agentflow.llm.base import LLMResponse
from agentflow.tools.registry import ToolRegistry

FUNCTION_CALL_SYSTEM_PROMPT = """你是一个{role}。

你可以使用提供的工具获取事实。规则：
1. 只调用完成当前问题所必需的最少工具；不要为了“更全面”额外调用无关工具。
2. 只调用用户问题直接需要的工具类别；不要主动查询用户没问的事实（例如没问预约就不要查笔记）。
3. 需要外部信息时调用工具；一次可以调用一个或多个工具。
4. 并行调用只用于多个不同对象（例如同时查北京和上海）；同一对象的多个查询变体只调用一次。
5. 信息足够时直接给出最终答案，不要再调用工具。
6. 不要编造工具结果；只依据工具返回内容回答。"""


class FunctionCallAgent(Agent):
    """原生 Function Calling 智能体。"""

    def __init__(
        self,
        name: str,
        llm,
        tools: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        config=None,
        custom_prompt: Optional[str] = None,
        memory=None,
        tracer=None,
        memory_enabled: Optional[bool] = None,
        context_builder=None,
        max_tool_calls: Optional[int] = None,
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
        self.tools = tools
        self.custom_prompt = custom_prompt
        self.max_tool_calls = max_tool_calls

    def _build_system_prompt(self) -> str:
        if self.custom_prompt is not None:
            return self.custom_prompt
        role = self.system_prompt or "智能助手"
        return FUNCTION_CALL_SYSTEM_PROMPT.format(role=role)

    def _execute(self, input_text: str, memory_context=None, **kwargs) -> str:
        if self.tools is None or len(self.tools) == 0:
            raise AgentError("FunctionCallAgent 需要至少一个工具")
        system_prompt = self._build_system_prompt()
        block = memory_text(memory_context)
        if block:
            system_prompt = f"{system_prompt}\n\n{block}"

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text},
        ]
        schemas = self.tools.to_openai_schema()
        max_steps = self.config.agent.max_steps
        tool_counts: dict[str, int] = {}
        total_tool_calls = 0
        for step_no in range(1, max_steps + 1):
            response = self.llm.chat_with_tools(
                messages, tools=schemas, temperature=0.0
            )
            self._emit(
                "llm",
                step=step_no,
                data={
                    "content": response.content,
                    "tool_calls": [call.name for call in response.tool_calls],
                },
            )
            if not response.tool_calls:
                answer = response.content.strip()
                if not answer:
                    raise AgentError("模型既没有调用工具，也没有返回答案")
                return answer

            messages.append(response.to_assistant_message())
            for call in response.tool_calls:
                budget_hit = False
                try:
                    tool = self.tools.get(call.name)
                except ToolError as exc:
                    observation = f"错误: {exc}"
                else:
                    cap = getattr(tool, "max_calls_per_run", None)
                    if (
                        self.max_tool_calls is not None
                        and total_tool_calls >= self.max_tool_calls
                    ):
                        observation = (
                            f"（本次任务工具调用上限 {self.max_tool_calls} 次，"
                            "已跳过该调用；请基于已有信息回答）"
                        )
                        budget_hit = True
                    elif (
                        cap is not None
                        and tool_counts.get(call.name, 0) >= cap
                    ):
                        observation = (
                            f"（工具 {call.name} 每次任务最多调用 {cap} 次，"
                            "已跳过该调用；请基于已有信息回答）"
                        )
                        budget_hit = True
                    else:
                        try:
                            observation = self.tools.execute(
                                call.name, call.arguments
                            )
                        except ToolError as exc:
                            observation = f"错误: {exc}"
                        tool_counts[call.name] = (
                            tool_counts.get(call.name, 0) + 1
                        )
                        total_tool_calls += 1
                self._emit(
                    "tool",
                    step=step_no,
                    data={
                        "tool": call.name,
                        "params": call.arguments,
                        "output": observation,
                        "call_id": call.id,
                        "budget_exhausted": budget_hit,
                    },
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": observation,
                    }
                )
        raise AgentError(
            f"FunctionCallAgent 已达最大步数 {max_steps} 仍未得到最终答案"
        )
