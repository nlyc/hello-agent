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
2. 需要外部信息时调用工具；一次可以调用一个或多个工具。
3. 信息足够时直接给出最终答案，不要再调用工具。
4. 不要编造工具结果；只依据工具返回内容回答。"""


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
                try:
                    observation = self.tools.execute(call.name, call.arguments)
                except ToolError as exc:
                    observation = f"错误: {exc}"
                self._emit(
                    "tool",
                    step=step_no,
                    data={
                        "tool": call.name,
                        "params": call.arguments,
                        "output": observation,
                        "call_id": call.id,
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
