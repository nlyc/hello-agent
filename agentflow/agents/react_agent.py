"""ReAct 智能体：Thought → Action → Observation 循环，Finish 先于工具查找。"""

import re
from typing import Optional

from agentflow.core.agent import Agent, memory_text
from agentflow.core.exceptions import AgentError, ToolError
from agentflow.tools.registry import ToolRegistry

REACT_SYSTEM_PROMPT = """你是一个{role}。

# 可用工具:
{tools}

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对 Thought 和 Action：
Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action 的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对 Thought-Action，除 Thought 与 Action 外不要输出任何其他内容
- Action 必须在同一行，不要换行
- 只调用完成当前问题所必需的最少工具；不要为了“更全面”额外调用无关工具
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束
- Finish 方括号内是完整的最终答案，可以分点、可以换行，但必须完整地放在方括号之内"""


def _truncate_extra_pairs(output: str) -> str:
    """模型可能连写多对 Thought-Action，只保留第一对。

    例外：Action 行为空（调用写在下一行）或第一个 Action 是 Finish——
    Finish 的答案是一段可跨行的长文本，按行截断会丢右方括号。
    """
    match = re.search(r"Action[:：]", output)
    if not match:
        return output
    newline = output.find("\n", match.start())
    if newline == -1:
        return output
    rest = output[match.end() : newline].strip()
    if not rest or rest.upper().startswith("FINISH"):
        return output
    return output[:newline].strip()


def _extract_action(output: str) -> Optional[str]:
    m = re.search(r"Action[:：]\s*(.*)", output, re.DOTALL)
    if not m or not m.group(1).strip():
        return None
    return m.group(1).strip()


def _extract_finish(action_str: str) -> Optional[str]:
    m = re.search(r"FINISH\s*[:：]?\s*[\[【](.*)[\]】]", action_str, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


def _parse_tool_call(action_str: str) -> tuple[Optional[str], dict]:
    tool_match = re.search(r"(\w+)\(", action_str)
    args_match = re.search(r"\((.*)\)", action_str, re.DOTALL)
    if not tool_match or not args_match:
        return None, {}
    tool_name = tool_match.group(1)
    args_str = args_match.group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
    if not kwargs:  # 容忍单引号写法
        kwargs = dict(re.findall(r"(\w+)='([^']*)'", args_str))
    return tool_name, kwargs


def _memory_items_message(memory_context) -> str:
    """把召回条目拼成一条额外 user 消息；无条目时返回空串。

    ``BuiltContext`` 的 ``text`` 已包含全部条目，再插一条会重复，
    因此直接跳过。
    """
    if getattr(memory_context, "kind", "") == "built":
        return ""
    items = getattr(memory_context, "items", None) or []
    lines: list[str] = []
    for entry in items:
        if isinstance(entry, dict):
            content = str(entry.get("content", "")).strip()
        else:
            content = str(getattr(entry, "content", "") or "").strip()
        if content:
            lines.append(f"- {content}")
    if not lines:
        return ""
    return "相关记忆：\n" + "\n".join(lines)


class ReActAgent(Agent):
    """ReAct 范式：走一步看一步，工具走统一 Registry。"""

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
        tools_desc = self.tools.get_tools_description() if self.tools else "（无可用工具）"
        role = self.system_prompt or "智能助手"
        return REACT_SYSTEM_PROMPT.format(role=role, tools=tools_desc)

    def _execute(self, input_text: str, memory_context=None, **kwargs) -> str:
        system_prompt = self._build_system_prompt()
        block = memory_text(memory_context)
        if block:
            system_prompt = f"{system_prompt}\n\n{block}"

        messages = [{"role": "system", "content": system_prompt}]
        memory_message = _memory_items_message(memory_context)
        if memory_message:
            messages.append({"role": "user", "content": memory_message})
        messages.append({"role": "user", "content": input_text})
        max_steps = self.config.agent.max_steps
        for step_no in range(1, max_steps + 1):
            output = self._call_llm(messages, step=step_no).strip()
            output = _truncate_extra_pairs(output)
            messages.append({"role": "assistant", "content": output})

            action_str = _extract_action(output)
            if action_str is None:
                messages.append(
                    {
                        "role": "user",
                        "content": "Observation: 错误: 未能解析到 Action 字段，请严格遵循 'Thought: ...' 与 'Action: ...' 的格式。",
                    }
                )
                continue

            # Finish 检查必须先于工具查找，否则最终答案里的工具名会被误判为工具调用
            if action_str.upper().startswith("FINISH"):
                finish = _extract_finish(action_str)
                if finish is not None:
                    return finish
                messages.append(
                    {
                        "role": "user",
                        "content": "Observation: 错误: Finish 格式应为 Finish[最终答案]，把完整答案放进方括号内。",
                    }
                )
                continue

            tool_name, params = _parse_tool_call(action_str)
            if tool_name is None:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Observation: 错误: 无法解析工具调用 '{action_str}'，格式应为 function_name(arg_name=\"arg_value\")。",
                    }
                )
                continue
            if self.tools is None or tool_name not in self.tools:
                messages.append({"role": "user", "content": f"Observation: 错误: 未定义的工具 '{tool_name}'。"})
                continue

            try:
                observation = self.tools.execute(tool_name, params)
            except ToolError as e:
                observation = f"错误: {e}"
            self._emit(
                "tool",
                step=step_no,
                data={"tool": tool_name, "params": params, "output": observation},
            )
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        raise AgentError(f"ReAct 已达最大步数 {max_steps} 仍未得到最终答案")
