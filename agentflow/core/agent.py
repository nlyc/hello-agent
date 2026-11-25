"""Agent 基类：模板方法——run() 公开、_execute() 抽象，强制统一执行入口。"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from agentflow.core.config import Config
from agentflow.core.message import Message
from agentflow.llm.base import BaseLLM

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型标注
    from agentflow.memory.context import MemoryContext


def memory_text(memory_context: "MemoryContext | None") -> str:
    """提取上下文中的可注入文本；为空或不存在时返回空串。

    优先使用 ``prompt_text``（BuiltContext 去掉了 Task 分区，避免与
    Agent 自己的输入重复）；旧 ``MemoryContext`` 回落到 ``text``。
    """
    if memory_context is None:
        return ""
    if hasattr(memory_context, "prompt_text"):
        return str(getattr(memory_context, "prompt_text", "") or "").strip()
    return str(getattr(memory_context, "text", "") or "").strip()


def prepend_memory_text(text: str, memory_context: "MemoryContext | None") -> str:
    """把记忆文本前置到任务文本前；无记忆时原样返回。"""
    block = memory_text(memory_context)
    if not block:
        return text
    return f"{block}\n\n{text}"


class Agent(ABC):
    """所有范式（ReAct / Reflection / PlanAndSolve）的基类。

    构造注入 llm（可替换）、memory（长期记忆）、tracer（全链路追踪），均可缺省。
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        memory=None,
        tracer=None,
        memory_enabled: Optional[bool] = None,
        context_builder=None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self.memory = memory
        self.tracer = tracer
        self._memory_enabled_override = memory_enabled
        self.context_builder = context_builder
        self._history: list[Message] = []

    @property
    def memory_enabled(self) -> bool:
        """记忆开关：显式传参优先，否则取 ``config.memory.enabled``。"""
        if self._memory_enabled_override is not None:
            return bool(self._memory_enabled_override)
        return bool(getattr(self.config.memory, "enabled", True))

    def run(self, input_text: str, **kwargs) -> str:
        """模板方法：构建上下文 → 子类执行范式循环 → 写回记忆并记录助手消息。"""
        self._emit("start", data={"input": input_text})
        self._history.append(Message(role="user", content=input_text))
        context = self._prepare_context(input_text)
        kwargs["memory_context"] = context
        kwargs["context"] = context
        try:
            result = self._execute(input_text, **kwargs)
        except Exception as e:
            # 失败轮次不应在历史里留下无回复的 user 消息，避免下一轮
            # ContextBuilder 把它当成真实对话注入。
            if (
                self._history
                and self._history[-1].role == "user"
                and self._history[-1].content == input_text
            ):
                self._history.pop()
            self._emit("error", data={"error": str(e)})
            raise
        self._history.append(Message(role="assistant", content=result))
        self._write_memory(input_text, result)
        self._emit("end", data={"result": result})
        return result

    def _prepare_context(self, input_text: str):
        """优先构建 v2.0 上下文；未注入 builder 时保持 v1.0 记忆路径。"""
        if self.context_builder is None:
            return self._prepare_memory_context(input_text)
        from agentflow.memory.context import MemoryContext

        try:
            # 当前用户消息已追加到 _history，避免上下文里再重复一次。
            context = self.context_builder.build(
                input_text, history=self._history[:-1]
            )
        except Exception as exc:
            self._emit("context.error", data={"error": str(exc), "phase": "build"})
            return MemoryContext(enabled=False)
        if context is None:
            self._emit("context.error", data={"error": "builder 返回空", "phase": "build"})
            return MemoryContext(enabled=False)
        stats = dict(getattr(context, "stats", {}) or {})
        self._emit(
            "context.build",
            data={
                "provider": type(self.context_builder).__name__,
                "tokens": stats.get("total_tokens"),
                "sections": stats.get("sections"),
                "packets_selected": stats.get("packets_selected"),
                "packets_dropped_relevance": stats.get(
                    "packets_dropped_relevance"
                ),
                "packets_dropped_budget": stats.get("packets_dropped_budget"),
                "source_errors": stats.get("source_errors"),
            },
        )
        return context

    def _prepare_memory_context(self, input_text: str) -> "MemoryContext":
        """构建本轮记忆上下文；失败时发 trace 并降级为空上下文。"""
        from agentflow.memory.context import MemoryContext

        if not self.memory_enabled or self.memory is None:
            return MemoryContext(enabled=self.memory_enabled)

        try:
            builder = getattr(self.memory, "build_context", None)
            if callable(builder):
                context = builder(input_text)
                if context is None:
                    context = MemoryContext()
            else:
                # 兼容 v1.0 的 BaseMemory：只有 recall(query) -> str
                recall = getattr(self.memory, "recall", None)
                if not callable(recall):
                    return MemoryContext()
                context = MemoryContext(text=str(recall(input_text) or ""))
        except Exception as e:
            self._emit("memory.error", data={"error": str(e), "phase": "build"})
            return MemoryContext(enabled=False)

        self._emit(
            "memory.build",
            data={
                "text": memory_text(context),
                "items": len(getattr(context, "items", []) or []),
                "working_items": len(getattr(context, "working_items", []) or []),
                "long_term_hits": len(getattr(context, "long_term_hits", []) or []),
            },
        )
        return context

    def _write_memory(self, input_text: str, result: str) -> None:
        """成功执行后写回记忆；旧 BaseMemory 无 record_turn 时保持 v1.0 行为。"""
        if not self.memory_enabled or self.memory is None:
            return
        recorder = getattr(self.memory, "record_turn", None)
        if not callable(recorder):
            return
        try:
            recorder(input_text, result)
        except Exception as e:
            self._emit("memory.error", data={"error": str(e), "phase": "write"})
            return
        self._emit("memory.write", data={"input": input_text, "result": result})

    @abstractmethod
    def _execute(self, input_text: str, **kwargs) -> str:
        """子类实现具体的范式循环。"""
        ...

    def _emit(self, event_type: str, step: int = 0, data: Optional[dict] = None) -> None:
        """有 tracer 时发一条 trace 事件；无则静默。"""
        if self.tracer is not None:
            self.tracer.emit(event_type, agent_name=self.name, step=step, data=data)

    def _call_llm(self, messages, step: int = 0, **kwargs) -> str:
        """带 trace 的 LLM 调用：结果原样返回，同时 emit 一条 llm 事件。"""
        result = self.llm.chat(messages, **kwargs)
        self._emit("llm", step=step, data={"output": result})
        return result

    def add_message(self, message: Message) -> None:
        self._history.append(message)

    def clear_history(self) -> None:
        self._history.clear()

    def get_history(self) -> list[Message]:
        """返回历史副本，防止外部篡改内部状态。"""
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()
