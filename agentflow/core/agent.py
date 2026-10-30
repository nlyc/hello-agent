"""Agent 基类：模板方法——run() 公开、_execute() 抽象，强制统一执行入口。"""

from abc import ABC, abstractmethod
from typing import Optional

from agentflow.core.config import Config
from agentflow.core.message import Message
from agentflow.llm.base import BaseLLM


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
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self.memory = memory
        self.tracer = tracer
        self._history: list[Message] = []

    def run(self, input_text: str, **kwargs) -> str:
        """模板方法：记录用户消息 → 子类执行范式循环 → 记录助手消息。"""
        self._emit("start", data={"input": input_text})
        self._history.append(Message(role="user", content=input_text))
        try:
            result = self._execute(input_text, **kwargs)
        except Exception as e:
            self._emit("error", data={"error": str(e)})
            raise
        self._history.append(Message(role="assistant", content=result))
        self._emit("end", data={"result": result})
        return result

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
