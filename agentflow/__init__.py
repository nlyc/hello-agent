"""agentflow —— 自研智能体框架 v1.0。

统一承载三种 Agent 范式（ReAct / Reflection / Plan-and-Solve）、低代码工作流、
可替换 Model Layer，并提供工具热插拔、Context/Memory 管理与全链路 Trace 回调。

分层：
- core      Message / Config / Agent 基类 / exceptions
- llm       可替换 Model Layer（公司中转站，OpenAI 兼容，换 model 即切换）
- tools     Tool / ToolRegistry（热插拔）/ ToolChain / 异步执行器 / builtin
- agents    三种范式实现
- memory    Context/Memory 管理（历史 + 可插拔持久化后端）
- trace     全链路 Trace 回调
- workflow  低代码工作流（自研轻量 StateGraph）
- utils     通用工具
"""

from agentflow.version import __version__

__all__ = ["__version__"]
