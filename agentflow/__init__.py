"""agentflow —— 自研智能体框架 v2.0。

在 v1.0 的三范式 / 低代码工作流 / 可替换 Model Layer / 工具热插拔 /
Context-Memory / Trace 之上，新增：
- context    上下文工程：GSSC、六分区、预算压缩（ContextBuilder）
- notes      项目式笔记：Markdown + front matter + 索引（NoteStore / NoteTool）
- protocols  MCP 工具接入、A2A 协作（最终都回归 Tool）
- evaluation 数据集、AST/QEM/Judge、对比报告

分层：
- core       Message / Config / Agent 基类 / exceptions
- llm        可替换 Model Layer（公司中转站，OpenAI 兼容，换 model 即切换）
- tools      Tool / ToolRegistry（热插拔）/ ToolChain / 异步执行器 / builtin
- agents     三种范式实现
- memory     Context/Memory 管理（历史 + 可插拔持久化后端）
- retrieval  文档加载 / 分块 / 向量库 / 混合检索 / RAG
- context    上下文管线
- notes      项目式笔记
- protocols  MCP / A2A
- evaluation 评估闭环
- trace      全链路 Trace 回调
- workflow   低代码工作流（自研轻量 StateGraph）
- utils      通用工具
"""

from agentflow.version import __version__

__all__ = ["__version__"]
