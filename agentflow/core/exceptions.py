"""异常体系：分层，便于上层按类型捕获并决定降级策略。"""

from __future__ import annotations

from typing import Any


class AgentFlowError(Exception):
    """agentflow 框架所有异常的基类。"""


class ConfigError(AgentFlowError):
    """配置缺失或非法。"""


class LLMError(AgentFlowError):
    """模型调用失败（网络、鉴权、空响应等）。"""


class ToolError(AgentFlowError):
    """工具执行失败。"""


class AgentError(AgentFlowError):
    """Agent 执行失败（解析、超步数等）。"""


class WorkflowError(AgentFlowError):
    """工作流（图）执行失败。"""


class RAGError(AgentFlowError):
    """RAG / 检索层通用异常。"""


class RetrievalError(RAGError):
    """检索过程失败（向量维度非法、索引状态异常等）。"""


class EmbeddingMismatchError(RAGError):
    """向量指纹或维度不一致，需要重建索引。"""


class DocumentLoadError(RAGError):
    """文档加载失败（格式不支持、解析为空、来源不安全等）。

    ``metadata`` 可携带机器可读的失败原因，例如扫描版 PDF 的
    ``{"needs_ocr": True}``。
    """

    def __init__(self, message: str, *, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metadata: dict[str, Any] = dict(metadata or {})


class MemoryError(AgentFlowError):
    """记忆层读写失败。"""


class ContextError(AgentFlowError):
    """上下文管线构建失败（配置非法、分区非法、预算无法满足等）。"""


class NoteError(AgentFlowError):
    """笔记读写失败（ID 非法、文件损坏、类型非法等）。"""


class MCPError(AgentFlowError):
    """MCP 协议错误（连接、超时、JSON-RPC 错误、工具调用失败等）。"""
