"""异常体系：分层，便于上层按类型捕获并决定降级策略。"""


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
    """文档加载失败（格式不支持、解析为空、来源不安全等）。"""


class MemoryError(AgentFlowError):
    """记忆层读写失败。"""
