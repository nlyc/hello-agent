"""配置管理：分组 + 默认值（零配置可跑）+ from_env() 从 .env 读取。

所有配置项统一经 ``agentflow.core.env`` 读取：优先项目根目录 ``.env``，
其次进程环境变量。业务代码请使用 ``Config.from_env()`` 或
``agentflow.core.env.get_env()``，不要直接 ``os.getenv``。
"""

from pydantic import BaseModel, Field

from agentflow.core.env import env_bool, env_float, env_int, get_env


class LLMConfig(BaseModel):
    """Model Layer 配置。base_url 指向公司中转站（OpenAI 兼容网关）。"""

    base_url: str = ""
    api_key: str = ""
    model: str = "deepseek-chat"
    temperature: float = 0.2
    timeout: float = 60.0


class AgentConfig(BaseModel):
    """Agent 运行参数。"""

    max_steps: int = 6
    max_tool_iterations: int = 3


class MemoryConfig(BaseModel):
    """记忆层配置。默认开启，使用本地 SQLite，零额外依赖。"""

    enabled: bool = True
    store: str = "sqlite"
    sqlite_path: str = ".data/memory.sqlite3"
    working_max_items: int = 50
    working_ttl_minutes: int = 60
    top_k: int = 5
    user_id: str = "default"


class RetrievalConfig(BaseModel):
    """检索层配置：向量后端、模型缓存与向量库选择。"""

    embed_backend: str = "hashing"  # hashing | fastembed
    fastembed_model: str = "BAAI/bge-small-zh-v1.5"
    fastembed_cache_path: str = ""
    hf_endpoint: str = "https://hf-mirror.com"
    vector_store: str = "sqlite"  # sqlite | qdrant
    sqlite_path: str = ".data/rag.sqlite3"


class QdrantConfig(BaseModel):
    """远程 Qdrant 配置。url 为空表示不启用远程向量库。"""

    url: str = ""
    api_key: str = ""
    collection: str = "agentflow"
    timeout: float = 30.0
    prefer_grpc: bool = False


class SearchConfig(BaseModel):
    """联网网页搜索配置：按后端优先级自动降级。"""

    tavily_api_key: str = ""
    serpapi_api_key: str = ""
    max_results: int = 3


class Config(BaseModel):
    """框架总配置，零配置即可实例化运行。"""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    max_history_length: int = 100
    debug: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        """从 .env / 环境变量读取（12-Factor：改配置不改代码）。

        优先级：Shell 环境变量 > 项目 .env > 代码默认值；
        模型等参数仍可被显式传参覆盖（见 OpenAICompatibleLLM）。
        """
        llm = LLMConfig(
            base_url=get_env("LLM_BASE_URL"),
            api_key=get_env("LLM_API_KEY"),
            model=get_env("LLM_MODEL", "deepseek-chat"),
            temperature=env_float("LLM_TEMPERATURE", 0.2),
            timeout=env_float("LLM_TIMEOUT", 60.0),
        )
        agent = AgentConfig(
            max_steps=env_int("AGENT_MAX_STEPS", 6),
            max_tool_iterations=env_int("AGENT_MAX_TOOL_ITERATIONS", 3),
        )
        memory = MemoryConfig(
            enabled=env_bool("MEMORY_ENABLED", True),
            store=get_env("MEMORY_STORE", "sqlite"),
            sqlite_path=get_env("MEMORY_SQLITE_PATH", ".data/memory.sqlite3"),
            working_max_items=env_int("MEMORY_WORKING_MAX_ITEMS", 50),
            working_ttl_minutes=env_int("MEMORY_WORKING_TTL_MINUTES", 60),
            top_k=env_int("MEMORY_TOP_K", 5),
            user_id=get_env("USER_ID", "default"),
        )
        retrieval = RetrievalConfig(
            embed_backend=get_env("EMBED_BACKEND", "hashing"),
            fastembed_model=get_env("FASTEMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
            fastembed_cache_path=get_env("FASTEMBED_CACHE_PATH"),
            hf_endpoint=get_env("HF_ENDPOINT", "https://hf-mirror.com"),
            vector_store=get_env("VECTOR_STORE", "sqlite"),
            sqlite_path=get_env("RAG_SQLITE_PATH", ".data/rag.sqlite3"),
        )
        qdrant = QdrantConfig(
            url=get_env("QDRANT_URL"),
            api_key=get_env("QDRANT_API_KEY"),
            collection=get_env("QDRANT_COLLECTION", "agentflow"),
            timeout=env_float("QDRANT_TIMEOUT", 30.0),
            prefer_grpc=env_bool("QDRANT_PREFER_GRPC", False),
        )
        search = SearchConfig(
            tavily_api_key=get_env("TAVILY_API_KEY"),
            serpapi_api_key=get_env("SERPAPI_API_KEY"),
            max_results=env_int("SEARCH_MAX_RESULTS", 3),
        )
        return cls(
            llm=llm,
            agent=agent,
            memory=memory,
            retrieval=retrieval,
            qdrant=qdrant,
            search=search,
            max_history_length=env_int("MAX_HISTORY_LENGTH", 100),
            debug=env_bool("DEBUG", False),
        )


def _env_bool(name: str, default: bool) -> bool:
    """兼容旧调用：读取布尔环境变量。"""
    return env_bool(name, default)
