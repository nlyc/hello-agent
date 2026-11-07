"""配置管理：分组 + 默认值（零配置可跑）+ from_env() 环境变量覆盖。"""

import os

from pydantic import BaseModel, Field


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


class Config(BaseModel):
    """框架总配置，零配置即可实例化运行。"""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    max_history_length: int = 100
    debug: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量读取（12-Factor：改环境不改代码）。

        优先级：环境变量 > 默认值；模型还可被显式传参覆盖（见 OpenAICompatibleLLM）。
        """
        llm = LLMConfig(
            base_url=os.getenv("LLM_BASE_URL", ""),
            api_key=os.getenv("LLM_API_KEY", ""),
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),
        )
        agent = AgentConfig(
            max_steps=int(os.getenv("AGENT_MAX_STEPS", "6")),
            max_tool_iterations=int(os.getenv("AGENT_MAX_TOOL_ITERATIONS", "3")),
        )
        memory = MemoryConfig(
            enabled=_env_bool("MEMORY_ENABLED", True),
            store=os.getenv("MEMORY_STORE", "sqlite"),
            sqlite_path=os.getenv("MEMORY_SQLITE_PATH", ".data/memory.sqlite3"),
            working_max_items=int(os.getenv("MEMORY_WORKING_MAX_ITEMS", "50")),
            working_ttl_minutes=int(os.getenv("MEMORY_WORKING_TTL_MINUTES", "60")),
            top_k=int(os.getenv("MEMORY_TOP_K", "5")),
            user_id=os.getenv("USER_ID", "default"),
        )
        return cls(
            llm=llm,
            agent=agent,
            memory=memory,
            max_history_length=int(os.getenv("MAX_HISTORY_LENGTH", "100")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )


def _env_bool(name: str, default: bool) -> bool:
    """读取布尔环境变量；未设置时返回默认值。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
