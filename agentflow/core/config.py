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


class Config(BaseModel):
    """框架总配置，零配置即可实例化运行。"""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
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
        return cls(
            llm=llm,
            agent=agent,
            max_history_length=int(os.getenv("MAX_HISTORY_LENGTH", "100")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )
