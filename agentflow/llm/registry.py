"""模型注册表：别名 → 真实 model 名（+可选独立 key / base_url），换 model 即切换。"""

from dataclasses import dataclass
from typing import Optional

from agentflow.core.exceptions import ConfigError


@dataclass(frozen=True)
class ModelSpec:
    """一个可切换模型的描述。base_url / api_key 缺省时沿用当前中转站。"""

    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class ModelRegistry:
    """维护「别名 → ModelSpec」映射，供 LLM 一行切换。"""

    def __init__(self, specs: Optional[dict[str, ModelSpec]] = None):
        self._specs: dict[str, ModelSpec] = {}
        if specs:
            self.register_many(specs)

    def register(
        self,
        alias: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._specs[alias] = ModelSpec(model=model, api_key=api_key, base_url=base_url)

    def register_many(self, specs: dict[str, ModelSpec]) -> None:
        self._specs.update(specs)

    def get(self, alias: str) -> ModelSpec:
        if alias not in self._specs:
            raise ConfigError(f"未知模型别名 '{alias}'，可用：{sorted(self._specs)}")
        return self._specs[alias]

    def list_aliases(self) -> list[str]:
        return sorted(self._specs)

    def __contains__(self, alias: str) -> bool:
        return alias in self._specs
