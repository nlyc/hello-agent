"""统一配置入口：优先读取项目根目录的 ``.env``，回落到进程环境变量。

设计要点：

- **零硬编码**：业务代码通过 ``get_env`` / ``Config.from_env`` 读取配置，
  不再各自散落 ``os.getenv``。
- **外部环境变量优先**：Shell 里显式设置的值会覆盖 ``.env``（12-Factor 习惯），
  但通过 ``.env`` 载入的值在文件更新后会同步刷新（便于 Web 控制台热改配置）。
- **可选依赖**：优先使用 ``python-dotenv``；未安装时用内置解析器，行为一致。
- **默认镜像**：未配置 ``HF_ENDPOINT`` 时默认 ``https://hf-mirror.com``，
  避免 FastEmbed 首次下载模型时直连 HuggingFace 失败。
- **延迟/热加载**：``get_env`` 每次调用都会检查 ``.env`` 的 mtime，
  文件变化后立即生效，无需重启进程。

路径解析顺序：``AGENTFLOW_ENV_FILE`` → 仓库根目录 ``.env`` → 当前工作目录 ``.env``。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / ".cache" / "fastembed"

# 记录哪些键是由 .env 注入的：只有这些键允许被后续文件更新覆盖；
# Shell 中原本就存在的键永远优先。
_loaded_keys: set[str] = set()
_shell_keys: set[str] = set(os.environ)
_loaded_path: Optional[Path] = None
_loaded_mtime: Optional[float] = None


def project_root() -> Path:
    """返回仓库根目录（agentflow 包的上一层）。"""
    return _PROJECT_ROOT


def find_env_file() -> Optional[Path]:
    """按优先级查找 .env 文件，找不到返回 None。"""
    explicit = os.environ.get("AGENTFLOW_ENV_FILE")
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(_PROJECT_ROOT / ".env")
    cwd_env = Path.cwd() / ".env"
    if cwd_env != candidates[-1]:
        candidates.append(cwd_env)
    for path in candidates:
        if path.is_file():
            return path
    return None


def _parse_dotenv(text: str) -> dict[str, str]:
    """解析 .env 文本（支持 export、引号、行内注释与常见转义）。"""
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            quote = value[0]
            value = value[1:-1]
            if quote == '"':
                value = (
                    value.replace("\\n", "\n")
                    .replace("\\r", "\r")
                    .replace("\\t", "\t")
                    .replace('\\"', '"')
                )
        else:
            # 未加引号时，允许 `KEY=value  # 注释` 写法
            for marker in (" #", "\t#"):
                if marker in value:
                    value = value.split(marker, 1)[0].rstrip()
                    break
        values[key] = value
    return values


def load_env(force: bool = False) -> Optional[Path]:
    """加载/刷新 .env，返回实际使用的路径（不存在则返回 None）。

    Shell 中已存在的变量优先；``.env`` 里新增/修改的键会同步到 ``os.environ``，
    被删除的键也会从进程中移除（仅限当初由 .env 注入的键）。
    """
    global _loaded_path, _loaded_mtime
    path = find_env_file()
    if path is None:
        _loaded_path, _loaded_mtime = None, None
        _apply_defaults()
        return None

    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    if not force and path == _loaded_path and mtime == _loaded_mtime:
        _apply_defaults()
        return path

    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None

    values = _parse_dotenv(text)
    for key in list(_loaded_keys):
        if key not in values and key not in _shell_keys:
            os.environ.pop(key, None)
            _loaded_keys.discard(key)
    for key, value in values.items():
        if key in _shell_keys:
            continue  # Shell 显式设置优先
        os.environ[key] = value
        _loaded_keys.add(key)

    _loaded_path, _loaded_mtime = path, mtime
    _apply_defaults()
    return path


def _apply_defaults() -> None:
    """为在线能力设置安全默认值（仅当用户没有显式配置时）。"""
    os.environ.setdefault("HF_ENDPOINT", _DEFAULT_HF_ENDPOINT)
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(_DEFAULT_CACHE_DIR))
    os.environ.setdefault("VECTOR_STORE", "sqlite")


def get_env(name: str, default: str = "") -> str:
    """读取配置：.env / Shell 环境变量 → 默认值（空字符串视为未设置）。"""
    load_env()
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def env_bool(name: str, default: bool = False) -> bool:
    """读取布尔配置，接受 1/true/yes/on。"""
    raw = get_env(name)
    if raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    """读取整数配置，非法值回落到默认值。"""
    raw = get_env(name)
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    """读取浮点配置，非法值回落到默认值。"""
    raw = get_env(name)
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# 可对外展示的配置键白名单：避免把整份系统环境变量暴露给 API。
_PUBLIC_KEYS = {
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "LLM_TIMEOUT",
    "TAVILY_API_KEY",
    "SERPAPI_API_KEY",
    "SEARCH_MAX_RESULTS",
    "EMBED_BACKEND",
    "FASTEMBED_MODEL",
    "FASTEMBED_CACHE_PATH",
    "HF_ENDPOINT",
    "VECTOR_STORE",
    "RAG_SQLITE_PATH",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
    "QDRANT_TIMEOUT",
    "QDRANT_PREFER_GRPC",
    "MEMORY_ENABLED",
    "MEMORY_STORE",
    "MEMORY_SQLITE_PATH",
    "MEMORY_TOP_K",
    "USER_ID",
    "AGENT_MAX_STEPS",
    "AGENT_MAX_TOOL_ITERATIONS",
    "MAX_HISTORY_LENGTH",
    "DEBUG",
}


def env_snapshot() -> dict[str, str]:
    """返回当前配置快照（白名单 + 密钥脱敏），供 Web 控制台展示。"""
    load_env()
    secret_keys = {
        "LLM_API_KEY",
        "TAVILY_API_KEY",
        "SERPAPI_API_KEY",
        "QDRANT_API_KEY",
    }
    result: dict[str, str] = {}
    for key in sorted(_PUBLIC_KEYS):
        value = os.environ.get(key, "")
        if not value:
            continue
        result[key] = "***" if key in secret_keys else value
    return result


# 导入即尝试加载一次；调用方无需关心初始化时机。
load_env()
