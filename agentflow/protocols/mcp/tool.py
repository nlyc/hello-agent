"""MCPTool：自动发现 server 工具，展开为带前缀的本地 Tool。"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Sequence

from agentflow.core.exceptions import MCPError, ToolError
from agentflow.protocols.mcp.client import MCPClient, Transport
from agentflow.tools.base import Tool, ToolParameter
from agentflow.tools.registry import ToolRegistry


class MCPTool(Tool):
    """MCP server 的门面工具。

    ``expand()`` 会连接 server、发现 N 个工具，并为每个工具创建一个带前缀的
    包装器，避免多 server 同名冲突。
    """

    def __init__(
        self,
        name: str,
        command: Sequence[str] | None = None,
        *,
        transport: Transport | None = None,
        timeout: float = 30.0,
        connect_timeout: float = 10.0,
        tracer=None,
        agent_name: str = "mcp",
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            description=(
                "MCP server 门面：action=list 查看远程工具；"
                "action=call + tool + arguments 直接调用；"
                "expand() 可展开为本地 Tool 注册进 ToolRegistry。"
            ),
        )
        self.prefix = _sanitize(name)
        self.client = MCPClient(
            command,
            transport=transport,
            timeout=connect_timeout,
            cwd=cwd,
            env=env,
        )
        self.call_timeout = float(timeout)
        self.tracer = tracer
        self.agent_name = agent_name
        self._expanded: list[MCPRemoteTool] | None = None

    def connect(self) -> dict[str, Any]:
        """完成 initialize 握手。"""
        return self.client.connect()

    def list_tools(self) -> list[dict[str, Any]]:
        """返回远程工具描述。"""
        return self.client.list_tools()

    def expand(self) -> list["MCPRemoteTool"]:
        """发现并包装远程工具；重复调用复用缓存。"""
        if self._expanded is not None:
            return list(self._expanded)
        try:
            tools = self.list_tools()
        except Exception as exc:
            raise ToolError(f"MCP server '{self.name}' 发现工具失败：{exc}") from exc
        wrappers: list[MCPRemoteTool] = []
        used: set[str] = set()
        for item in tools:
            remote_name = str(item.get("name", "")).strip()
            if not remote_name:
                continue
            public_name = f"{self.prefix}_{_sanitize(remote_name)}"
            if public_name in used:
                raise ToolError(f"MCP 工具名冲突：{public_name}")
            used.add(public_name)
            wrappers.append(
                MCPRemoteTool(
                    client=self.client,
                    remote_name=remote_name,
                    public_name=public_name,
                    description=str(item.get("description", "") or ""),
                    input_schema=dict(item.get("inputSchema", {}) or {}),
                    timeout=self.call_timeout,
                    tracer=self.tracer,
                    agent_name=self.agent_name,
                )
            )
        self._expanded = wrappers
        return list(wrappers)

    def expand_and_register(self, registry: ToolRegistry) -> list["MCPRemoteTool"]:
        """展开并注册到 ToolRegistry，返回包装器列表。"""
        wrappers = self.expand()
        for wrapper in wrappers:
            registry.register_tool(wrapper)
        return wrappers

    def run(self, parameters: dict[str, Any]) -> str:
        """直接操作门面：list / call。"""
        params = parameters or {}
        action = str(params.get("action", "list")).strip().lower()
        try:
            if action == "list":
                tools = self.list_tools()
                lines = [
                    f"- {item.get('name')}: {item.get('description', '')}"
                    for item in tools
                ]
                return "\n".join(lines) if lines else "（server 未暴露工具）"
            if action == "call":
                tool = str(params.get("tool", "")).strip()
                if not tool:
                    return "错误: call 需要 tool"
                arguments, error = _as_arguments(params.get("arguments"))
                if error:
                    return error
                return self.client.call_tool(tool, arguments)
            return f"错误: 未知 action '{action}'，可用：list / call"
        except (MCPError, ToolError) as exc:
            return f"错误: {exc}"
        except Exception as exc:
            return f"错误: MCP 操作失败：{exc}"

    def get_parameters(self) -> list[ToolParameter]:
        """门面工具参数自描述。"""
        return [
            ToolParameter("action", "string", "list / call，默认 list"),
            ToolParameter("tool", "string", "远程工具名（call 用）"),
            ToolParameter("arguments", "object", "参数对象或 JSON 字符串（call 用）"),
        ]

    def close(self) -> None:
        """关闭连接。"""
        self.client.close()

    def __enter__(self) -> "MCPTool":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class MCPRemoteTool(Tool):
    """单个远程 MCP 工具的本地包装器。"""

    def __init__(
        self,
        *,
        client: MCPClient,
        remote_name: str,
        public_name: str,
        description: str,
        input_schema: dict[str, Any],
        timeout: float = 30.0,
        tracer=None,
        agent_name: str = "mcp",
    ) -> None:
        super().__init__(
            name=public_name,
            description=description or f"MCP 远程工具 {remote_name}",
        )
        self.client = client
        self.remote_name = remote_name
        self.input_schema = dict(input_schema or {})
        self.timeout = float(timeout)
        self.tracer = tracer
        self.agent_name = agent_name

    def run(self, parameters: dict[str, Any]) -> str:
        """转换参数类型后调用远程工具。"""
        arguments = coerce_arguments(parameters or {}, self.input_schema)
        started = time.perf_counter()
        try:
            result = self.client.call_tool(self.remote_name, arguments)
        except Exception as exc:
            self._emit(
                {
                    "server_tool": self.remote_name,
                    "arguments": arguments,
                    "error": str(exc),
                    "duration_ms": round(
                        (time.perf_counter() - started) * 1000, 2
                    ),
                }
            )
            raise ToolError(
                f"MCP 工具 '{self.remote_name}' 调用失败：{exc}"
            ) from exc
        self._emit(
            {
                "server_tool": self.remote_name,
                "arguments": arguments,
                "output": result,
                "duration_ms": round(
                    (time.perf_counter() - started) * 1000, 2
                ),
            }
        )
        return result

    def get_parameters(self) -> list[ToolParameter]:
        """把 JSON Schema 转为 ToolParameter。"""
        return schema_to_parameters(self.input_schema)

    def _emit(self, data: dict[str, Any]) -> None:
        if self.tracer is not None:
            self.tracer.emit(
                "mcp.call",
                agent_name=self.agent_name,
                step=0,
                data=data,
            )


def schema_to_parameters(schema: dict[str, Any]) -> list[ToolParameter]:
    """JSON Schema properties → ToolParameter。"""
    properties = dict(schema.get("properties", {}) or {})
    required = set(schema.get("required", []) or [])
    parameters: list[ToolParameter] = []
    for name, spec in properties.items():
        spec = dict(spec or {})
        parameters.append(
            ToolParameter(
                name=str(name),
                type=str(spec.get("type", "string")),
                description=str(spec.get("description", "") or ""),
                required=name in required,
                default=spec.get("default"),
            )
        )
    return parameters


def coerce_arguments(
    arguments: dict[str, Any], schema: dict[str, Any]
) -> dict[str, Any]:
    """按 JSON Schema 把 ReAct 传来的字符串参数转换成正确类型。"""
    properties = dict(schema.get("properties", {}) or {})
    result: dict[str, Any] = {}
    for key, value in (arguments or {}).items():
        spec = properties.get(key, {})
        result[key] = _coerce_value(value, spec, key)
    return result


def _coerce_value(
    value: Any, schema: dict[str, Any], key: str = ""
) -> Any:
    if value is None:
        return None
    type_name = str(schema.get("type", "string"))
    try:
        if type_name == "integer":
            return int(value) if not isinstance(value, str) else int(float(value))
        if type_name == "number":
            return float(value)
        if type_name == "boolean":
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in ("true", "1", "yes", "y", "是"):
                return True
            if text in ("false", "0", "no", "n", "否"):
                return False
            raise ValueError(f"无法转换为布尔值：{value!r}")
        if type_name == "object":
            if isinstance(value, dict):
                return value
            parsed = json.loads(str(value))
            if not isinstance(parsed, dict):
                raise ValueError("不是 JSON 对象")
            return parsed
        if type_name == "array":
            if isinstance(value, (list, tuple)):
                return list(value)
            parsed = json.loads(str(value))
            if not isinstance(parsed, list):
                raise ValueError("不是 JSON 数组")
            return parsed
        return str(value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ToolError(
            f"参数 '{key or schema.get('title', '')}' 无法转换为 {type_name}："
            f"{value!r}（{exc}）"
        ) from exc


def _as_arguments(value: Any) -> tuple[dict[str, Any], str | None]:
    """解析 arguments：对象或 JSON 字符串。"""
    if value is None or value == "":
        return {}, None
    if isinstance(value, dict):
        return dict(value), None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}, "错误: arguments 必须是 JSON 对象"
        if isinstance(parsed, dict):
            return parsed, None
        return {}, "错误: arguments 必须是 JSON 对象"
    return {}, "错误: arguments 必须是对象或 JSON 字符串"


def _sanitize(name: str) -> str:
    """工具名只保留字母数字下划线（含 CJK），避免 ReAct 解析失败。"""
    return re.sub(r"\W", "_", str(name or "").strip()) or "mcp"
