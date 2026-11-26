"""MCP 最小 Server：tools/list + tools/call，支持 stdio 与内存调用。"""

from __future__ import annotations

import inspect
import json
import sys
from typing import Any, Callable

from agentflow.protocols.mcp.jsonrpc import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PROTOCOL_VERSION,
    JsonRpcError,
    decode_line,
    encode_error,
    encode_result,
)


class MCPServer:
    """把普通 Python 函数暴露成 MCP 工具。

    参数：
        name: server 名称，写入 ``initialize`` 的 serverInfo。
        version: server 版本。
    """

    def __init__(self, name: str = "agentflow-mcp", version: str = "2.0.1") -> None:
        self.name = name
        self.version = version
        self._tools: dict[str, dict[str, Any]] = {}
        self._resources: dict[str, dict[str, Any]] = {}
        self._prompts: dict[str, dict[str, Any]] = {}

    def add_tool(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
    ) -> str:
        """注册一个函数；docstring 首行作为工具描述。"""
        tool_name = str(name or func.__name__).strip()
        if not tool_name:
            raise ValueError("MCP 工具名不能为空")
        if tool_name in self._tools:
            raise ValueError(f"MCP 工具已存在：{tool_name}")
        description_text = (
            str(description).strip()
            if description is not None
            else _first_doc_line(func)
        )
        schema = dict(input_schema or _infer_schema(func))
        self._tools[tool_name] = {
            "func": func,
            "description": description_text,
            "inputSchema": schema,
        }
        return tool_name

    def list_tools(self) -> list[dict[str, Any]]:
        """返回 MCP 工具描述列表。"""
        return [
            {
                "name": name,
                "description": item["description"],
                "inputSchema": item["inputSchema"],
            }
            for name, item in sorted(self._tools.items())
        ]

    # ------------------------------------------------------- resources/prompts

    def add_resource(
        self,
        uri: str,
        *,
        name: str = "",
        mime_type: str = "text/plain",
        description: str = "",
        content: str | None = None,
        loader: Callable[[], Any] | None = None,
    ) -> str:
        """注册一个资源；``content`` 与 ``loader`` 二选一。"""
        key = str(uri).strip()
        if not key:
            raise ValueError("资源 uri 不能为空")
        if key in self._resources:
            raise ValueError(f"资源已存在：{key}")
        if content is None and loader is None:
            raise ValueError("resource 需要 content 或 loader")
        self._resources[key] = {
            "uri": key,
            "name": str(name or key),
            "mimeType": str(mime_type),
            "description": str(description),
            "content": content,
            "loader": loader,
        }
        return key

    def list_resources(self) -> list[dict[str, Any]]:
        """返回资源描述列表（不包含内容/加载器）。"""
        return [
            {
                "uri": item["uri"],
                "name": item["name"],
                "mimeType": item["mimeType"],
                "description": item["description"],
            }
            for _, item in sorted(self._resources.items())
        ]

    def read_resource(self, uri: str) -> dict[str, Any]:
        """读取资源内容。"""
        item = self._resources.get(str(uri).strip())
        if item is None:
            raise JsonRpcError(INVALID_PARAMS, f"未知资源：{uri}")
        try:
            value = item["loader"]() if item["loader"] is not None else item["content"]
        except Exception as exc:
            raise JsonRpcError(INTERNAL_ERROR, f"资源读取失败：{exc}") from exc
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="replace")
        else:
            text = str(value)
        return {
            "contents": [
                {
                    "uri": item["uri"],
                    "mimeType": item["mimeType"],
                    "text": text,
                }
            ]
        }

    def add_prompt(
        self,
        name: str,
        *,
        description: str = "",
        arguments: list[dict[str, Any]] | None = None,
        template: str = "",
        builder: Callable[[dict[str, Any]], Any] | None = None,
    ) -> str:
        """注册 Prompt 模板；``template`` 用 ``str.format(**arguments)``，或传 ``builder``。"""
        key = str(name).strip()
        if not key:
            raise ValueError("prompt 名称不能为空")
        if key in self._prompts:
            raise ValueError(f"prompt 已存在：{key}")
        if not template and builder is None:
            raise ValueError("prompt 需要 template 或 builder")
        self._prompts[key] = {
            "name": key,
            "description": str(description),
            "arguments": list(arguments or []),
            "template": template,
            "builder": builder,
        }
        return key

    def list_prompts(self) -> list[dict[str, Any]]:
        """返回 Prompt 描述列表。"""
        return [
            {
                "name": item["name"],
                "description": item["description"],
                "arguments": item["arguments"],
            }
            for _, item in sorted(self._prompts.items())
        ]

    def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """渲染 Prompt 并返回 MCP 风格消息。"""
        item = self._prompts.get(str(name).strip())
        if item is None:
            raise JsonRpcError(INVALID_PARAMS, f"未知 prompt：{name}")
        values = dict(arguments or {})
        try:
            text = (
                item["builder"](values)
                if item["builder"] is not None
                else item["template"].format(**values)
            )
        except KeyError as exc:
            raise JsonRpcError(INVALID_PARAMS, f"缺少 prompt 参数：{exc}") from exc
        except Exception as exc:
            raise JsonRpcError(INTERNAL_ERROR, f"prompt 渲染失败：{exc}") from exc
        return {
            "description": item["description"],
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": str(text)},
                }
            ],
        }

    def handle(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """处理单个 JSON-RPC 方法；失败抛 ``JsonRpcError``。"""
        payload = dict(params or {})
        if method == "initialize":
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {},
                },
                "serverInfo": {"name": self.name, "version": self.version},
            }
        if method == "tools/list":
            return {"tools": self.list_tools()}
        if method == "resources/list":
            return {"resources": self.list_resources()}
        if method == "resources/read":
            uri = str(payload.get("uri", "")).strip()
            if not uri:
                raise JsonRpcError(INVALID_PARAMS, "resources/read 需要 uri")
            return self.read_resource(uri)
        if method == "prompts/list":
            return {"prompts": self.list_prompts()}
        if method == "prompts/get":
            name = str(payload.get("name", "")).strip()
            if not name:
                raise JsonRpcError(INVALID_PARAMS, "prompts/get 需要 name")
            return self.get_prompt(name, payload.get("arguments"))
        if method == "tools/call":
            name = str(payload.get("name", "")).strip()
            if not name:
                raise JsonRpcError(INVALID_PARAMS, "tools/call 需要 name")
            item = self._tools.get(name)
            if item is None:
                raise JsonRpcError(
                    INVALID_PARAMS, f"未知工具 '{name}'，可用：{sorted(self._tools)}"
                )
            arguments = payload.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise JsonRpcError(INVALID_PARAMS, "arguments 必须是对象")
            try:
                result = item["func"](**arguments)
            except TypeError as exc:
                return {
                    "content": [{"type": "text", "text": f"参数错误：{exc}"}],
                    "isError": True,
                }
            except Exception as exc:  # 工具失败以 isError 返回，不中断协议
                return {
                    "content": [
                        {"type": "text", "text": f"{type(exc).__name__}: {exc}"}
                    ],
                    "isError": True,
                }
            return {
                "content": [{"type": "text", "text": str(result)}],
                "isError": False,
            }
        raise JsonRpcError(METHOD_NOT_FOUND, f"未知方法：{method}")

    def run_http(
        self, host: str = "127.0.0.1", port: int = 8000
    ) -> None:
        """以 HTTP/SSE 方式运行（阻塞）。"""
        from agentflow.protocols.mcp.http_server import MCPHttpServer

        MCPHttpServer(self, host=host, port=port).run()

    def run_stdio(self) -> None:
        """从 stdin 逐行读取 JSON-RPC，向 stdout 写响应。"""
        for raw in sys.stdin.buffer:
            if not raw.strip():
                continue
            request_id = None
            try:
                request = decode_line(raw)
                request_id = request.get("id")
                method = str(request.get("method", ""))
                result = self.handle(method, request.get("params"))
                response = encode_result(request_id, result)
            except JsonRpcError as exc:
                response = encode_error(
                    request_id, exc.code, exc.message, exc.data
                )
            except Exception as exc:  # pragma: no cover - 防御性
                response = encode_error(
                    request_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}"
                )
            sys.stdout.write(response + "\n")
            sys.stdout.flush()


def _first_doc_line(func: Callable[..., Any]) -> str:
    """取 docstring 首行作为工具描述。"""
    doc = inspect.getdoc(func) or ""
    return doc.splitlines()[0].strip() if doc else func.__name__


def _infer_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """根据函数签名推断简单 JSON Schema。"""
    properties: dict[str, Any] = {}
    required: list[str] = []
    signature = inspect.signature(func)
    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        properties[name] = {"type": _type_name(parameter.annotation)}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        else:
            properties[name]["default"] = parameter.default
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _type_name(annotation: Any) -> str:
    """把 Python 类型注解映射到 JSON Schema 类型。

    兼容 ``from __future__ import annotations`` 产生的字符串注解。
    """
    if isinstance(annotation, str):
        text = annotation.strip().lower()
        # 处理 ``int | None`` / ``typing.Optional[int]`` 这类组合。
        if "|" in text:
            text = text.split("|", 1)[0].strip()
        if text.startswith("typing.optional["):
            text = text[len("typing.optional[") :].rstrip("]").strip()
        mapping = {
            "int": "integer",
            "integer": "integer",
            "float": "number",
            "number": "number",
            "bool": "boolean",
            "boolean": "boolean",
            "dict": "object",
            "object": "object",
            "list": "array",
            "tuple": "array",
            "set": "array",
            "array": "array",
            "str": "string",
            "string": "string",
        }
        return mapping.get(text, "string")
    if annotation in (int,):
        return "integer"
    if annotation in (float,):
        return "number"
    if annotation in (bool,):
        return "boolean"
    if annotation in (dict,):
        return "object"
    if annotation in (list, tuple, set):
        return "array"
    return "string"
