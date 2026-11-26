"""JSON-RPC 2.0 最小编解码与错误类型。

MCP 使用 JSON-RPC 2.0；本实现只覆盖 ``initialize`` / ``tools/list`` /
``tools/call`` 三个方法，传输层与协议层解耦，便于 stdio 与内存测试共用。
"""

from __future__ import annotations

import json
from typing import Any

PROTOCOL_VERSION = "2024-11-05"

# JSON-RPC 标准错误码。
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class JsonRpcError(Exception):
    """协议层错误：携带 JSON-RPC 错误码，供 Server 转成 error 响应。"""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = int(code)
        self.message = str(message)
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            payload["data"] = self.data
        return payload


def encode_request(
    request_id: Any, method: str, params: dict[str, Any] | None = None
) -> bytes:
    """生成一行 JSON-RPC 请求（含换行）。"""
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": str(method),
        "params": dict(params or {}),
    }
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def encode_result(request_id: Any, result: Any) -> str:
    """生成 JSON-RPC 成功响应（不含换行）。"""
    return json.dumps(
        {"jsonrpc": "2.0", "id": request_id, "result": result},
        ensure_ascii=False,
    )


def encode_error(
    request_id: Any, code: int, message: str, data: Any = None
) -> str:
    """生成 JSON-RPC 错误响应（不含换行）。"""
    error: dict[str, Any] = {"code": int(code), "message": str(message)}
    if data is not None:
        error["data"] = data
    return json.dumps(
        {"jsonrpc": "2.0", "id": request_id, "error": error},
        ensure_ascii=False,
    )


def decode_line(line: bytes | str) -> dict[str, Any]:
    """解析一行 JSON-RPC 消息；失败抛 ``JsonRpcError(-32700)``。"""
    try:
        text = line.decode("utf-8") if isinstance(line, bytes) else str(line)
        payload = json.loads(text)
    except (UnicodeDecodeError, ValueError) as exc:
        raise JsonRpcError(PARSE_ERROR, f"JSON 解析失败：{exc}") from exc
    if not isinstance(payload, dict):
        raise JsonRpcError(INVALID_REQUEST, "JSON-RPC 消息必须是对象")
    return payload
