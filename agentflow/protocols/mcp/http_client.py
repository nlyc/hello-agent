"""MCP HTTP 传输：JSON-RPC over HTTP（StreamableHTTP 的最小子集）。"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agentflow.core.exceptions import MCPError
from agentflow.protocols.mcp.jsonrpc import JsonRpcError, decode_line


class MCPHttpTransport:
    """向 MCP HTTP server 发送单次 JSON-RPC 请求。

    ``url`` 是完整的 MCP 端点，例如 ``http://127.0.0.1:8765/mcp``。
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        token: str = "",
    ) -> None:
        self.url = str(url).rstrip("/")
        self.timeout = float(timeout)
        self.token = str(token)
        self._next_id = 0
        self.closed = False

    def request(
        self, method: str, params: dict[str, Any] | None, *, timeout: float
    ) -> Any:
        if self.closed:
            raise MCPError("传输已关闭")
        self._next_id += 1
        request_id = self._next_id
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": str(method),
            "params": dict(params or {}),
        }
        request = Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urlopen(
                request, timeout=timeout or self.timeout
            ) as response:
                body = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise MCPError(f"MCP HTTP {exc.code}：{detail or exc.reason}") from exc
        except URLError as exc:
            raise MCPError(f"MCP 连接失败：{exc.reason}") from exc

        try:
            decoded = decode_line(body)
        except JsonRpcError as exc:
            raise MCPError(f"MCP 响应解析失败：{exc.message}") from exc
        if decoded.get("error"):
            error = decoded["error"]
            raise MCPError(
                f"MCP 错误 {error.get('code')}: {error.get('message', '')}"
            )
        if "result" not in decoded:
            raise MCPError("MCP 响应缺少 result")
        return decoded["result"]

    def close(self) -> None:
        self.closed = True
