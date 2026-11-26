"""MCP HTTP/SSE server：单端点 JSON-RPC + 最小 SSE 握手。

- ``POST /mcp``：接收 JSON-RPC 2.0 请求，返回 JSON 响应。
- ``GET /mcp/sse``：SSE 握手（ready + keepalive），用于演示传输能力。
- 默认只绑定 ``127.0.0.1``；可选 Bearer Token 与请求体上限。
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from agentflow.protocols.mcp.jsonrpc import (
    INTERNAL_ERROR,
    JsonRpcError,
    decode_line,
    encode_error,
    encode_result,
)
from agentflow.protocols.mcp.server import MCPServer


class MCPHttpServer:
    """把 ``MCPServer`` 暴露为 HTTP/SSE 服务。"""

    def __init__(
        self,
        server: MCPServer,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        token: str = "",
        max_body_bytes: int = 1_000_000,
    ) -> None:
        self.mcp = server
        self.host = host
        self.port = int(port)
        self.token = token
        self.max_body_bytes = max(1, int(max_body_bytes))
        self._http: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        """MCP 端点 URL（客户端 POST 到这里）。"""
        return f"http://{self.host}:{self.port}/mcp"

    def start(self) -> str:
        """后台启动 HTTP 服务，返回实际端点 URL。"""
        if self._http is not None:
            return self.url
        self._http = ThreadingHTTPServer(
            (self.host, self.port), _MCPHttpHandler
        )
        self._http.mcp_http = self  # type: ignore[attr-defined]
        self._http.daemon_threads = True
        self.port = int(self._http.server_address[1])
        self._thread = threading.Thread(
            target=self._http.serve_forever, daemon=True
        )
        self._thread.start()
        return self.url

    def stop(self) -> None:
        """停止 HTTP 服务。"""
        if self._http is None:
            return
        self._http.shutdown()
        self._http.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._http = None
        self._thread = None

    def run(self) -> None:
        """阻塞运行，直到 Ctrl+C。"""
        self.start()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()

    def __enter__(self) -> "MCPHttpServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


class _MCPHttpHandler(BaseHTTPRequestHandler):
    server_version = "AgentFlowMCP/2.0"

    @property
    def config(self) -> MCPHttpServer:
        return self.server.mcp_http  # type: ignore[attr-defined]

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in ("/mcp", "/mcp/"):
            self._send_json(404, {"error": f"未知路径：{path}"})
            return
        if not self._authorized():
            return
        payload = self._read_json()
        if payload is None:
            return
        request_id = payload.get("id")
        try:
            result = self.config.mcp.handle(
                str(payload.get("method", "")), payload.get("params")
            )
            body = encode_result(request_id, result)
        except JsonRpcError as exc:
            body = encode_error(request_id, exc.code, exc.message, exc.data)
        except Exception as exc:  # pragma: no cover - 防御性
            body = encode_error(
                request_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}"
            )
        self._send_raw(200, body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/mcp/sse":
            self._send_json(404, {"error": f"未知路径：{path}"})
            return
        if not self._authorized():
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        ready = json.dumps(
            {"server": self.config.mcp.name, "version": self.config.mcp.version},
            ensure_ascii=False,
        )
        try:
            self.wfile.write(
                f"event: ready\ndata: {ready}\n\n".encode("utf-8")
            )
            self.wfile.flush()
            while True:
                time.sleep(10)
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.close_connection = True

    def _authorized(self) -> bool:
        if not self.config.token:
            return True
        header = str(self.headers.get("Authorization", ""))
        if header == f"Bearer {self.config.token}":
            return True
        if self.command == "POST":
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if 0 < length <= 10_000_000:
                try:
                    self.rfile.read(length)
                except OSError:
                    pass
            self.close_connection = True
        self._send_json(401, {"error": "未授权"})
        return False

    def _read_json(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length > self.config.max_body_bytes:
            self.close_connection = True
            self._send_json(413, {"error": "请求体过大"})
            return None
        try:
            raw = self.rfile.read(length) if length else b""
            if not raw:
                return {}
            payload = decode_line(raw)
        except JsonRpcError:
            self._send_json(400, {"error": "请求体不是合法 JSON-RPC"})
            return None
        return payload

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send_raw(
            status, json.dumps(payload, ensure_ascii=False)
        )

    def _send_raw(self, status: int, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: Any) -> None:
        return
