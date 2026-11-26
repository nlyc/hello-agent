"""MCP 最小 Client：stdio 传输 + 内存传输，方法子集 initialize / tools/list / tools/call。"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Protocol, Sequence

from agentflow.core.exceptions import MCPError
from agentflow.protocols.mcp.jsonrpc import (
    PROTOCOL_VERSION,
    JsonRpcError,
    encode_request,
    decode_line,
)


class Transport(Protocol):
    """传输协议：一次 request 对应一次响应。"""

    def request(
        self, method: str, params: dict[str, Any] | None, *, timeout: float
    ) -> Any:
        ...

    def close(self) -> None:
        ...


class InMemoryTransport:
    """内存传输：直接调用 ``MCPServer.handle``，用于单测与同进程 Demo。"""

    def __init__(self, handler: Callable[[str, dict[str, Any] | None], Any]) -> None:
        self.handler = handler
        self.closed = False

    def request(
        self, method: str, params: dict[str, Any] | None, *, timeout: float
    ) -> Any:
        if self.closed:
            raise MCPError("传输已关闭")
        return self.handler(method, params)

    def close(self) -> None:
        self.closed = True


class StdioTransport:
    """stdio 传输：启动子进程，换行分隔 JSON-RPC，读取线程 + 队列超时。"""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        max_stderr_chars: int = 4000,
    ) -> None:
        if not command:
            raise MCPError("MCP server command 不能为空")
        self.command = [str(part) for part in command]
        merged_env = dict(os.environ)
        merged_env.setdefault("PYTHONIOENCODING", "utf-8")
        if env:
            merged_env.update(env)
        try:
            self._process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=merged_env,
                bufsize=0,
            )
        except OSError as exc:
            raise MCPError(f"MCP server 启动失败：{exc}") from exc
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._stderr_parts: list[str] = []
        self._max_stderr_chars = max_stderr_chars
        self._lock = threading.Lock()
        self._next_id = 0
        self.closed = False
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(
            target=self._read_stderr, daemon=True
        )
        self._stderr_reader.start()

    def request(
        self, method: str, params: dict[str, Any] | None, *, timeout: float
    ) -> Any:
        with self._lock:
            if self.closed:
                raise MCPError("传输已关闭")
            if self._process.poll() is not None:
                raise MCPError(self._exit_message())
            self._next_id += 1
            request_id = self._next_id
            try:
                self._process.stdin.write(
                    encode_request(request_id, method, params)
                )
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise MCPError(self._exit_message(str(exc))) from exc
            deadline = time.monotonic() + max(0.001, float(timeout))
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MCPError(f"MCP 请求超时（>{timeout:g}s）：{method}")
                try:
                    line = self._queue.get(timeout=remaining)
                except queue.Empty as exc:
                    raise MCPError(
                        f"MCP 请求超时（>{timeout:g}s）：{method}"
                    ) from exc
                if line is None:
                    raise MCPError(self._exit_message("server 进程已退出"))
                payload = decode_line(line)
                if payload.get("id") != request_id:
                    # 通知/迟到响应：忽略，继续等匹配 id。
                    continue
                return self._handle_response(payload)

    def _handle_response(self, payload: dict[str, Any]) -> Any:
        error = payload.get("error")
        if error:
            message = str(error.get("message", "MCP 调用失败"))
            code = error.get("code")
            raise MCPError(f"MCP 错误 {code}: {message}")
        if "result" not in payload:
            raise MCPError("MCP 响应缺少 result")
        return payload["result"]

    def _read_stdout(self) -> None:
        stream = self._process.stdout
        if stream is None:
            self._queue.put(None)
            return
        try:
            for line in stream:
                self._queue.put(line)
        finally:
            self._queue.put(None)

    def _read_stderr(self) -> None:
        stream = self._process.stderr
        if stream is None:
            return
        try:
            for line in stream:
                text = line.decode("utf-8", errors="replace")
                self._stderr_parts.append(text)
                combined = "".join(self._stderr_parts)
                if len(combined) > self._max_stderr_chars:
                    self._stderr_parts = [combined[-self._max_stderr_chars :]]
        except Exception:  # pragma: no cover - 进程退出时流关闭
            return

    def _exit_message(self, detail: str = "") -> str:
        code = self._process.poll()
        stderr = "".join(self._stderr_parts).strip()
        parts = [f"MCP server 已退出（code={code}）"]
        if detail:
            parts.append(detail)
        if stderr:
            parts.append(f"stderr: {stderr[-1000:]}")
        return "；".join(parts)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        process = self._process
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.terminate()
            process.wait(timeout=3)
        except (subprocess.TimeoutExpired, OSError):
            try:
                process.kill()
            except OSError:
                pass
        for thread in (self._reader, self._stderr_reader):
            thread.join(timeout=1)


class MCPClient:
    """最小 MCP 客户端。"""

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        transport: Transport | None = None,
        timeout: float = 10.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.timeout = float(timeout)
        self._transport: Transport = transport or StdioTransport(
            command or [], cwd=cwd, env=env
        )
        self.server_info: dict[str, Any] = {}
        self._connected = False

    def connect(self) -> dict[str, Any]:
        """发送 initialize；返回 serverInfo 等握手信息。"""
        result = self._transport.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "agentflow", "version": "2.0.0"},
            },
            timeout=self.timeout,
        )
        self.server_info = dict(result or {})
        self._connected = True
        return self.server_info

    def list_tools(self) -> list[dict[str, Any]]:
        """发现 server 暴露的全部工具。"""
        self._ensure_connected()
        result = self._transport.request("tools/list", {}, timeout=self.timeout)
        tools = (result or {}).get("tools", [])
        return list(tools)

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> str:
        """调用远程工具；``isError`` 时抛 ``MCPError``。"""
        self._ensure_connected()
        result = self._transport.request(
            "tools/call",
            {"name": name, "arguments": dict(arguments or {})},
            timeout=self.timeout if timeout is None else timeout,
        )
        result = dict(result or {})
        parts = [
            str(item.get("text", ""))
            for item in result.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        text = "\n".join(part for part in parts if part).strip()
        if result.get("isError"):
            raise MCPError(text or f"MCP 工具 '{name}' 调用失败")
        return text

    def list_resources(self) -> list[dict[str, Any]]:
        """发现 server 暴露的资源。"""
        self._ensure_connected()
        result = self._transport.request(
            "resources/list", {}, timeout=self.timeout
        )
        return list((result or {}).get("resources", []))

    def read_resource(self, uri: str) -> str:
        """读取资源文本内容。"""
        self._ensure_connected()
        result = self._transport.request(
            "resources/read", {"uri": str(uri)}, timeout=self.timeout
        )
        contents = list((result or {}).get("contents", []))
        return "\n".join(
            str(item.get("text", ""))
            for item in contents
            if isinstance(item, dict)
        ).strip()

    def list_prompts(self) -> list[dict[str, Any]]:
        """发现 server 暴露的 Prompt。"""
        self._ensure_connected()
        result = self._transport.request("prompts/list", {}, timeout=self.timeout)
        return list((result or {}).get("prompts", []))

    def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """渲染 Prompt，返回 MCP 消息结构。"""
        self._ensure_connected()
        result = self._transport.request(
            "prompts/get",
            {"name": str(name), "arguments": dict(arguments or {})},
            timeout=self.timeout,
        )
        return dict(result or {})

    def close(self) -> None:
        """关闭传输并回收子进程。"""
        self._transport.close()

    def _ensure_connected(self) -> None:
        if not self._connected:
            self.connect()

    def __enter__(self) -> "MCPClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def stdio_command(
    module: str, *, python: str | None = None
) -> list[str]:
    """便捷构造 ``python -m module`` 的 stdio 命令。"""
    return [python or sys.executable, "-m", module]
