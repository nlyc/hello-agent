"""A2A 最小 Server：AgentCard + Task 生命周期 + 四个 HTTP 端点。

端点：
- GET  /.well-known/agent-card.json
- POST /v1/tasks
- GET  /v1/tasks/{id}
- POST /v1/tasks/{id}/cancel
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

from agentflow.core.exceptions import A2AError
from agentflow.protocols.a2a.types import (
    AgentCard,
    Artifact,
    Skill,
    Task,
    TaskStatus,
)


class A2AServer:
    """把一个或多个 Python 函数暴露成 A2A 技能。

    参数：
        name: Agent 名称。
        description: Agent 描述。
        host: 默认只绑定 ``127.0.0.1``。
        port: 0 表示自动分配。
        token: 可选 Bearer Token。
        max_body_bytes: 请求体上限。
        task_ttl: 终态任务内存保留秒数。
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        *,
        version: str = "2.0.0",
        host: str = "127.0.0.1",
        port: int = 0,
        token: str = "",
        max_body_bytes: int = 1_000_000,
        task_ttl: float = 3600.0,
        async_mode: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.version = version
        self.host = host
        self.port = int(port)
        self.token = token
        self.max_body_bytes = max(1, int(max_body_bytes))
        self.task_ttl = float(task_ttl)
        self.async_mode = bool(async_mode)
        self._skills: dict[str, Skill] = {}
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self._tasks: dict[str, Task] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()
        self._http: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------------- skills

    def skill(
        self,
        name: str,
        description: str = "",
        *,
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
    ):
        """装饰器：注册一项技能。handler 接收一个 payload 字典。"""

        def decorator(func: Callable[[dict[str, Any]], Any]):
            self.add_skill(
                func,
                name=name,
                description=description,
                input_schema=input_schema,
                output_schema=output_schema,
            )
            return func

        return decorator

    def add_skill(
        self,
        func: Callable[[dict[str, Any]], Any],
        *,
        name: str,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> Skill:
        """注册技能函数。"""
        key = str(name).strip()
        if not key:
            raise A2AError("技能名不能为空")
        if key in self._skills:
            raise A2AError(f"技能已存在：{key}")
        skill = Skill(
            name=key,
            description=str(description or func.__doc__ or "").strip(),
            input_schema=dict(input_schema or {}),
            output_schema=dict(output_schema or {}),
        )
        self._skills[key] = skill
        self._handlers[key] = func
        return skill

    def card(self) -> AgentCard:
        """生成 AgentCard；url 使用实际绑定地址。"""
        return AgentCard(
            name=self.name,
            description=self.description,
            url=self.url,
            version=self.version,
            skills=list(self._skills.values()),
            capabilities={"streaming": False, "tasks": True},
        )

    @property
    def url(self) -> str:
        """当前服务 URL。"""
        return f"http://{self.host}:{self.port}"

    # ----------------------------------------------------------------- tasks

    def submit_task(self, skill: str, payload: dict[str, Any]) -> Task:
        """创建任务并执行；``async_mode=True`` 时后台执行并立即返回。"""
        key = str(skill).strip()
        handler = self._handlers.get(key)
        if handler is None:
            raise A2AError(f"未知技能 '{key}'，可用：{sorted(self._handlers)}")
        task = Task.create(key, payload)
        with self._lock:
            self._prune_tasks()
            self._tasks[task.id] = task
            self._events[task.id] = []
        if self.async_mode:
            worker = threading.Thread(
                target=self._run_task,
                args=(task, handler, dict(payload or {})),
                daemon=True,
            )
            worker.start()
            return task
        return self._run_task(task, handler, dict(payload or {}))

    def _run_task(
        self,
        task: Task,
        handler: Callable[[dict[str, Any]], Any],
        payload: dict[str, Any],
    ) -> Task:
        """执行任务并写入状态事件；失败写入 Task，不抛给 HTTP。"""
        try:
            task.transition(TaskStatus.WORKING)
        except A2AError:
            return task
        self._append_event(task.id, {"type": "status", "task": task.to_dict()})
        try:
            result = handler(payload)
        except Exception as exc:
            task.error = f"{type(exc).__name__}: {exc}"
            try:
                task.transition(TaskStatus.FAILED)
            except A2AError:
                pass
            self._append_event(task.id, {"type": "status", "task": task.to_dict()})
            return task
        if task.status == TaskStatus.CANCELED:
            self._append_event(task.id, {"type": "status", "task": task.to_dict()})
            return task
        task.artifacts = _to_artifacts(result)
        try:
            task.transition(TaskStatus.COMPLETED)
        except A2AError:
            pass
        self._append_event(task.id, {"type": "status", "task": task.to_dict()})
        return task

    def _append_event(self, task_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.setdefault(task_id, []).append(dict(event))

    def stream_events(
        self,
        task_id: str,
        *,
        timeout: float = 120.0,
        poll_interval: float = 0.2,
    ):
        """按顺序产出任务状态事件，直到终态或超时。"""
        task = self.get_task(task_id)
        index = 0
        deadline = time.monotonic() + float(timeout)
        while True:
            with self._lock:
                events = list(self._events.get(task_id, []))
                status = task.status
            while index < len(events):
                yield events[index]
                index += 1
            if status.terminal and index >= len(events):
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(min(poll_interval, max(0.01, deadline - time.monotonic())))

    def get_task(self, task_id: str) -> Task:
        """查询任务；不存在抛 ``A2AError``。"""
        with self._lock:
            task = self._tasks.get(str(task_id))
        if task is None:
            raise A2AError(f"未找到任务：{task_id}")
        return task

    def cancel_task(self, task_id: str) -> Task:
        """取消任务；终态任务不可取消。"""
        task = self.get_task(task_id)
        task.transition(TaskStatus.CANCELED)
        self._append_event(task.id, {"type": "status", "task": task.to_dict()})
        return task

    def _prune_tasks(self) -> None:
        """清理超过 TTL 的终态任务与事件。"""
        now = time.time()
        expired = [
            task_id
            for task_id, task in self._tasks.items()
            if task.status.terminal
            and (now - task.updated_at.timestamp()) > self.task_ttl
        ]
        for task_id in expired:
            self._tasks.pop(task_id, None)
            self._events.pop(task_id, None)

    # ---------------------------------------------------------------- server

    def start(self) -> str:
        """后台启动 HTTP 服务，返回实际 URL。"""
        if self._http is not None:
            return self.url
        self._http = ThreadingHTTPServer(
            (self.host, self.port), _A2ARequestHandler
        )
        self._http.a2a = self  # type: ignore[attr-defined]
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
        if self._http is not None:
            raise A2AError("服务已经启动")
        self.start()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()

    def __enter__(self) -> "A2AServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


class _A2ARequestHandler(BaseHTTPRequestHandler):
    """A2A HTTP 适配层；只做路由、鉴权、大小限制与 JSON。"""

    server_version = "AgentFlowA2A/2.0"

    @property
    def a2a(self) -> A2AServer:
        return self.server.a2a  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/.well-known/agent-card.json":
            self._send_json(200, self.a2a.card().to_dict())
            return
        if not self._authorized():
            return
        events_match = re.fullmatch(
            r"/v1/tasks/([A-Za-z0-9_\-]+)/events", path
        )
        if events_match:
            try:
                self.a2a.get_task(events_match.group(1))
            except A2AError as exc:
                self._send_json(404, {"error": str(exc)})
                return
            self._stream_task_events(events_match.group(1))
            return
        match = re.fullmatch(r"/v1/tasks/([A-Za-z0-9_\-]+)", path)
        if match:
            try:
                self._send_json(200, self.a2a.get_task(match.group(1)).to_dict())
            except A2AError as exc:
                self._send_json(404, {"error": str(exc)})
            return
        self._send_json(404, {"error": f"未知路径：{path}"})

    def do_POST(self) -> None:
        if not self._authorized():
            return
        path = urlparse(self.path).path
        payload = self._read_json()
        if payload is None:
            return
        if path == "/v1/tasks":
            skill = str(payload.get("skill", "")).strip()
            if not skill:
                self._send_json(400, {"error": "缺少 skill"})
                return
            try:
                task = self.a2a.submit_task(skill, dict(payload.get("input") or {}))
            except A2AError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            self._send_json(200, task.to_dict())
            return
        match = re.fullmatch(r"/v1/tasks/([A-Za-z0-9_\-]+)/cancel", path)
        if match:
            try:
                task = self.a2a.cancel_task(match.group(1))
            except A2AError as exc:
                self._send_json(409, {"error": str(exc)})
                return
            self._send_json(200, task.to_dict())
            return
        self._send_json(404, {"error": f"未知路径：{path}"})

    def _stream_task_events(self, task_id: str) -> None:
        """SSE 流式输出任务状态事件。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for event in self.a2a.stream_events(task_id):
                payload = json.dumps(event, ensure_ascii=False)
                self.wfile.write(
                    f"event: {event.get('type', 'message')}\n"
                    f"data: {payload}\n\n".encode("utf-8")
                )
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.close_connection = True

    def _authorized(self) -> bool:
        if not self.a2a.token:
            return True
        header = str(self.headers.get("Authorization", ""))
        if header == f"Bearer {self.a2a.token}":
            return True
        if self.command == "POST":
            # 鉴权失败时请求体尚未读取；读掉小请求体并关闭连接，避免 RST。
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
            self._send_json(400, {"error": "Content-Length 非法"})
            return None
        if length > self.a2a.max_body_bytes:
            # 小超限请求先读掉请求体，避免客户端收到连接重置；超大请求直接拒绝。
            if 0 < length <= 10_000_000:
                try:
                    self.rfile.read(length)
                except OSError:
                    pass
            self.close_connection = True
            self._send_json(413, {"error": "请求体过大"})
            return None
        try:
            raw = self.rfile.read(length) if length else b""
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, ValueError):
            self._send_json(400, {"error": "请求体必须是 JSON 对象"})
            return None
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "请求体必须是 JSON 对象"})
            return None
        return payload

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """静默 HTTP 访问日志，避免污染 Demo 输出。"""
        return


def _to_artifacts(result: Any) -> list[Artifact]:
    """把技能返回值统一转换为 Artifact 列表。"""
    if result is None:
        return []
    if isinstance(result, Artifact):
        return [result]
    if isinstance(result, (list, tuple)):
        artifacts: list[Artifact] = []
        for index, item in enumerate(result, start=1):
            artifacts.extend(_to_artifacts(item))
        return artifacts
    if isinstance(result, dict):
        return [
            Artifact(
                name="result",
                mime="application/json",
                content=json.dumps(result, ensure_ascii=False),
            )
        ]
    return [Artifact(name="result", mime="text/plain", content=str(result))]
