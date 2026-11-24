"""A2A 最小 Client：发现 AgentCard、提交任务、轮询终态、取回 Artifact。"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agentflow.core.exceptions import A2AError
from agentflow.protocols.a2a.types import AgentCard, Artifact, Task, TaskStatus


class A2AClient:
    """调用远程 A2A Agent。

    参数：
        url: Agent 根地址，如 ``http://127.0.0.1:5001``。
        timeout: 单次 HTTP 请求超时。
        token: 可选 Bearer Token。
        poll_interval: 任务非终态时的轮询间隔。
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 60.0,
        token: str = "",
        poll_interval: float = 0.1,
    ) -> None:
        self.url = str(url).rstrip("/")
        self.timeout = float(timeout)
        self.token = token
        self.poll_interval = max(0.01, float(poll_interval))

    def get_card(self) -> AgentCard:
        """拉取 ``/.well-known/agent-card.json``。"""
        return AgentCard.from_dict(
            self._request("GET", "/.well-known/agent-card.json")
        )

    def create_task(self, skill: str, payload: dict[str, Any] | None = None) -> Task:
        """创建任务；服务端同步执行并返回终态或执行中任务。"""
        data = self._request(
            "POST",
            "/v1/tasks",
            {"skill": str(skill), "input": dict(payload or {})},
        )
        return Task.from_dict(data)

    def get_task(self, task_id: str) -> Task:
        """查询任务。"""
        return Task.from_dict(self._request("GET", f"/v1/tasks/{task_id}"))

    def cancel_task(self, task_id: str) -> Task:
        """取消任务。"""
        return Task.from_dict(
            self._request("POST", f"/v1/tasks/{task_id}/cancel", {})
        )

    def execute_skill(
        self,
        skill: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> list[Artifact]:
        """提交任务并等待终态，返回工件列表。"""
        card = self.get_card()
        names = {item.name for item in card.skills}
        if skill not in names:
            raise A2AError(
                f"Agent '{card.name}' 没有技能 '{skill}'，可用：{sorted(names)}"
            )
        task = self.create_task(skill, payload)
        deadline = time.monotonic() + float(timeout or self.timeout)
        while not task.status.terminal:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise A2AError(f"A2A 任务超时：{task.id} / {skill}")
            time.sleep(min(self.poll_interval, remaining))
            task = self.get_task(task.id)
        if task.status == TaskStatus.FAILED:
            raise A2AError(f"A2A 任务失败：{task.error or '未知错误'}")
        if task.status == TaskStatus.CANCELED:
            raise A2AError(f"A2A 任务已取消：{task.id}")
        return list(task.artifacts)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = (
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        )
        request = Request(
            f"{self.url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise A2AError(
                f"A2A HTTP {exc.code}：{detail or exc.reason}"
            ) from exc
        except URLError as exc:
            raise A2AError(f"A2A 连接失败：{exc.reason}") from exc
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise A2AError(f"A2A 响应不是 JSON：{exc}") from exc
        if not isinstance(parsed, dict):
            raise A2AError("A2A 响应必须是 JSON 对象")
        return parsed
