"""A2ATool：把远程 A2A Agent 变成当前 Agent 的一个 Tool。"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Sequence

from agentflow.core.exceptions import A2AError, ToolError
from agentflow.protocols.a2a.client import A2AClient
from agentflow.tools.base import Tool, ToolParameter


class A2ATool(Tool):
    """远程 Agent 的本地包装器。

    参数：
        url: Agent 根地址。
        name: 本地工具名；默认按 AgentCard 名称生成 ``a2a_<name>``。
        token: 可选 Bearer Token。
        tracer: 可选 Tracer，用于发 ``a2a.call``。
    """

    def __init__(
        self,
        url: str | A2AClient,
        name: str | None = None,
        *,
        token: str = "",
        timeout: float = 60.0,
        tracer=None,
        agent_name: str = "a2a",
        fetch_card: bool = True,
    ) -> None:
        self.client = (
            url if isinstance(url, A2AClient) else A2AClient(url, timeout=timeout, token=token)
        )
        self.tracer = tracer
        self.agent_name = agent_name
        card = self.client.get_card() if fetch_card else None
        self.card = card
        if card is not None:
            local_name = name or f"a2a_{_sanitize(card.name)}"
            skills = "；".join(
                f"{skill.name}: {skill.description}" for skill in card.skills
            )
            description = f"远程 Agent「{card.name}」：{skills or '无技能'}"
        else:
            local_name = name or "a2a_remote"
            description = "远程 A2A Agent"
        super().__init__(name=local_name, description=description)

    def run(self, parameters: dict[str, Any]) -> str:
        """执行远程技能；错误统一转为 ``ToolError``。"""
        params = parameters or {}
        skill = str(params.get("skill", "")).strip()
        if not skill:
            raise ToolError("A2A 调用需要 skill")
        payload = _as_payload(params.get("input"))
        started = time.perf_counter()
        try:
            artifacts = self.client.execute_skill(skill, payload)
        except A2AError as exc:
            self._emit(
                skill,
                payload,
                error=str(exc),
                started=started,
            )
            raise ToolError(str(exc)) from exc
        text = _format_artifacts(artifacts)
        self._emit(skill, payload, output=text, started=started)
        return text or "（任务完成，无工件）"

    def get_parameters(self) -> list[ToolParameter]:
        """工具参数自描述。"""
        return [
            ToolParameter("skill", "string", "远程技能名", True),
            ToolParameter(
                "input",
                "string",
                "技能输入：字符串，或 JSON 对象字符串",
            ),
        ]

    def _emit(
        self,
        skill: str,
        payload: dict[str, Any],
        *,
        output: str = "",
        error: str = "",
        started: float,
    ) -> None:
        if self.tracer is None:
            return
        self.tracer.emit(
            "a2a.call",
            agent_name=self.agent_name,
            step=0,
            data={
                "agent": self.card.name if self.card else self.name,
                "skill": skill,
                "input": payload,
                "output": output,
                "error": error,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )


def _format_artifacts(artifacts) -> str:
    """单工件直接返回正文，多工件带名称，便于 Agent 链式传递。"""
    if len(artifacts) == 1 and artifacts[0].name == "result":
        return str(artifacts[0].content).strip()
    return "\n".join(
        f"[{artifact.name}] {artifact.content}" for artifact in artifacts
    ).strip()


def _as_payload(value: Any) -> dict[str, Any]:
    """把 ReAct 的 input 参数转成 payload 字典。"""
    if isinstance(value, dict):
        return dict(value)
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = json.loads(text)
        except ValueError:
            return {"input": text}
        if isinstance(parsed, dict):
            return parsed
        return {"input": parsed}
    return {"input": value}


def _sanitize(name: str) -> str:
    """工具名只保留字母数字下划线（含 CJK）。"""
    return re.sub(r"\W", "_", str(name or "").strip()) or "remote"
