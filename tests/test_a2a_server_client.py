"""A2A Server / Client 测试：发现、任务、失败、鉴权与大小限制。"""

from __future__ import annotations

import json
import time
import urllib.request

import pytest

from agentflow.core.exceptions import A2AError
from agentflow.protocols.a2a import A2AClient, A2AServer, TaskStatus


def make_server(**kwargs) -> A2AServer:
    server = A2AServer("研究助手", "收集旅行事实", **kwargs)

    @server.skill("gather_facts", "收集城市事实")
    def gather_facts(payload):
        city = payload.get("input", "北京")
        return f"{city}：故宫周一闭馆；带父母建议东宫门平缓路线。"

    @server.skill("boom", "总是失败")
    def boom(payload):
        raise RuntimeError("boom")

    return server


def test_card_and_execute_skill():
    server = make_server()
    url = server.start()
    try:
        client = A2AClient(url, timeout=10)
        card = client.get_card()
        assert card.name == "研究助手"
        assert {skill.name for skill in card.skills} == {"gather_facts", "boom"}
        assert card.url == url

        artifacts = client.execute_skill("gather_facts", {"input": "北京"})
        assert len(artifacts) == 1
        assert "故宫周一闭馆" in artifacts[0].content
        assert artifacts[0].mime == "text/plain"
    finally:
        server.stop()


def test_unknown_skill_and_task_failure():
    server = make_server()
    url = server.start()
    try:
        client = A2AClient(url, timeout=10)
        with pytest.raises(A2AError, match="没有技能"):
            client.execute_skill("missing")
        with pytest.raises(A2AError, match="boom"):
            client.execute_skill("boom")
        with pytest.raises(A2AError, match="未找到任务"):
            client.get_task("task-missing")
    finally:
        server.stop()


def test_token_auth():
    server = make_server(token="secret")
    url = server.start()
    try:
        anonymous = A2AClient(url, timeout=10)
        assert anonymous.get_card().name == "研究助手"  # card 公开
        with pytest.raises(A2AError, match="401"):
            anonymous.execute_skill("gather_facts", {"input": "北京"})

        authorized = A2AClient(url, token="secret", timeout=10)
        artifacts = authorized.execute_skill("gather_facts", {"input": "上海"})
        assert "上海" in artifacts[0].content
    finally:
        server.stop()


def test_body_size_limit():
    server = make_server(max_body_bytes=16)
    url = server.start()
    try:
        body = json.dumps(
            {"skill": "gather_facts", "input": {"input": "x" * 100}}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{url}/v1/tasks",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as info:
            urllib.request.urlopen(request, timeout=5)
        assert info.value.code == 413
    finally:
        server.stop()


def test_cancel_terminal_task_rejected():
    server = make_server()
    server.start()
    try:
        task = server.submit_task("gather_facts", {"input": "北京"})
        assert task.status == TaskStatus.COMPLETED
        with pytest.raises(A2AError):
            server.cancel_task(task.id)
    finally:
        server.stop()


def test_server_run_rejects_second_start():
    server = make_server()
    server.start()
    try:
        with pytest.raises(A2AError):
            server.run()
    finally:
        server.stop()


def test_async_task_streaming():
    server = A2AServer("流式助手", "测试流式事件", async_mode=True)

    @server.skill("slow", "慢任务")
    def slow(payload):
        time.sleep(0.08)
        return "流式完成"

    url = server.start()
    try:
        client = A2AClient(url, timeout=10)
        task = client.create_task("slow", {})
        assert task.status.value in ("submitted", "working")
        events = list(client.stream_events(task.id, timeout=5))
        assert events
        assert all(event["type"] == "status" for event in events)
        final = events[-1]["task"]
        assert final["status"] == "completed"
        assert final["artifacts"][0]["content"] == "流式完成"
    finally:
        server.stop()


def test_sync_task_events_are_replayable():
    server = make_server()
    url = server.start()
    try:
        client = A2AClient(url, timeout=10)
        task = client.create_task("gather_facts", {"input": "北京"})
        assert task.status.value == "completed"
        events = list(client.stream_events(task.id, timeout=5))
        assert events[-1]["task"]["status"] == "completed"
    finally:
        server.stop()
