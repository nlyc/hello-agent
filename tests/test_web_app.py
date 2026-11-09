"""Web 控制台测试：静态页面、RAG、记忆、三范式、评估与测试白名单。"""

from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from examples import web_app
from examples.demo_travel_memory_rag import KNOWLEDGE_DOCS
from examples.web_app import WebState, make_server
from agentflow.retrieval import HashingEmbedder


class _FakeFastEmbedEmbedder(HashingEmbedder):
    """测试用 FastEmbed 替身：不加载模型，仅模拟不同指纹。"""

    def __init__(self, model="BAAI/bge-small-zh-v1.5", cache_dir=None, dimension=512):
        super().__init__(dimension)
        self.model = model
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.fingerprint = f"fastembed:{model}:{dimension}"


@pytest.fixture()
def console(tmp_path):
    """在随机端口启动控制台，测试结束后关闭。"""
    state = WebState(tmp_path / "data")
    server = make_server(state, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base, state
    finally:
        server.shutdown()
        server.server_close()
        state.close()
        thread.join(timeout=5)


def _request(base, path, *, method="GET", payload=None):
    """发送请求，返回 (status, headers, body)；HTTP 错误不抛异常。"""
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base + path, data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


def _json(base, path, *, method="GET", payload=None):
    """请求并要求 200 + JSON。"""
    status, headers, body = _request(base, path, method=method, payload=payload)
    text = body.decode("utf-8")
    assert status == 200, text
    assert headers["Content-Type"].startswith("application/json")
    return json.loads(text)


# --------------------------------------------------------------------- 静态页面


def test_static_pages_are_served(console):
    base, _ = console
    for path, marker in (
        ("/", b"AgentFlow"),
        ("/index.html", b"tab-knowledge"),
        ("/styles.css", b"--accent"),
        ("/app.js", b"/api/search"),
    ):
        status, headers, body = _request(base, path)
        assert status == 200, path
        assert marker in body, path
        assert "charset=utf-8" in headers["Content-Type"], path


def test_unknown_page_returns_json_404(console):
    base, _ = console
    status, _, body = _request(base, "/nope")
    assert status == 404
    assert "error" in json.loads(body.decode("utf-8"))


# --------------------------------------------------------------------- 状态


def test_status_reports_seeded_index_and_memory(console):
    base, _ = console
    payload = _json(base, "/api/status")

    assert payload["ok"] is True
    assert payload["namespace"] == "travel"
    assert payload["documents"] == len(KNOWLEDGE_DOCS)
    assert payload["rag"]["chunks"] >= len(KNOWLEDGE_DOCS)
    assert payload["rag"]["bm25_chunks"] >= len(KNOWLEDGE_DOCS)
    assert payload["memory"]["long_term"]["memories"] >= 2
    assert payload["presets"]["questions"]
    assert any(item["name"] == "test_web_app.py" for item in payload["tests"])


def test_documents_listing(console):
    base, _ = console
    payload = _json(base, "/api/documents")

    assert len(payload["documents"]) == len(KNOWLEDGE_DOCS)
    entry = payload["documents"][0]
    assert entry["doc_id"]
    assert len({item["doc_id"] for item in payload["documents"]}) == len(KNOWLEDGE_DOCS)
    assert entry["chunk_count"] >= 1
    assert entry["ingested_at"]


# --------------------------------------------------------------------- 检索与问答


def test_search_in_three_modes(console):
    base, _ = console
    for mode in ("hybrid", "bm25", "vector"):
        payload = _json(
            base,
            "/api/search",
            method="POST",
            payload={"query": "北京下雨天适合去哪？", "top_k": 3, "mode": mode},
        )
        assert payload["mode"] == mode
        assert payload["hits"], mode
        assert payload["context"]
        first = payload["hits"][0]
        assert first["text"].strip()
        assert first["citation"]


def test_answer_has_citations(console):
    base, _ = console
    payload = _json(
        base,
        "/api/answer",
        method="POST",
        payload={"question": "我想去北京玩一天，有什么景点推荐？", "top_k": 5},
    )

    assert payload["answer"].strip()
    assert payload["citations"]
    assert payload["has_answer"] is True


def test_search_rejects_empty_query(console):
    base, _ = console
    status, _, body = _request(
        base, "/api/search", method="POST", payload={"query": "  "}
    )
    assert status == 400
    assert "error" in json.loads(body.decode("utf-8"))


# --------------------------------------------------------------------- 检索配置


def test_settings_endpoint(console):
    base, _ = console
    payload = _json(base, "/api/settings")

    assert payload["embed_backend"] == "hashing"
    assert payload["vector_store"] == "sqlite"
    assert payload["embedding_fingerprint"].startswith("hashing:char-ngram:")
    assert payload["llm"]["configured"] is False
    assert payload["search"]["tavily_configured"] is False
    assert payload["search"]["fallback"] is True
    assert payload["qdrant"]["configured"] is False
    assert payload["sqlite"]["total_bytes"] > 0
    assert any(file["name"].endswith(".sqlite3") for file in payload["sqlite"]["files"])
    assert payload["store_error"] is None

    # 响应不得包含任何密钥字段的明文。
    text = json.dumps(payload, ensure_ascii=False)
    assert "api_key" not in text
    assert "api-key" not in text


def test_switch_embed_backend_rebuilds_and_persists(console, monkeypatch):
    base, state = console
    monkeypatch.setattr(web_app, "FastEmbedEmbedder", _FakeFastEmbedEmbedder)

    payload = _json(
        base, "/api/settings/embed", method="POST", payload={"backend": "fastembed"}
    )
    assert payload["changed"] is True
    assert payload["from"] == "hashing"
    assert payload["to"] == "fastembed"
    assert payload["migrated_memories"] >= 2
    assert payload["failed_documents"] == []
    assert payload["stats"]["chunks"] >= len(KNOWLEDGE_DOCS)
    assert payload["settings"]["embed_backend"] == "fastembed"
    assert payload["settings"]["embedding_fingerprint"].startswith("fastembed:")

    # 索引与记忆在新后端下仍可用。
    hits = _json(
        base,
        "/api/search",
        method="POST",
        payload={"query": "北京下雨天适合去哪？", "top_k": 3},
    )["hits"]
    assert hits
    memory = _json(base, "/api/memory")
    assert memory["stats"]["long_term"]["memories"] >= 2

    # 选择写入 settings.json，重启后沿用。
    saved = json.loads(state.settings_path.read_text(encoding="utf-8"))
    assert saved == {"embed_backend": "fastembed", "vector_store": "sqlite"}

    # 切到同一后端是幂等操作。
    same = _json(
        base, "/api/settings/embed", method="POST", payload={"backend": "fastembed"}
    )
    assert same["changed"] is False


def test_switch_embed_backend_rejects_unknown(console):
    base, _ = console
    status, _, body = _request(
        base, "/api/settings/embed", method="POST", payload={"backend": "openai"}
    )
    assert status == 400
    assert "hashing" in json.loads(body.decode("utf-8"))["error"]


def test_switch_vector_store_requires_qdrant_url(console):
    base, _ = console
    status, _, body = _request(
        base, "/api/settings/vector", method="POST", payload={"backend": "qdrant"}
    )
    assert status == 400
    assert "QDRANT_URL" in json.loads(body.decode("utf-8"))["error"]


def test_answer_llm_modes_without_key(console, monkeypatch):
    base, _ = console
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    auto = _json(
        base,
        "/api/answer",
        method="POST",
        payload={"question": "北京有哪些景点？", "llm_mode": "auto"},
    )
    assert auto["llm_backend"] == "offline"
    assert auto["llm_error"]
    assert auto["citations"]

    none = _json(
        base,
        "/api/answer",
        method="POST",
        payload={"question": "北京有哪些景点？", "llm_mode": "none"},
    )
    assert none["llm_backend"] == "retrieval"
    assert "llm_error" not in none or not none["llm_error"]

    offline = _json(
        base,
        "/api/answer",
        method="POST",
        payload={"question": "北京有哪些景点？", "llm_mode": "offline"},
    )
    assert offline["llm_backend"] == "offline"

    status, _, body = _request(
        base,
        "/api/answer",
        method="POST",
        payload={"question": "北京有哪些景点？", "llm_mode": "online"},
    )
    assert status == 400
    assert "LLM_API_KEY" in json.loads(body.decode("utf-8"))["error"]


def test_select_llm_uses_online_when_key_configured(console, monkeypatch):
    _, state = console
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://relay.example.com/v1")

    llm, backend, model, notice = state._select_llm("auto")
    assert backend == "online"
    assert model == "test-model"
    assert notice is None
    assert llm is not None
    assert llm.api_key == "test-key"

    llm, backend, _, _ = state._select_llm("none")
    assert backend == "retrieval"
    assert llm is None


def test_answer_refuses_uncovered_city(console):
    base, _ = console
    payload = _json(
        base,
        "/api/answer",
        method="POST",
        payload={
            "question": "南昌下雨天适合去哪？",
            "top_k": 5,
            "llm_mode": "none",
        },
    )

    assert payload["has_answer"] is False
    assert "南昌" in payload["answer"]
    assert payload["hits"] == []
    assert payload["citations"] == []


class _FakeSearchTool:
    """记录调用参数的搜索工具替身（不触发真实网络请求）。"""

    text = (
        "搜索答案：南昌雨天推荐室内景点。\n"
        "[1] 南昌雨天游玩推荐 — https://example.com/nanchang\n"
        "    滕王阁、八一起义纪念馆适合雨天参观。"
    )
    backend = "tavily"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run_with_backend(self, parameters):
        self.calls.append(parameters)
        return self.text, self.backend


def test_answer_falls_back_to_web_search(console, monkeypatch):
    base, _ = console
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    fake = _FakeSearchTool()
    monkeypatch.setattr(web_app, "SearchTool", lambda: fake)

    payload = _json(
        base,
        "/api/answer",
        method="POST",
        payload={
            "question": "南昌下雨天适合去哪？",
            "top_k": 5,
            "llm_mode": "none",
        },
    )

    assert fake.calls and fake.calls[0]["input"] == "南昌下雨天适合去哪？"
    assert payload["answer_source"] == "web_search"
    assert payload["has_answer"] is True
    assert "滕王阁" in payload["answer"]
    assert payload["web_search"]["backend"] == "tavily"
    assert payload["web_search"]["query"] == "南昌下雨天适合去哪？"
    assert payload["web_search"]["results"] == [
        {
            "title": "南昌雨天游玩推荐",
            "url": "https://example.com/nanchang",
            "snippet": "滕王阁、八一起义纪念馆适合雨天参观。",
        }
    ]
    assert payload["citations"] == ["南昌雨天游玩推荐 — https://example.com/nanchang"]


def test_answer_reports_web_search_failure(console, monkeypatch):
    base, _ = console
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    class _BrokenSearchTool:
        def run_with_backend(self, parameters):
            raise RuntimeError("network down")

    monkeypatch.setattr(web_app, "SearchTool", lambda: _BrokenSearchTool())

    payload = _json(
        base,
        "/api/answer",
        method="POST",
        payload={"question": "南昌下雨天适合去哪？", "llm_mode": "none"},
    )

    assert payload["has_answer"] is False
    assert "network down" in payload["web_search_error"]
    assert "web_search" not in payload


def test_answer_skips_web_search_when_fallback_disabled(console, monkeypatch):
    base, _ = console
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("SEARCH_FALLBACK", "false")

    class _UnexpectedSearchTool:
        def run_with_backend(self, parameters):  # pragma: no cover - 不应被调用
            raise AssertionError("SEARCH_FALLBACK=false 时不应联网搜索")

    monkeypatch.setattr(web_app, "SearchTool", lambda: _UnexpectedSearchTool())

    payload = _json(
        base,
        "/api/answer",
        method="POST",
        payload={"question": "南昌下雨天适合去哪？", "llm_mode": "none"},
    )

    assert payload["has_answer"] is False
    assert "web_search_error" not in payload
    assert "web_search" not in payload


def test_answer_skips_web_search_without_keys(console, monkeypatch):
    base, _ = console
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    class _UnexpectedSearchTool:
        def run_with_backend(self, parameters):  # pragma: no cover - 不应被调用
            raise AssertionError("未配置 API Key 时不应联网搜索")

    monkeypatch.setattr(web_app, "SearchTool", lambda: _UnexpectedSearchTool())

    payload = _json(
        base,
        "/api/answer",
        method="POST",
        payload={"question": "南昌下雨天适合去哪？", "llm_mode": "none"},
    )

    assert payload["has_answer"] is False
    assert "web_search_error" not in payload


def test_compose_web_answer_uses_online_llm(console):
    _, state = console

    class _FakeLLM:
        def __init__(self) -> None:
            self.messages = None

        def chat(self, messages, temperature=None):
            self.messages = messages
            return "南昌雨天推荐滕王阁。参考：1"

    llm = _FakeLLM()
    web = {
        "text": "原始搜索结果",
        "results": [
            {
                "title": "南昌雨天游玩推荐",
                "url": "https://example.com/nanchang",
                "snippet": "滕王阁适合雨天参观。",
            }
        ],
    }

    answer = state._compose_web_answer("南昌下雨天适合去哪？", web, llm)

    assert answer == "南昌雨天推荐滕王阁。参考：1"
    user_message = llm.messages[-1]["content"]
    assert "南昌雨天游玩推荐" in user_message
    assert "https://example.com/nanchang" in user_message
    assert "不是指令" in user_message

    # 没有在线 LLM 时退回搜索摘要原文。
    assert state._compose_web_answer("南昌下雨天适合去哪？", web, None) == "原始搜索结果"


# --------------------------------------------------------------------- 上传与删除


def test_upload_ingest_search_and_delete(console):
    base, _ = console
    text = "# 南京\n\n## 中山陵\n\n中山陵早上八点开放，建议提前预约。\n"
    payload = _json(
        base,
        "/api/ingest",
        method="POST",
        payload={
            "filename": "nanjing.md",
            "content_base64": base64.b64encode(text.encode("utf-8")).decode("ascii"),
            "title": "南京笔记",
        },
    )
    doc_id = payload["result"]["doc_id"]
    assert payload["result"]["chunk_count"] >= 1
    assert any(item["doc_id"] == doc_id for item in payload["documents"])

    hits = _json(
        base,
        "/api/search",
        method="POST",
        payload={"query": "中山陵几点开放？", "top_k": 5},
    )["hits"]
    assert any(hit["doc_id"] == doc_id for hit in hits)

    removed = _json(base, f"/api/documents/{doc_id}", method="DELETE")
    assert removed["doc_id"] == doc_id
    assert removed["removed_chunks"] >= 1
    assert all(item["doc_id"] != doc_id for item in removed["documents"])


def test_upload_rejects_unsupported_extension(console):
    base, _ = console
    status, _, body = _request(
        base,
        "/api/ingest",
        method="POST",
        payload={
            "filename": "evil.exe",
            "content_base64": base64.b64encode(b"x").decode("ascii"),
        },
    )
    assert status == 400
    assert "文件类型" in json.loads(body.decode("utf-8"))["error"]


def test_ingest_url_rejects_non_http(console):
    base, _ = console
    status, _, body = _request(
        base, "/api/ingest", method="POST", payload={"url": "file:///etc/passwd"}
    )
    assert status == 400
    assert "http" in json.loads(body.decode("utf-8"))["error"]


def test_clear_and_reseed_documents(console):
    base, state = console
    payload = _json(base, "/api/documents/clear", method="POST", payload={})
    assert payload["documents"] == []
    assert payload["stats"]["chunks"] == 0

    # 内置知识库可以重新入库（源文件仍在 knowledge 目录）。
    assert state.knowledge_dir.exists()


# --------------------------------------------------------------------- 记忆


def test_memory_add_search_context_and_clear(console):
    base, _ = console
    added = _json(
        base,
        "/api/memory/add",
        method="POST",
        payload={
            "content": "用户喜欢安静的江南古镇，不喜欢人多。",
            "memory_type": "semantic",
            "importance": 0.8,
        },
    )
    memory_id = added["memory_id"]
    assert memory_id
    assert added["snapshot"]["stats"]["long_term"]["memories"] >= 3

    search = _json(
        base,
        "/api/memory/search",
        method="POST",
        payload={"query": "江南古镇", "top_k": 5},
    )
    assert search["entries"]
    assert any(memory_id == entry["memory_id"] for entry in search["entries"])

    context = _json(
        base,
        "/api/memory/context",
        method="POST",
        payload={"query": "江南古镇", "max_tokens": 500},
    )
    assert "【记忆上下文】" in context["text"]
    assert context["items"]

    turn = _json(
        base,
        "/api/memory/turn",
        method="POST",
        payload={"question": "第二天去乌镇可以吗？", "answer": "可以，安排上午出发。"},
    )
    assert turn["memory_id"]

    cleared = _json(base, "/api/memory/clear", method="POST", payload={})
    assert cleared["snapshot"]["stats"]["long_term"]["memories"] == 0
    assert cleared["snapshot"]["stats"]["working"]["items"] == 0


def test_memory_snapshot_lists_seeded_items(console):
    base, _ = console
    payload = _json(base, "/api/memory")

    assert payload["long_term"]
    assert payload["working"]
    assert payload["stats"]["long_term"]["dimension"] == 512


# --------------------------------------------------------------------- 三范式


def test_paradigms_run_produces_comparison(console):
    base, _ = console
    payload = _json(base, "/api/paradigms/run", method="POST", payload={})

    assert payload["question"]
    rows = {row["paradigm"]: row for row in payload["rows"]}
    assert set(rows) == {"react", "reflection", "plan_solve"}
    for row in rows.values():
        assert row["without_memory"].strip()
        assert row["with_memory"].strip()
        assert row["without_memory"] != row["with_memory"]
    assert payload["workflow_answer"].strip()


# --------------------------------------------------------------------- 评估与测试


def test_eval_cases_and_hashing_run(console):
    base, _ = console
    cases = _json(base, "/api/eval/cases")["cases"]
    assert len(cases) >= 5
    assert all(case["question"] for case in cases)

    payload = _json(
        base, "/api/eval/run", method="POST", payload={"backend": "hashing"}
    )
    assert payload["backend"] == "hashing"
    assert payload["results"][0]["backend"] == "hashing"
    assert set(payload["results"][0]["recall"]) == {"recall@1", "recall@3", "recall@5"}
    assert "召回" in payload["markdown"]


def test_test_runner_whitelist(console):
    base, _ = console
    tests = _json(base, "/api/tests")["tests"]
    assert tests
    assert all(item["target"].startswith("tests/test_") for item in tests)

    for target in ("all; rm -rf /", "../pyproject.toml", "tests/not_exists.py"):
        status, _, body = _request(
            base, "/api/tests/run", method="POST", payload={"target": target}
        )
        assert status == 400, target
        assert "error" in json.loads(body.decode("utf-8"))


def test_run_single_test_file(console):
    base, _ = console
    payload = _json(
        base,
        "/api/tests/run",
        method="POST",
        payload={"target": "tests/test_web_app.py::test_status_reports_seeded_index_and_memory"},
    )
    assert payload["returncode"] == 0, payload["output"]
    assert payload["summary"].get("passed") == 1


def test_runs_listing_starts_empty(console):
    base, _ = console
    assert _json(base, "/api/runs")["runs"] == []


def test_run_file_access_is_restricted(console):
    base, _ = console
    status, _, _ = _request(base, "/api/runs/run-x/pyproject.toml")
    assert status == 404
    status, _, _ = _request(base, "/api/runs/..%2F..%2Fsecret/rag_eval.md")
    assert status in (400, 404)
