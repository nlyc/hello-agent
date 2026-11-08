"""AgentFlow 本地 Web 控制台：在浏览器里操作 RAG、记忆、三范式、Demo、评估与测试。

零新增依赖：仅使用标准库 ``http.server``，静态页面位于 ``examples/webui/``。
服务默认只监听 ``127.0.0.1``，所有数据写入 ``.data/webui``。

启动：

    py -3.14 -m examples.web_app

然后打开 http://127.0.0.1:8000 。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from agentflow.memory import LongTermMemory, MemoryManager
from agentflow.retrieval import (
    BM25Index,
    Chunk,
    HashingEmbedder,
    RAGPipeline,
    SQLiteVectorStore,
    load_document,
)
from agentflow.trace import Tracer
from examples.demo_travel_memory_rag import (
    DEMO_USER_ID,
    KNOWLEDGE_DOCS,
    MAIN_QUESTION,
    TravelDemoLLM,
    demo_paradigms,
    demo_workflow,
    run_demo,
    seed_memory,
)
from scripts.eval_rag import load_cases, run_eval

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBUI_DIR = Path(__file__).resolve().parent / "webui"
DEFAULT_DATA_DIR = PROJECT_ROOT / ".data" / "webui"

SUPPORTED_UPLOADS = {".md", ".markdown", ".txt", ".pdf", ".docx"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_BODY_BYTES = 40 * 1024 * 1024
MAX_TEXT_CHARS = 200_000
ALLOWED_RUN_FILES = ("rag_eval.md", "memory.json", "trace.jsonl")
TEST_TARGET_PATTERN = re.compile(
    r"^tests/test_[A-Za-z0-9_]+\.py(?:::test_[A-Za-z0-9_]+){0,2}$"
)

PRESET_QUESTIONS = [
    "我想去北京玩一天，有什么景点推荐？",
    "北京下雨天适合去哪？",
    "香山秋天什么时候去最好？",
    "上海外滩夜景几点最好？",
    "杭州西湖怎么玩？",
    "成都有什么必去景点？",
]
PRESET_MEMORY_QUERIES = [
    "北京怎么安排",
    "雨天去哪些室内景点",
    "我原来第二天的安排",
    "用户偏好",
]


class ApiError(Exception):
    """带 HTTP 状态码的业务异常。"""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _now_text() -> str:
    """当前时间字符串，用于目录名与展示。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    """把输入安全地转换为指定区间的整数。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


class WebState:
    """Web 控制台的全部可变状态。

    ``Lock`` 保护索引/记忆的读写；``job_lock`` 串行化 Demo、评估与测试等
    耗时任务，避免两个浏览器标签页同时跑同一份数据库。
    """

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        *,
        namespace: str = "travel",
        user_id: str = DEMO_USER_ID,
        seed: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.namespace = namespace
        self.user_id = user_id
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir = self.data_dir / "uploads"
        self.knowledge_dir = self.data_dir / "knowledge"
        self.runs_dir = self.data_dir / "runs"
        self.manifest_path = self.data_dir / "documents.json"
        for directory in (self.uploads_dir, self.knowledge_dir, self.runs_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.lock = threading.RLock()
        self.job_lock = threading.Lock()
        self.embedder = HashingEmbedder()
        self.rag_store = SQLiteVectorStore(self.data_dir / "rag.sqlite3")
        self.bm25 = BM25Index()
        self.pipeline = RAGPipeline(
            self.embedder,
            self.rag_store,
            self.bm25,
            namespace=self.namespace,
        )
        self.long_term = LongTermMemory(
            self.embedder,
            path=self.data_dir / "memory.sqlite3",
            user_id=self.user_id,
        )
        self.memory = MemoryManager(self.embedder, self.long_term, user_id=self.user_id)
        self.documents: list[dict[str, Any]] = self._load_manifest()

        if seed:
            with self.lock:
                self._seed()

    # --- 初始化与清单 -----------------------------------------------------

    def _load_manifest(self) -> list[dict[str, Any]]:
        """读取文档清单；文件损坏时退化为空列表。"""
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return payload if isinstance(payload, list) else []

    def _save_manifest(self) -> None:
        """原子写入文档清单。"""
        temp = self.manifest_path.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(self.documents, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp.replace(self.manifest_path)

    def _record_document(
        self, result: Any, *, display_source: str | None = None
    ) -> dict[str, Any]:
        """把入库结果写入清单（按 doc_id 覆盖）。"""
        entry = result.to_dict()
        entry["source"] = display_source or entry.get("source_uri", "")
        entry["ingested_at"] = _now_text()
        self.documents = [
            item for item in self.documents if item.get("doc_id") != entry["doc_id"]
        ]
        self.documents.append(entry)
        self.documents.sort(key=lambda item: item.get("ingested_at", ""))
        self._save_manifest()
        return entry

    def _seed(self) -> None:
        """首次启动时写入内置知识库与演示记忆。"""
        for item in KNOWLEDGE_DOCS:
            path = self.knowledge_dir / f"{item['doc_id']}.md"
            if not path.exists():
                path.write_text(item["text"], encoding="utf-8")
            result = self.pipeline.ingest(
                str(path),
                namespace=self.namespace,
                title=item["title"],
                metadata={"city": item["city"], "origin": "builtin"},
            )
            if any(
                doc.get("doc_id") == result.doc_id for doc in self.documents
            ):
                continue
            self._record_document(result, display_source=path.name)

        if self.long_term.count() == 0:
            seed_memory(self.memory)

        if len(self.bm25) == 0 and self.rag_store.count(self.namespace) > 0:
            self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        """进程重启后按持久化向量库重建内存 BM25 索引。"""
        total = self.rag_store.count(self.namespace)
        if total <= 0:
            return
        probe = [0.0] * len(self.embedder.embed_query(""))
        hits = self.rag_store.search(
            probe, top_k=total, filters={"namespace": self.namespace}
        )
        chunks = [
            Chunk(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                namespace=hit.namespace,
                text=hit.text,
                heading_path=hit.heading_path,
                page=hit.page,
                content_hash=hit.content_hash or "",
                metadata=dict(hit.metadata or {}),
            )
            for hit in hits
        ]
        self.bm25.add(chunks)

    # --- 状态 -------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """汇总索引、记忆与可用测试信息。"""
        with self.lock:
            return {
                "ok": True,
                "namespace": self.namespace,
                "user_id": self.user_id,
                "data_dir": str(self.data_dir),
                "rag": self.pipeline.stats(),
                "memory": self.memory.stats(),
                "documents": len(self.documents),
                "presets": {
                    "questions": PRESET_QUESTIONS,
                    "memory_queries": PRESET_MEMORY_QUERIES,
                    "main_question": MAIN_QUESTION,
                },
                "tests": self.list_tests(),
            }

    # --- 知识库 -----------------------------------------------------------

    def list_documents(self) -> dict[str, Any]:
        """返回文档清单与索引统计。"""
        with self.lock:
            return {
                "documents": list(self.documents),
                "stats": self.pipeline.stats(),
            }

    def ingest_upload(
        self, filename: str, content_base64: str, title: str | None = None
    ) -> dict[str, Any]:
        """接收浏览器上传的 base64 文件并入库。"""
        safe_name = Path(str(filename or "")).name
        suffix = Path(safe_name).suffix.lower()
        if not safe_name or suffix not in SUPPORTED_UPLOADS:
            raise ApiError(
                "不支持的文件类型；仅支持 .md / .txt / .pdf / .docx"
            )
        try:
            data = base64.b64decode(content_base64 or "", validate=True)
        except (ValueError, TypeError) as exc:
            raise ApiError(f"文件内容不是合法 base64：{exc}") from exc
        if not data:
            raise ApiError("文件为空")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ApiError("文件超过 25 MB 限制", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)

        target = self.uploads_dir / safe_name
        target.write_bytes(data)
        with self.lock:
            result = self.pipeline.ingest(
                str(target),
                namespace=self.namespace,
                title=(title or "").strip() or None,
                metadata={"origin": "upload", "original_filename": safe_name},
            )
            entry = self._record_document(result, display_source=safe_name)
            return {"result": entry, "documents": list(self.documents), "stats": self.pipeline.stats()}

    def ingest_url(self, url: str, title: str | None = None) -> dict[str, Any]:
        """抓取公网 URL、落盘为 Markdown 后入库，便于重启后离线重建。"""
        url = str(url or "").strip()
        if not url.startswith(("http://", "https://")):
            raise ApiError("URL 必须以 http:// 或 https:// 开头")
        try:
            document = load_document(
                url,
                namespace=self.namespace,
                title=(title or "").strip() or None,
                metadata={"origin": "url"},
            )
        except Exception as exc:
            raise ApiError(f"抓取失败：{exc}") from exc

        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        target = self.uploads_dir / f"url_{digest}.md"
        target.write_text(document.text, encoding="utf-8")
        with self.lock:
            result = self.pipeline.ingest(
                str(target),
                namespace=self.namespace,
                title=document.title,
                metadata={"origin": "url", "url": url},
            )
            entry = self._record_document(result, display_source=url)
            return {"result": entry, "documents": list(self.documents), "stats": self.pipeline.stats()}

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        """删除文档的全部 chunk、BM25 条目与清单记录。"""
        doc_id = str(doc_id or "").strip()
        if not doc_id:
            raise ApiError("doc_id 不能为空")
        with self.lock:
            removed = self.rag_store.delete(doc_id, self.namespace)
            self.bm25.remove_doc(doc_id, self.namespace)
            self.documents = [
                item for item in self.documents if item.get("doc_id") != doc_id
            ]
            self._save_manifest()
            return {
                "doc_id": doc_id,
                "removed_chunks": removed,
                "documents": list(self.documents),
                "stats": self.pipeline.stats(),
            }

    def clear_documents(self) -> dict[str, Any]:
        """清空索引与清单（不删除磁盘上的源文件）。"""
        with self.lock:
            self.rag_store.clear(self.namespace)
            self.bm25.clear()
            self.documents = []
            self._save_manifest()
            return {"documents": [], "stats": self.pipeline.stats()}

    def search(
        self, query: str, top_k: int = 5, mode: str = "hybrid"
    ) -> dict[str, Any]:
        """执行 hybrid / bm25 / vector 三选一检索。"""
        query = str(query or "").strip()
        if not query:
            raise ApiError("检索问题不能为空")
        limit = _clamp_int(top_k, 5, 1, 20)
        if mode not in {"hybrid", "bm25", "vector"}:
            raise ApiError("mode 必须是 hybrid / bm25 / vector")
        with self.lock:
            if mode == "bm25":
                hits = self.bm25.search(
                    query, top_k=limit, filters={"namespace": self.namespace}
                )
            elif mode == "vector":
                hits = self.rag_store.search(
                    self.embedder.embed_query(query),
                    top_k=limit,
                    filters={"namespace": self.namespace},
                )
            else:
                hits = self.pipeline.search(query, top_k=limit)
            return {
                "query": query,
                "mode": mode,
                "top_k": limit,
                "hits": [hit.to_dict() for hit in hits],
                "context": self.pipeline.build_context(hits),
            }

    def answer(
        self, question: str, top_k: int = 5, use_llm: bool = True
    ) -> dict[str, Any]:
        """基于检索结果生成带引用的回答。"""
        question = str(question or "").strip()
        if not question:
            raise ApiError("问题不能为空")
        limit = _clamp_int(top_k, 5, 1, 20)
        with self.lock:
            result = self.pipeline.answer(
                question,
                llm=TravelDemoLLM() if use_llm else None,
                top_k=limit,
            )
            return result.to_dict()

    # --- 记忆 -------------------------------------------------------------

    def memory_snapshot(self, limit: int = 50) -> dict[str, Any]:
        """返回工作记忆与长期记忆的当前快照。"""
        count = _clamp_int(limit, 50, 1, 200)
        with self.lock:
            working = [item.to_dict() for item in self.memory.working.all_items()]
            long_term = [
                hit.to_dict() for hit in self.long_term.search("", top_k=count)
            ]
            return {
                "stats": self.memory.stats(),
                "working": working,
                "long_term": long_term,
                "user_id": self.user_id,
            }

    def memory_add(
        self,
        content: str,
        *,
        memory_type: str = "semantic",
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """写入一条记忆（同时进入工作记忆与长期记忆）。"""
        content = str(content or "").strip()
        if not content:
            raise ApiError("记忆内容不能为空")
        try:
            weight = float(importance)
        except (TypeError, ValueError):
            weight = 0.5
        weight = max(0.0, min(1.0, weight))
        memory_type = str(memory_type or "semantic").strip() or "semantic"
        with self.lock:
            memory_id = self.memory.add(
                content,
                importance=weight,
                memory_type=memory_type,
                metadata=dict(metadata or {}),
            )
            return {"memory_id": memory_id, "snapshot": self.memory_snapshot()}

    def memory_search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """合并检索工作记忆与长期记忆。"""
        query = str(query or "").strip()
        if not query:
            raise ApiError("检索问题不能为空")
        limit = _clamp_int(top_k, 5, 1, 50)
        with self.lock:
            return {
                "query": query,
                "entries": self.memory.search(query, top_k=limit),
            }

    def memory_context(self, query: str, max_tokens: int = 1500) -> dict[str, Any]:
        """构建注入 Agent 的记忆上下文文本。"""
        query = str(query or "").strip()
        budget = _clamp_int(max_tokens, 1500, 1, 8000)
        with self.lock:
            context = self.memory.build_context(query, max_tokens=budget)
            return {
                "query": query,
                "max_tokens": budget,
                "text": context.text,
                "items": list(context.items),
                "working_items": [
                    item.to_dict() for item in context.working_items
                ],
                "long_term_hits": [hit.to_dict() for hit in context.long_term_hits],
            }

    def memory_turn(
        self, question: str, answer: str, importance: float = 0.5
    ) -> dict[str, Any]:
        """记录一轮对话为 episodic 记忆。"""
        question = str(question or "").strip()
        if not question:
            raise ApiError("用户输入不能为空")
        try:
            weight = max(0.0, min(1.0, float(importance)))
        except (TypeError, ValueError):
            weight = 0.5
        with self.lock:
            memory_id = self.memory.record_turn(
                question, str(answer or ""), importance=weight
            )
            return {"memory_id": memory_id, "snapshot": self.memory_snapshot()}

    def memory_clear(self) -> dict[str, Any]:
        """清空当前用户的工作记忆与长期记忆。"""
        with self.lock:
            self.memory.clear()
            return {"snapshot": self.memory_snapshot()}

    # --- 三范式与 Demo ----------------------------------------------------

    def run_paradigms(self) -> dict[str, Any]:
        """用独立的临时记忆跑三范式对照，避免污染控制台记忆。"""
        with self.job_lock:
            embedder = HashingEmbedder()
            store = LongTermMemory(embedder, path=":memory:", user_id="demo_run")
            try:
                memory = MemoryManager(embedder, store, user_id="demo_run")
                seed_memory(memory)
                tracer = Tracer()
                rows = demo_paradigms(memory, tracer)
                workflow = demo_workflow(tracer)
                return {
                    "question": MAIN_QUESTION,
                    "rows": rows,
                    "workflow_answer": workflow,
                }
            finally:
                store.close()

    def run_demo_job(
        self,
        *,
        ingest: str | None = None,
        query: str | None = None,
        embed_backend: str = "hashing",
    ) -> dict[str, Any]:
        """在独立目录运行完整 Demo，返回报告与产物列表。"""
        if embed_backend not in {"hashing", "fastembed"}:
            raise ApiError("embed_backend 必须是 hashing 或 fastembed")
        ingest = (ingest or "").strip() or None
        query = (query or "").strip() or None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"run-{stamp}-{int(time.time() * 1000) % 1000:03d}"
        out_dir = self.runs_dir / run_id
        with self.job_lock:
            try:
                result = run_demo(
                    out_dir,
                    embed_backend=embed_backend,
                    ingest=ingest,
                    query=query,
                )
            except Exception as exc:
                raise ApiError(f"Demo 运行失败：{exc}") from exc
        return {
            "run_id": run_id,
            "out_dir": result["out_dir"],
            "report": result["report"],
            "stats": result["stats"],
            "paradigms": result["paradigms"],
            "uploads": result["uploads"],
            "rag_compare": result["rag_compare"],
            "itinerary": result["itinerary"],
            "cross_session": result["cross_session"],
            "workflow_answer": result["workflow_answer"],
            "artifacts": self._run_files(out_dir),
        }

    def _run_files(self, out_dir: Path) -> list[dict[str, Any]]:
        """列出某次 Demo 允许下载的产物。"""
        files = []
        for name in ALLOWED_RUN_FILES:
            path = out_dir / name
            if path.is_file():
                files.append(
                    {
                        "name": name,
                        "size": path.stat().st_size,
                        "url": f"/api/runs/{out_dir.name}/{name}",
                        "download_url": f"/api/runs/{out_dir.name}/{name}?download=1",
                    }
                )
        return files

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """按时间倒序列出历史 Demo 运行。"""
        runs = []
        for path in sorted(
            (item for item in self.runs_dir.iterdir() if item.is_dir()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )[: max(1, limit)]:
            runs.append(
                {
                    "run_id": path.name,
                    "created_at": datetime.fromtimestamp(
                        path.stat().st_mtime
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "files": self._run_files(path),
                }
            )
        return runs

    # --- 评估与测试 -------------------------------------------------------

    def eval_cases(self) -> list[dict[str, Any]]:
        """返回评估集内容。"""
        try:
            return load_cases(PROJECT_ROOT / "data" / "eval" / "travel_qa.jsonl")
        except Exception as exc:
            raise ApiError(f"读取评估集失败：{exc}") from exc

    def run_eval_job(self, backend: str = "all", online: bool = False) -> dict[str, Any]:
        """运行 RAG 评估（召回指标 + 回答对照）。"""
        if backend not in {"all", "bm25", "hashing", "fastembed"}:
            raise ApiError("backend 必须是 all / bm25 / hashing / fastembed")
        with self.job_lock:
            try:
                result = run_eval(backend, online=bool(online))
            except Exception as exc:
                raise ApiError(f"评估失败：{exc}") from exc
        return {
            "backend": backend,
            "markdown": result["markdown"],
            "results": result["results"],
            "cases": result["cases"],
            "skipped": result["skipped"],
        }

    def list_tests(self) -> list[dict[str, Any]]:
        """列出可单独运行的测试文件。"""
        tests_dir = PROJECT_ROOT / "tests"
        items = []
        for path in sorted(tests_dir.glob("test_*.py")):
            label = path.stem
            try:
                first_line = path.read_text(encoding="utf-8").splitlines()[0]
            except (OSError, IndexError):
                first_line = ""
            description = first_line.strip().strip("\"'").strip()
            items.append(
                {
                    "target": f"tests/{path.name}",
                    "name": path.name,
                    "label": label,
                    "description": description,
                }
            )
        return items

    def run_tests(self, target: str = "all") -> dict[str, Any]:
        """在子进程中运行白名单内的 pytest 目标。"""
        target = (target or "all").strip()
        if target == "all":
            arguments = [sys.executable, "-m", "pytest", "-q"]
        else:
            if not TEST_TARGET_PATTERN.fullmatch(target):
                raise ApiError("测试目标必须形如 tests/test_xxx.py[::test_name]")
            file_part = target.split("::", 1)[0]
            if not (PROJECT_ROOT / file_part).is_file():
                raise ApiError(f"测试文件不存在：{file_part}")
            arguments = [sys.executable, "-m", "pytest", "-q", target]

        started = time.perf_counter()
        with self.job_lock:
            try:
                completed = subprocess.run(
                    arguments,
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=900,
                )
            except subprocess.TimeoutExpired as exc:
                raise ApiError("测试运行超过 15 分钟，已终止", 504) from exc
        duration = round(time.perf_counter() - started, 2)
        output = (completed.stdout or "") + (completed.stderr or "")
        if len(output) > MAX_TEXT_CHARS:
            output = output[-MAX_TEXT_CHARS:]
        return {
            "target": target,
            "returncode": completed.returncode,
            "duration_ms": int(duration * 1000),
            "output": output,
            "summary": _parse_pytest_summary(output),
        }

    # --- 生命周期 ---------------------------------------------------------

    def close(self) -> None:
        """关闭 SQLite 连接。"""
        with self.lock:
            try:
                self.long_term.close()
            finally:
                self.rag_store.close()

    def __enter__(self) -> "WebState":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _parse_pytest_summary(output: str) -> dict[str, int]:
    """从 pytest 末尾行提取 passed / failed / skipped / errors 数量。"""
    counts: dict[str, int] = {}
    for name in ("passed", "failed", "skipped", "error", "errors"):
        matches = re.findall(rf"(\d+) {name}\b", output)
        if matches:
            key = "errors" if name == "error" else name
            counts[key] = int(matches[-1])
    return counts


class ConsoleServer(ThreadingHTTPServer):
    """携带共享状态的 HTTP 服务。"""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        *,
        state: WebState,
        webui_dir: Path = WEBUI_DIR,
    ) -> None:
        self.state = state
        self.webui_dir = Path(webui_dir)
        super().__init__(server_address, handler)


class ConsoleHandler(BaseHTTPRequestHandler):
    """静态页面 + JSON API。"""

    server_version = "AgentFlowConsole/1.0"
    protocol_version = "HTTP/1.1"

    # --- 基础响应 ---------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:
        """只记录错误，避免轮询把终端刷满。"""
        return

    @property
    def state(self) -> WebState:
        return self.server.state  # type: ignore[attr-defined]

    def _send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _send_bytes(
        self,
        data: bytes,
        content_type: str,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length > MAX_BODY_BYTES:
            raise ApiError("请求体过大（上限 25 MB 文件）", 413)
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ApiError(f"请求体不是合法 JSON：{exc}") from exc
        if not isinstance(payload, dict):
            raise ApiError("请求体必须是 JSON 对象")
        return payload

    # --- 路由 -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path.startswith("/api/"):
                self._handle_api("GET", path, parse_qs(parsed.query))
                return
            self._serve_static(path)
        except ApiError as exc:
            self._send_error_json(exc.status, str(exc))
        except Exception as exc:  # pragma: no cover - 防御性兜底
            traceback.print_exc()
            self._send_error_json(500, f"服务器错误：{exc}")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if not parsed.path.startswith("/api/"):
                raise ApiError("未找到接口", 404)
            payload = self._read_json()
            self._handle_api("POST", parsed.path, payload)
        except ApiError as exc:
            self._send_error_json(exc.status, str(exc))
        except Exception as exc:  # pragma: no cover
            traceback.print_exc()
            self._send_error_json(500, f"服务器错误：{exc}")

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            self._handle_api("DELETE", parsed.path, {})
        except ApiError as exc:
            self._send_error_json(exc.status, str(exc))
        except Exception as exc:  # pragma: no cover
            traceback.print_exc()
            self._send_error_json(500, f"服务器错误：{exc}")

    def _handle_api(self, method: str, path: str, payload: Any) -> None:
        state = self.state

        if method == "GET" and path == "/api/status":
            self._send_json(state.status())
            return
        if method == "GET" and path == "/api/documents":
            self._send_json(state.list_documents())
            return
        if method == "GET" and path == "/api/memory":
            self._send_json(state.memory_snapshot())
            return
        if method == "GET" and path == "/api/runs":
            self._send_json({"runs": state.list_runs()})
            return
        if method == "GET" and path == "/api/eval/cases":
            self._send_json({"cases": state.eval_cases()})
            return
        if method == "GET" and path == "/api/tests":
            self._send_json({"tests": state.list_tests()})
            return

        if method == "POST" and path == "/api/ingest":
            if not isinstance(payload, dict):
                raise ApiError("请求体必须是 JSON 对象")
            if payload.get("url"):
                self._send_json(
                    state.ingest_url(
                        str(payload["url"]), payload.get("title")
                    )
                )
            else:
                self._send_json(
                    state.ingest_upload(
                        str(payload.get("filename") or ""),
                        str(payload.get("content_base64") or ""),
                        payload.get("title"),
                    )
                )
            return
        if method == "POST" and path == "/api/search":
            self._send_json(
                state.search(
                    str(payload.get("query") or ""),
                    payload.get("top_k", 5),
                    str(payload.get("mode") or "hybrid"),
                )
            )
            return
        if method == "POST" and path == "/api/answer":
            self._send_json(
                state.answer(
                    str(payload.get("question") or ""),
                    payload.get("top_k", 5),
                    bool(payload.get("use_llm", True)),
                )
            )
            return
        if method == "POST" and path == "/api/documents/clear":
            self._send_json(state.clear_documents())
            return
        if method == "POST" and path == "/api/memory/add":
            self._send_json(
                state.memory_add(
                    str(payload.get("content") or ""),
                    memory_type=str(payload.get("memory_type") or "semantic"),
                    importance=payload.get("importance", 0.5),
                    metadata=payload.get("metadata") or {},
                )
            )
            return
        if method == "POST" and path == "/api/memory/search":
            self._send_json(
                state.memory_search(
                    str(payload.get("query") or ""), payload.get("top_k", 5)
                )
            )
            return
        if method == "POST" and path == "/api/memory/context":
            self._send_json(
                state.memory_context(
                    str(payload.get("query") or ""),
                    payload.get("max_tokens", 1500),
                )
            )
            return
        if method == "POST" and path == "/api/memory/turn":
            self._send_json(
                state.memory_turn(
                    str(payload.get("question") or ""),
                    str(payload.get("answer") or ""),
                    payload.get("importance", 0.5),
                )
            )
            return
        if method == "POST" and path == "/api/memory/clear":
            self._send_json(state.memory_clear())
            return
        if method == "POST" and path == "/api/paradigms/run":
            self._send_json(state.run_paradigms())
            return
        if method == "POST" and path == "/api/demo/run":
            self._send_json(
                state.run_demo_job(
                    ingest=payload.get("ingest"),
                    query=payload.get("query"),
                    embed_backend=str(payload.get("embed_backend") or "hashing"),
                )
            )
            return
        if method == "POST" and path == "/api/eval/run":
            self._send_json(
                state.run_eval_job(
                    str(payload.get("backend") or "all"),
                    bool(payload.get("online", False)),
                )
            )
            return
        if method == "POST" and path == "/api/tests/run":
            self._send_json(state.run_tests(str(payload.get("target") or "all")))
            return

        if method == "DELETE" and path.startswith("/api/documents/"):
            doc_id = unquote(path[len("/api/documents/"):])
            self._send_json(state.delete_document(doc_id))
            return

        if method == "GET":
            match = re.fullmatch(r"/api/runs/([A-Za-z0-9_\-]+)/([A-Za-z0-9_.\-]+)", path)
            if match:
                self._serve_run_file(match.group(1), match.group(2), payload)
                return

        raise ApiError("未找到接口", 404)

    # --- 静态文件与产物 ---------------------------------------------------

    def _serve_static(self, path: str) -> None:
        files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        entry = files.get(path)
        if entry is None:
            raise ApiError("未找到页面", 404)
        name, content_type = entry
        target = self.server.webui_dir / name  # type: ignore[attr-defined]
        if not target.is_file():
            raise ApiError(f"静态文件缺失：{name}", 500)
        self._send_bytes(target.read_bytes(), content_type)

    def _serve_run_file(self, run_id: str, name: str, query: Any) -> None:
        if name not in ALLOWED_RUN_FILES:
            raise ApiError("不允许访问该文件", 404)
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", run_id):
            raise ApiError("非法的运行 ID", 400)
        base = self.state.runs_dir / run_id
        target = base / name
        if not target.is_file():
            raise ApiError("产物不存在", 404)
        content_type = (
            "text/markdown; charset=utf-8"
            if name.endswith(".md")
            else "application/json; charset=utf-8"
            if name.endswith(".json")
            else "application/x-ndjson; charset=utf-8"
        )
        headers = {}
        if isinstance(query, dict) and (query.get("download") or [""])[0]:
            headers["Content-Disposition"] = f'attachment; filename="{name}"'
        self._send_bytes(target.read_bytes(), content_type, headers=headers)


def make_server(
    state: WebState,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ConsoleServer:
    """创建（但不启动）控制台服务。"""
    return ConsoleServer((host, port), ConsoleHandler, state=state)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="AgentFlow 本地 Web 控制台")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument(
        "--data-dir", default=str(DEFAULT_DATA_DIR), help="数据目录（索引 / 记忆 / 运行产物）"
    )
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    parser.add_argument("--no-seed", action="store_true", help="不写入内置知识库与演示记忆")
    args = parser.parse_args(argv)

    state = WebState(args.data_dir, seed=not args.no_seed)
    server = make_server(state, args.host, args.port)
    host, port = server.server_address[0], server.server_address[1]
    url_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{url_host}:{port}"
    print(f"AgentFlow Web 控制台已启动：{url}")
    print(f"数据目录：{state.data_dir}")
    print("按 Ctrl+C 停止服务。")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()
        state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
