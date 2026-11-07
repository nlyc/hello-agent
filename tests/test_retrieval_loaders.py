"""文档加载器测试：本地 TXT/Markdown、PDF/DOCX（可选依赖）、URL 与 SSRF 防护。"""

from __future__ import annotations

import functools
import http.server
import threading
from pathlib import Path

import pytest

from agentflow.core.exceptions import DocumentLoadError
from agentflow.retrieval import Chunker, load_document
from agentflow.retrieval import loaders


def _write(path: Path, content: str | bytes, encoding: str = "utf-8") -> Path:
    """写入测试文件并返回路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding=encoding)
    return path


def test_load_plain_text(tmp_path: Path):
    path = _write(tmp_path / "guide.txt", "故宫每天八点三十分开放。")
    document = load_document(path)
    assert document.text == "故宫每天八点三十分开放。"
    assert document.source_type == "file"
    assert document.metadata["doc_type"] == "txt"
    assert document.metadata["markdown"] is False
    assert len(document.doc_id) == 16
    assert len(document.content_hash) == 64


def test_load_gb18030_text(tmp_path: Path):
    path = _write(
        tmp_path / "legacy.txt", "北京烤鸭推荐全聚德。".encode("gb18030"), encoding="utf-8"
    )
    document = load_document(path)
    assert "北京烤鸭" in document.text


def test_load_markdown_uses_first_h1_as_title(tmp_path: Path):
    path = _write(
        tmp_path / "beijing.md",
        "# 北京旅游指南\n\n故宫每天八点三十分开放。\n",
    )
    document = load_document(path)
    assert document.title == "北京旅游指南"
    assert document.metadata["markdown"] is True
    assert document.metadata["doc_type"] == "markdown"


def test_doc_format_is_rejected(tmp_path: Path):
    path = _write(tmp_path / "old.doc", b"legacy word")
    with pytest.raises(DocumentLoadError, match="\\.doc"):
        load_document(path)


def test_unknown_extension_is_rejected(tmp_path: Path):
    path = _write(tmp_path / "data.xyz", "hello")
    with pytest.raises(DocumentLoadError, match="不支持的文件类型"):
        load_document(path)


def test_missing_file_is_rejected(tmp_path: Path):
    with pytest.raises(DocumentLoadError, match="文件不存在"):
        load_document(tmp_path / "not-found.txt")


def test_file_size_limit(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(loaders, "MAX_FILE_BYTES", 4)
    path = _write(tmp_path / "big.txt", "0123456789")
    with pytest.raises(DocumentLoadError, match="大小上限"):
        load_document(path)


def test_empty_document_is_rejected(tmp_path: Path):
    path = _write(tmp_path / "empty.txt", "   \n\n")
    with pytest.raises(DocumentLoadError, match="为空"):
        load_document(path)


def test_docx_headings_and_chunk_path(tmp_path: Path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "guide.docx"
    document = docx.Document()
    document.add_heading("北京旅游", level=1)
    document.add_paragraph("故宫每天八点三十分开放。")
    document.add_heading("美食", level=2)
    document.add_paragraph("北京烤鸭推荐全聚德。")
    document.save(str(path))

    loaded = load_document(path)
    assert loaded.metadata["doc_type"] == "docx"
    assert loaded.metadata["markdown"] is True
    assert "# 北京旅游" in loaded.text
    assert loaded.title == "北京旅游"

    chunks = Chunker().split(loaded)
    assert [chunk.heading_path for chunk in chunks] == [
        "北京旅游",
        "北京旅游 / 美食",
    ]


def test_scanned_pdf_needs_ocr(tmp_path: Path):
    pypdf = pytest.importorskip("pypdf")
    path = tmp_path / "scanned.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as handle:
        writer.write(handle)

    with pytest.raises(DocumentLoadError) as excinfo:
        load_document(path)
    assert excinfo.value.metadata.get("needs_ocr") is True
    assert "扫描版" in str(excinfo.value)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """测试用静态文件服务器，屏蔽访问日志。"""

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        """不输出日志。"""


@pytest.fixture()
def http_server(tmp_path: Path):
    """启动本地静态文件服务器，返回根目录与基础 URL。"""
    _write(
        tmp_path / "index.html",
        "<html><head><title>北京旅游</title></head><body>"
        "<h1>北京旅游</h1><p>故宫每天八点三十分开放。</p></body></html>",
    )
    _write(tmp_path / "guide.txt", "北京烤鸭推荐全聚德。")
    _write(tmp_path / "data.bin", b"\x00\x01\x02")

    handler = functools.partial(_QuietHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_load_html_url(http_server: str):
    document = load_document(
        f"{http_server}/index.html", allow_private_hosts=True
    )
    assert document.source_type == "url"
    assert document.title == "北京旅游"
    assert "故宫每天八点三十分开放" in document.text
    assert document.metadata["source_type"] == "url"


def test_load_text_url(http_server: str):
    document = load_document(
        f"{http_server}/guide.txt", allow_private_hosts=True
    )
    assert "北京烤鸭" in document.text
    assert document.source_type == "url"


def test_url_rejects_private_host_by_default(http_server: str):
    with pytest.raises(DocumentLoadError) as excinfo:
        load_document(f"{http_server}/index.html")
    assert excinfo.value.metadata.get("ssrf_blocked") is True


def test_url_rejects_file_scheme():
    with pytest.raises(DocumentLoadError, match="http/https"):
        load_document("file:///C:/Windows/win.ini")


def test_url_rejects_localhost():
    with pytest.raises(DocumentLoadError, match="本机地址"):
        load_document("http://localhost:8000/index.html")


def test_url_rejects_unsupported_content_type(http_server: str):
    with pytest.raises(DocumentLoadError, match="Content-Type"):
        load_document(f"{http_server}/data.bin", allow_private_hosts=True)
