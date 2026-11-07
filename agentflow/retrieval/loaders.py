"""文档加载器：本地 TXT / Markdown / PDF / DOCX 与在线 URL。

设计目标：
- 默认零额外依赖：TXT、Markdown、HTML 仅用标准库；
- PDF / DOCX 按需导入 ``pypdf`` / ``python-docx``，缺失时给出安装提示；
- URL 入库做 SSRF 防护：只允许 http/https、拒绝内网/回环/链路本地地址、
  限制大小与 Content-Type，并对每一次重定向重新校验。
"""

from __future__ import annotations

import hashlib
import io
import ipaddress
import os
import re
import socket
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from agentflow.core.exceptions import DocumentLoadError
from agentflow.retrieval.types import Document

# 资源上限：防止超大文件、超长 PDF、zip 炸弹拖垮进程。
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_URL_BYTES = 20 * 1024 * 1024
MAX_TEXT_BYTES = 5 * 1024 * 1024
MAX_PDF_PAGES = 300
MAX_DOCX_ENTRIES = 10_000
MAX_DOCX_UNCOMPRESSED = 200 * 1024 * 1024
MAX_DOCX_RATIO = 200
URL_TIMEOUT = 10.0

_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_MARKDOWN_TITLE_RE = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)
_HEADING_STYLE_RE = re.compile(r"^(?:heading|标题)\s*([1-6])$", re.IGNORECASE)

ALLOWED_CONTENT_TYPES = {
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "text/html",
    "application/xhtml+xml",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}
_SKIP_TAGS = {"script", "style", "noscript", "template"}


@dataclass
class _Loaded:
    """加载器内部结果：正文、标题、MIME 与元数据。"""

    text: str
    title: str = ""
    mime: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def load_document(
    source: str | Path,
    *,
    namespace: str = "travel",
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
    allow_private_hosts: bool = False,
) -> Document:
    """加载本地文件或在线 URL，返回统一的 :class:`Document`。

    参数：
        source: 本地路径或 ``http(s)://`` 地址。
        namespace: 文档所属命名空间，默认 ``travel``。
        title: 显式标题；为空时尝试从文档内容推断。
        metadata: 调用方附加元数据，会与加载器元数据合并。
        allow_private_hosts: 仅测试/内网部署显式开启，默认拒绝内网地址。

    异常：
        DocumentLoadError: 格式不支持、解析为空、来源不安全或超限。
    """
    source_text = str(source)
    if _URL_RE.match(source_text):
        return _load_url_document(
            source_text,
            namespace=namespace,
            title=title,
            metadata=metadata,
            allow_private_hosts=allow_private_hosts,
        )
    return _load_file_document(
        Path(source_text), namespace=namespace, title=title, metadata=metadata
    )


def _load_file_document(
    path: Path,
    *,
    namespace: str,
    title: str | None,
    metadata: dict[str, Any] | None,
) -> Document:
    """加载本地文件并按扩展名分派。"""
    source_uri = _normalize_local_path(path)
    if not path.exists():
        raise DocumentLoadError(f"文件不存在：{path}")
    if not path.is_file():
        raise DocumentLoadError(f"不是普通文件：{path}")

    suffix = path.suffix.lower()
    if suffix == ".doc":
        raise DocumentLoadError(
            "暂不支持旧版 .doc 格式，请另存为 .docx 后重试",
            metadata={"source_uri": source_uri, "doc_type": "doc"},
        )
    if suffix not in {".txt", ".md", ".markdown", ".pdf", ".docx"}:
        raise DocumentLoadError(
            f"不支持的文件类型：{suffix or '无扩展名'}（支持 .txt/.md/.pdf/.docx）",
            metadata={"source_uri": source_uri},
        )

    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise DocumentLoadError(
            f"文件超过大小上限 {MAX_FILE_BYTES // (1024 * 1024)}MB：{size} 字节",
            metadata={"source_uri": source_uri},
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DocumentLoadError(f"读取文件失败：{exc}") from exc

    if suffix == ".pdf":
        loaded = _parse_pdf(data, source_label=source_uri)
    elif suffix == ".docx":
        loaded = _parse_docx(data, source_label=source_uri)
    else:
        text = _decode_bytes(data, charset=None)
        if suffix in {".md", ".markdown"}:
            loaded = _Loaded(
                text=text,
                title=_first_markdown_title(text),
                mime="text/markdown",
                metadata={"doc_type": "markdown", "markdown": True},
            )
        else:
            loaded = _Loaded(
                text=text,
                title="",
                mime="text/plain",
                metadata={"doc_type": "txt", "markdown": False},
            )

    return _build_document(
        loaded,
        namespace=namespace,
        source_type="file",
        source_uri=source_uri,
        title=title,
        metadata=metadata,
        fallback_title=path.stem,
    )


def _load_url_document(
    url: str,
    *,
    namespace: str,
    title: str | None,
    metadata: dict[str, Any] | None,
    allow_private_hosts: bool,
) -> Document:
    """下载并解析在线 URL。"""
    _validate_url(url, allow_private_hosts=allow_private_hosts)
    opener = build_opener(_SafeRedirectHandler(allow_private_hosts=allow_private_hosts))
    request = Request(
        url,
        headers={
            "User-Agent": "agentflow-rag/1.0 (+offline-test)",
            "Accept": "text/html,text/plain,application/pdf,"
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        },
    )
    try:
        with opener.open(request, timeout=URL_TIMEOUT) as response:
            final_url = _normalize_url(response.geturl())
            _validate_url(final_url, allow_private_hosts=allow_private_hosts)
            content_type = response.headers.get_content_type().lower()
            if content_type not in ALLOWED_CONTENT_TYPES:
                raise DocumentLoadError(
                    f"不支持的 Content-Type：{content_type or '未知'}",
                    metadata={"source_uri": final_url},
                )
            length = response.headers.get("Content-Length")
            if length and length.isdigit() and int(length) > MAX_URL_BYTES:
                raise DocumentLoadError(
                    f"远程文件超过大小上限 {MAX_URL_BYTES // (1024 * 1024)}MB",
                    metadata={"source_uri": final_url},
                )
            data = response.read(MAX_URL_BYTES + 1)
            if len(data) > MAX_URL_BYTES:
                raise DocumentLoadError(
                    f"远程文件超过大小上限 {MAX_URL_BYTES // (1024 * 1024)}MB",
                    metadata={"source_uri": final_url},
                )
            charset = response.headers.get_content_charset()
    except DocumentLoadError:
        raise
    except Exception as exc:  # 网络错误统一转成可读异常
        raise DocumentLoadError(
            f"URL 加载失败：{exc}", metadata={"source_uri": url}
        ) from exc

    if content_type == "application/pdf":
        loaded = _parse_pdf(data, source_label=final_url)
    elif content_type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        loaded = _parse_docx(data, source_label=final_url)
    elif content_type in {"text/html", "application/xhtml+xml"}:
        text, html_title = _html_to_text(_decode_bytes(data, charset=charset))
        loaded = _Loaded(
            text=text,
            title=html_title,
            mime=content_type,
            metadata={"doc_type": "html", "markdown": True, "source_type": "url"},
        )
    else:
        text = _decode_bytes(data, charset=charset)
        loaded = _Loaded(
            text=text,
            title="",
            mime=content_type,
            metadata={"doc_type": "text", "markdown": False, "source_type": "url"},
        )

    return _build_document(
        loaded,
        namespace=namespace,
        source_type="url",
        source_uri=final_url,
        title=title,
        metadata=metadata,
        fallback_title=urlsplit(final_url).hostname or final_url,
    )


def _build_document(
    loaded: _Loaded,
    *,
    namespace: str,
    source_type: str,
    source_uri: str,
    title: str | None,
    metadata: dict[str, Any] | None,
    fallback_title: str,
) -> Document:
    """把加载结果整理成 Document，并计算稳定 doc_id 与内容哈希。"""
    text = _normalize_page_text(loaded.text)
    if not text.strip():
        raise DocumentLoadError(
            "文档解析后为空，无法入库",
            metadata={"source_uri": source_uri, "needs_ocr": False},
        )
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise DocumentLoadError(
            f"解析文本超过上限 {MAX_TEXT_BYTES // (1024 * 1024)}MB",
            metadata={"source_uri": source_uri},
        )

    merged_metadata: dict[str, Any] = dict(metadata or {})
    merged_metadata.update(loaded.metadata)
    resolved_title = (title or loaded.title or fallback_title).strip()
    merged_metadata.update(
        {
            "source": resolved_title,
            "source_uri": source_uri,
            "title": resolved_title,
            "doc_type": merged_metadata.get("doc_type", "text"),
            "source_type": source_type,
        }
    )
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    doc_id = hashlib.sha256(
        f"{source_type}|{source_uri}".encode("utf-8")
    ).hexdigest()[:16]
    return Document(
        doc_id=doc_id,
        namespace=namespace,
        source_type=source_type,
        source_uri=source_uri,
        title=resolved_title,
        content_hash=content_hash,
        mime=loaded.mime,
        text=text,
        metadata=merged_metadata,
    )


def _decode_bytes(data: bytes, *, charset: str | None) -> str:
    """按声明编码、UTF-8、gb18030 依次尝试解码。"""
    candidates: list[str] = []
    if charset:
        candidates.append(charset)
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        if encoding not in candidates:
            candidates.append(encoding)
    for encoding in candidates:
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def _parse_pdf(data: bytes, *, source_label: str) -> _Loaded:
    """用 pypdf 提取 PDF 文本，保留页边界（``\\f``）。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise DocumentLoadError(
            "未安装 pypdf，无法解析 PDF。请执行："
            "py -3.14 -m pip install 'pypdf>=5.0'",
            metadata={"source_uri": source_label},
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    raise DocumentLoadError("PDF 已加密，无法读取")
            except DocumentLoadError:
                raise
            except Exception as exc:
                raise DocumentLoadError(f"PDF 解密失败：{exc}") from exc
        page_count = len(reader.pages)
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(f"PDF 解析失败：{exc}") from exc

    if page_count > MAX_PDF_PAGES:
        raise DocumentLoadError(
            f"PDF 页数超过上限 {MAX_PDF_PAGES}：{page_count} 页",
            metadata={"source_uri": source_label, "page_count": page_count},
        )

    pages: list[str] = []
    for index in range(page_count):
        try:
            pages.append(reader.pages[index].extract_text() or "")
        except Exception:
            pages.append("")
    if not any(page.strip() for page in pages):
        raise DocumentLoadError(
            "PDF 未提取到任何文本，可能是扫描版，需要 OCR 后才能入库",
            metadata={"source_uri": source_label, "needs_ocr": True},
        )

    title = ""
    try:
        raw_title = (reader.metadata or {}).get("/Title") if reader.metadata else None
        if raw_title:
            title = str(raw_title).strip()
    except Exception:
        title = ""
    text = "\f".join(_normalize_page_text(page) for page in pages)
    return _Loaded(
        text=text,
        title=title,
        mime="application/pdf",
        metadata={"doc_type": "pdf", "markdown": False, "page_count": page_count},
    )


def _parse_docx(data: bytes, *, source_label: str) -> _Loaded:
    """用 python-docx 提取 DOCX 段落、标题与表格。"""
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise DocumentLoadError(
            "未安装 python-docx，无法解析 Word。请执行："
            "py -3.14 -m pip install 'python-docx>=1.1'",
            metadata={"source_uri": source_label},
        ) from exc
    _check_docx_archive(data, source_label=source_label)

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentLoadError(f"DOCX 解析失败：{exc}") from exc

    lines: list[str] = []
    first_heading = ""
    for block in _iter_docx_blocks(document):
        if _is_docx_paragraph(block):
            text = block.text.strip()
            if not text:
                continue
            level = _docx_heading_level(block)
            if level:
                lines.append("#" * level + " " + text)
                if not first_heading:
                    first_heading = text
            else:
                lines.append(text)
        else:
            for row in block.rows:
                cells = [_normalize_inline(cell.text) for cell in row.cells]
                if any(cells):
                    lines.append(" | ".join(cells))
            if block.rows:
                lines.append("")

    text = "\n".join(lines)
    if not text.strip():
        raise DocumentLoadError(
            "DOCX 未提取到文本内容", metadata={"source_uri": source_label}
        )
    return _Loaded(
        text=text,
        title=first_heading,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        metadata={"doc_type": "docx", "markdown": True},
    )


def _iter_docx_blocks(document: Any) -> Iterable[Any]:
    """按正文顺序产出 Paragraph / Table（python-docx 无公开迭代器）。"""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            yield Paragraph(child, document)
        elif tag == "tbl":
            yield Table(child, document)


def _is_docx_paragraph(block: Any) -> bool:
    """判断块是否为段落。"""
    from docx.text.paragraph import Paragraph

    return isinstance(block, Paragraph)


def _docx_heading_level(paragraph: Any) -> int:
    """从段落样式名推断标题层级，非标题返回 0。"""
    style = getattr(paragraph, "style", None)
    name = getattr(style, "name", "") if style is not None else ""
    if not name:
        return 0
    match = _HEADING_STYLE_RE.match(name.strip())
    if match:
        return int(match.group(1))
    if name.strip().lower() == "title":
        return 1
    return 0


def _check_docx_archive(data: bytes, *, source_label: str) -> None:
    """检查 DOCX zip 结构，防止 zip 炸弹。"""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_DOCX_ENTRIES:
                raise DocumentLoadError(
                    f"DOCX 压缩包条目过多：{len(infos)}",
                    metadata={"source_uri": source_label},
                )
            total = sum(info.file_size for info in infos)
            compressed = max(1, sum(info.compress_size for info in infos))
            if total > MAX_DOCX_UNCOMPRESSED:
                raise DocumentLoadError(
                    "DOCX 解压后体积过大，已拒绝处理",
                    metadata={"source_uri": source_label},
                )
            if total > 10 * 1024 * 1024 and total / compressed > MAX_DOCX_RATIO:
                raise DocumentLoadError(
                    "DOCX 压缩比异常，疑似 zip 炸弹",
                    metadata={"source_uri": source_label},
                )
    except zipfile.BadZipFile as exc:
        raise DocumentLoadError(
            "DOCX 文件损坏或不是有效的 zip 包", metadata={"source_uri": source_label}
        ) from exc


class _HTMLTextExtractor(HTMLParser):
    """把 HTML 转成带 Markdown 标题的纯文本。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._title_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """处理开始标签。"""
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")
        if re.fullmatch(r"h[1-6]", tag):
            self._parts.append("\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag in {"td", "th"}:
            self._parts.append(" | ")
        elif tag == "br":
            self._parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """处理自闭合标签。"""
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        """处理结束标签。"""
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        """收集正文或标题文本。"""
        if self._skip_depth:
            return
        if self._in_title:
            self._title_parts.append(data)
        else:
            self._parts.append(data)

    @property
    def title(self) -> str:
        """返回 ``<title>`` 文本。"""
        return _normalize_inline("".join(self._title_parts))


def _html_to_text(html: str) -> tuple[str, str]:
    """解析 HTML，返回正文与标题。"""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:
        raise DocumentLoadError(f"HTML 解析失败：{exc}") from exc
    return _normalize_page_text("".join(parser._parts)), parser.title


def _validate_url(url: str, *, allow_private_hosts: bool = False) -> None:
    """校验 URL 协议与主机，阻止 SSRF。"""
    parts = urlsplit(url)
    if parts.scheme.lower() not in {"http", "https"}:
        raise DocumentLoadError(
            f"只支持 http/https URL，拒绝：{parts.scheme or '未知协议'}",
            metadata={"source_uri": url},
        )
    if parts.username or parts.password:
        raise DocumentLoadError("URL 不允许包含用户名或密码", metadata={"source_uri": url})
    host = parts.hostname
    if not host:
        raise DocumentLoadError("URL 缺少主机名", metadata={"source_uri": url})
    if allow_private_hosts:
        return
    if host.lower() in {"localhost", "localhost.localdomain"} or host.lower().endswith(
        ".localhost"
    ):
        raise DocumentLoadError(
            "不允许访问本机地址", metadata={"source_uri": url, "ssrf_blocked": True}
        )
    try:
        port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise DocumentLoadError(f"URL 端口非法：{exc}") from exc
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise DocumentLoadError(
            f"域名解析失败：{host}", metadata={"source_uri": url}
        ) from exc
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address.split("%", 1)[0])
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise DocumentLoadError(
                f"不允许访问内网或保留地址：{host} -> {address}",
                metadata={"source_uri": url, "ssrf_blocked": True},
            )


class _SafeRedirectHandler(HTTPRedirectHandler):
    """对每一次重定向目标重新做 URL 安全校验。"""

    def __init__(self, *, allow_private_hosts: bool = False) -> None:
        super().__init__()
        self._allow_private_hosts = allow_private_hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        """校验新地址后交给父类处理。"""
        _validate_url(newurl, allow_private_hosts=self._allow_private_hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _normalize_url(url: str) -> str:
    """规范化 URL：小写协议/主机、去掉 fragment 与默认端口。"""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = host
    if port and not (
        (parts.scheme.lower() == "http" and port == 80)
        or (parts.scheme.lower() == "https" and port == 443)
    ):
        netloc = f"{host}:{port}"
    path = parts.path or "/"
    return urlunsplit((parts.scheme.lower(), netloc, path, parts.query, ""))


def _normalize_local_path(path: Path) -> str:
    """本地路径规范化（大小写与分隔符统一），用于稳定 doc_id。"""
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path
    normalized = os.path.normcase(str(resolved))
    return Path(normalized).as_posix()


def _first_markdown_title(text: str) -> str:
    """取 Markdown 第一个一级标题作为文章标题。"""
    match = _MARKDOWN_TITLE_RE.search(text)
    return match.group(1).strip() if match else ""


def _normalize_inline(text: str) -> str:
    """压缩行内空白。"""
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_page_text(text: str) -> str:
    """规范化换行与空白，但保留 PDF 的 ``\\f`` 页分隔。"""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
