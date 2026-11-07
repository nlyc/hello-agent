"""标题感知分块器：Markdown / DOCX 标题路径、PDF 页码与 token 滑窗。

默认 ``chunk_tokens=400``、``overlap=60``（CJK 每字约 1 token），
单块硬上限 ``max_chunk_tokens=512``。分块结果带稳定的 ``content_hash``
与 ``chunk_id``，便于去重和增量更新。
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

from agentflow.core.exceptions import DocumentLoadError, RAGError
from agentflow.memory.message_history import estimate_tokens
from agentflow.retrieval.types import Chunk, Document

_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
# 一个“token 单元”：单个 CJK 字符（附带尾随空白）或一段非空白非 CJK 文本。
_TOKEN_UNIT_RE = re.compile(r"[一-鿿]\s*|[^\s一-鿿]+\s*")

DEFAULT_CHUNK_TOKENS = 400
DEFAULT_OVERLAP = 60
MAX_CHUNK_TOKENS = 512
MAX_DOCUMENT_CHUNKS = 5000


class Chunker:
    """把 :class:`Document` 切分为带标题路径与页码的 :class:`Chunk`。

    参数：
        chunk_tokens: 目标块大小（token 估算值）。
        overlap: 相邻块重叠 token 数。
        max_chunk_tokens: 单块硬上限，默认 512。
        max_chunks: 单文档最大块数，超出抛 ``DocumentLoadError``。
    """

    def __init__(
        self,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        overlap: int = DEFAULT_OVERLAP,
        max_chunk_tokens: int = MAX_CHUNK_TOKENS,
        max_chunks: int = MAX_DOCUMENT_CHUNKS,
    ) -> None:
        if chunk_tokens <= 0:
            raise RAGError("chunk_tokens 必须为正整数")
        if overlap < 0:
            raise RAGError("overlap 不能为负数")
        if max_chunk_tokens <= 0:
            raise RAGError("max_chunk_tokens 必须为正整数")
        if max_chunks <= 0:
            raise RAGError("max_chunks 必须为正整数")
        budget = min(int(chunk_tokens), int(max_chunk_tokens))
        if overlap >= budget:
            raise RAGError("overlap 必须小于 chunk_tokens / max_chunk_tokens")
        self.chunk_tokens = int(chunk_tokens)
        self.overlap = int(overlap)
        self.max_chunk_tokens = int(max_chunk_tokens)
        self.max_chunks = int(max_chunks)

    def split(self, document: Document) -> list[Chunk]:
        """把文档切成有序 chunk；空文档返回空列表。"""
        if not document.text or not document.text.strip():
            return []
        chunks: list[Chunk] = []
        markdown = bool(document.metadata.get("markdown", True))
        for page, page_text in self._split_pages(document):
            for heading_path, section in self._split_sections(page_text, markdown):
                pieces = self._window(section)
                if not pieces:
                    continue
                for text in pieces:
                    if len(chunks) >= self.max_chunks:
                        raise DocumentLoadError(
                            f"单文档 chunk 数超过上限 {self.max_chunks}，"
                            "请拆分文档或调大 chunk_tokens"
                        )
                    chunks.append(
                        self._make_chunk(
                            document,
                            text=text,
                            heading_path=heading_path,
                            page=page,
                            index=len(chunks),
                        )
                    )
        return chunks

    def _split_pages(self, document: Document) -> list[tuple[int | None, str]]:
        """按 PDF 页边界拆分；非 PDF 视为单页。"""
        has_pages = document.source_type == "pdf" or "\f" in document.text
        if not has_pages:
            return [(None, document.text)]
        pages = document.text.split("\f")
        return [(index + 1, page) for index, page in enumerate(pages)]

    def _split_sections(self, text: str, markdown: bool) -> list[tuple[str, str]]:
        """按 Markdown 标题层级切段，返回 ``(heading_path, section_text)``。"""
        if not markdown:
            return [("", text)]
        sections: list[tuple[str, str]] = []
        stack: list[tuple[int, str]] = []
        buffer: list[str] = []
        in_fence = False

        def flush() -> None:
            body = "\n".join(buffer).strip()
            buffer.clear()
            if body:
                sections.append((" / ".join(title for _, title in stack), body))

        for line in text.splitlines():
            if _FENCE_RE.match(line):
                in_fence = not in_fence
                buffer.append(line)
                continue
            match = None if in_fence else _MARKDOWN_HEADING_RE.match(line)
            if match:
                flush()
                level = len(match.group(1))
                title = match.group(2).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
            else:
                buffer.append(line)
        flush()
        return sections

    def _window(self, text: str) -> list[str]:
        """在 token 单元上滑动窗口，保留相邻块 overlap。"""
        units: Sequence[str] = _TOKEN_UNIT_RE.findall(text)
        if not units:
            return []
        budget = min(self.chunk_tokens, self.max_chunk_tokens)
        overlap = min(self.overlap, budget - 1)
        pieces: list[str] = []
        start = 0
        total = len(units)
        while start < total:
            end = min(start + budget, total)
            piece = "".join(units[start:end]).strip()
            if piece:
                if estimate_tokens(piece) > self.max_chunk_tokens:
                    # 理论上不会触发；兜底确保硬上限。
                    piece = _truncate_to_tokens(piece, self.max_chunk_tokens)
                pieces.append(piece)
            if end >= total:
                break
            next_start = end - overlap
            if next_start <= start:
                next_start = end
            start = next_start
        return pieces

    def _make_chunk(
        self,
        document: Document,
        *,
        text: str,
        heading_path: str,
        page: int | None,
        index: int,
    ) -> Chunk:
        """构造带稳定 id / hash 的 chunk。"""
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        chunk_id = "chunk_" + hashlib.sha256(
            f"{document.doc_id}|{page}|{heading_path}|{content_hash}|{index}".encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        metadata = {
            key: value
            for key, value in document.metadata.items()
            if key not in {"pages", "markdown"}
        }
        metadata.setdefault("source", document.title or document.source_uri)
        metadata.setdefault("source_uri", document.source_uri)
        metadata.setdefault("doc_type", document.source_type)
        metadata["title"] = document.title
        metadata["chunk_index"] = index
        return Chunk(
            chunk_id=chunk_id,
            doc_id=document.doc_id,
            namespace=document.namespace,
            text=text,
            heading_path=heading_path,
            page=page,
            token_count=estimate_tokens(text),
            content_hash=content_hash,
            metadata=metadata,
        )


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """按 token 估算截断文本（二分查找字符边界）。"""
    if estimate_tokens(text) <= max_tokens:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip()
