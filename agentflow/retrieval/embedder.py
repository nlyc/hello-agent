"""Embedder：文本向量化协议与两种离线/可选实现。

默认 ``HashingEmbedder`` 为纯标准库特征哈希，零下载、跨进程确定；
``FastEmbedEmbedder`` 懒加载可选依赖，缺失时给出清晰安装提示。
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Protocol, Sequence

from agentflow.core.exceptions import RAGError

_CJK_RE = re.compile(r"[一-鿿]+")
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    """检索用分词：CJK 字 bigram + 英文/数字小写词。

    CJK 按连续片段切分，在片段内生成二元组；片段只有一个字时保留单字。
    英文、数字和下划线按词切分并转小写。该分词方式同时供 BM25 与
    HashingEmbedder 使用，保证稀疏与稠密召回的词汇口径一致。
    """
    if not text:
        return []
    tokens: list[str] = []
    for match in _CJK_RE.finditer(text):
        run = match.group(0)
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    tokens.extend(word.lower() for word in _WORD_RE.findall(text))
    return tokens


class Embedder(Protocol):
    """文本向量化协议。"""

    dimension: int
    fingerprint: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """批量向量化。"""
        ...

    def embed_query(self, text: str) -> list[float]:
        """向量化单条查询。"""
        ...


class HashingEmbedder:
    """基于特征哈希的确定性 Embedder（默认实现）。

    使用 BLAKE2b 对 token 做稳定哈希，带符号累加到固定维度后做 L2 归一化。
    同一文本在任意进程、任意机器上都得到同一向量，适合离线测试和
    无模型环境下的最小闭环。
    """

    def __init__(self, dimension: int = 512) -> None:
        if dimension <= 0:
            raise RAGError("HashingEmbedder 的 dimension 必须为正整数")
        self.dimension = int(dimension)
        self.fingerprint = f"hashing:char-ngram:{self.dimension}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """批量向量化文本。"""
        return [self._embed_one(str(text)) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """向量化查询文本。"""
        return self._embed_one(str(text))

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimension
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[index] += sign
        return _l2_normalize(vector)


class FastEmbedEmbedder:
    """FastEmbed 语义向量实现（可选依赖，懒加载）。

    默认模型 ``BAAI/bge-small-zh-v1.5``，输出 512 维。首次使用会下载模型，
    缓存目录可用 ``cache_dir`` 或环境变量 ``FASTEMBED_CACHE_PATH`` 指定。
    未安装 ``fastembed`` 时抛 ``RAGError``，不做静默降级。
    """

    def __init__(
        self,
        model: str = "BAAI/bge-small-zh-v1.5",
        cache_dir: str | Path | None = None,
        dimension: int = 512,
    ) -> None:
        if dimension <= 0:
            raise RAGError("FastEmbedEmbedder 的 dimension 必须为正整数")
        self.model = model
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.dimension = int(dimension)
        self.fingerprint = f"fastembed:{model}:{self.dimension}"
        self._model = None

    def _ensure_model(self):
        """按需导入并初始化 FastEmbed 模型。"""
        if self._model is not None:
            return self._model
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise RAGError(
                "未安装 fastembed，语义模式不可用。请执行："
                "py -3.14 -m pip install 'fastembed>=0.9,<1.0'"
            ) from exc
        kwargs = {"model_name": self.model}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        self._model = TextEmbedding(**kwargs)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """批量向量化文本，首次调用会加载模型。"""
        model = self._ensure_model()
        vectors = [list(map(float, item)) for item in model.embed(list(texts))]
        if vectors:
            self.dimension = len(vectors[0])
            self.fingerprint = f"fastembed:{self.model}:{self.dimension}"
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """向量化查询文本。"""
        vectors = self.embed([text])
        return vectors[0] if vectors else []


def _l2_normalize(vector: Sequence[float]) -> list[float]:
    """对向量做 L2 归一化；全零向量原样返回。"""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return [float(value) for value in vector]
    return [float(value) / norm for value in vector]
