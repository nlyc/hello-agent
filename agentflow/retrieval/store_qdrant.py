"""Qdrant 向量库适配器（可选依赖，懒加载）。

Qdrant 只是向量数据库，不提供 embedding；向量仍由 ``Embedder`` 生成。
未安装 ``qdrant-client`` 时构造即抛 ``RAGError``，不做静默降级。
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Sequence

from agentflow.core.exceptions import EmbeddingMismatchError, RAGError
from agentflow.retrieval.store_sqlite import unpack_vector
from agentflow.retrieval.types import Chunk, RetrievalHit

class QdrantVectorStore:
    """基于 Qdrant 的远程向量库。

    参数：
        url: Qdrant 服务地址（如 ``https://xxx.cloud.qdrant.io:6333``）。
        api_key: 可选 API Key。
        collection: 集合名，默认 ``agentflow_rag``。
        dimension: 向量维度；为空时在首次 upsert 时自动创建集合。
        timeout: 请求超时秒数。
        prefer_grpc: 是否优先使用 gRPC。
    """

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        collection: str = "agentflow_rag",
        dimension: int | None = None,
        timeout: float = 30.0,
        prefer_grpc: bool = False,
    ) -> None:
        if not url:
            raise RAGError("QdrantVectorStore 需要 url")
        if not collection:
            raise RAGError("QdrantVectorStore 需要 collection 名称")
        self.url = url
        self.api_key = api_key
        self.collection = collection
        self.dimension = int(dimension) if dimension else None
        self.timeout = float(timeout)
        self.prefer_grpc = bool(prefer_grpc)
        self._client: Any | None = None
        self._lock = threading.RLock()
        self._collection_ready = False

    # ------------------------------------------------------------------ 基础
    def _ensure_client(self) -> Any:
        """懒加载 qdrant-client。"""
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise RAGError(
                "未安装 qdrant-client，Qdrant 后端不可用。请执行："
                "py -3.14 -m pip install 'qdrant-client>=1.12'"
            ) from exc
        self._client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=self.timeout,
            prefer_grpc=self.prefer_grpc,
        )
        return self._client

    def _ensure_collection(self, vector_size: int) -> None:
        """确保集合存在且维度一致。"""
        if self._collection_ready:
            if self.dimension and self.dimension != vector_size:
                raise EmbeddingMismatchError(
                    f"Qdrant 集合维度 {self.dimension} 与当前向量维度 {vector_size} 不一致"
                )
            return
        from qdrant_client.models import Distance, VectorParams

        client = self._ensure_client()
        if client.collection_exists(self.collection):
            actual = self._collection_vector_size(client)
            if actual is not None and actual != vector_size:
                raise EmbeddingMismatchError(
                    f"Qdrant 集合 '{self.collection}' 维度 {actual} 与当前向量维度 "
                    f"{vector_size} 不一致；请更换集合或重建索引"
                )
            self.dimension = actual or vector_size
        else:
            client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=vector_size, distance=Distance.COSINE
                ),
            )
            self.dimension = vector_size
        self._collection_ready = True

    def _collection_vector_size(self, client: Any) -> int | None:
        """读取已有集合的向量维度。"""
        try:
            info = client.get_collection(self.collection)
            vectors = info.config.params.vectors
            if isinstance(vectors, dict):
                vectors = next(iter(vectors.values()))
            return int(vectors.size)
        except Exception:
            return None

    # ------------------------------------------------------------------ 写入
    def upsert(self, chunks: Sequence[Chunk]) -> int:
        """写入或更新一批 chunk，返回条数。"""
        prepared: list[tuple[Chunk, list[float]]] = []
        for chunk in chunks:
            if not chunk.chunk_id:
                raise RAGError("Qdrant 向量库要求 chunk_id 非空")
            if not chunk.namespace:
                raise RAGError("Qdrant 向量库要求 namespace 非空")
            if chunk.embedding is None:
                raise RAGError("chunk 缺少 embedding，无法写入 Qdrant")
            vector = (
                unpack_vector(chunk.embedding)
                if isinstance(chunk.embedding, bytes)
                else [float(value) for value in chunk.embedding]
            )
            if not vector:
                raise RAGError("chunk embedding 不能为空向量")
            prepared.append((chunk, vector))
        if not prepared:
            return 0
        sizes = {len(vector) for _, vector in prepared}
        if len(sizes) != 1:
            raise EmbeddingMismatchError("同一批次待写入向量维度不一致")
        vector_size = sizes.pop()

        # 先确认依赖可用，保证缺包时抛 RAGError 而不是 ImportError。
        self._ensure_client()
        from qdrant_client.models import PointStruct

        with self._lock:
            self._ensure_collection(vector_size)
            self._validate_fingerprints(prepared)
            points = [
                PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector=vector,
                    payload=_chunk_payload(chunk),
                )
                for chunk, vector in prepared
            ]
            client = self._ensure_client()
            client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def _validate_fingerprints(
        self, prepared: Sequence[tuple[Chunk, list[float]]]
    ) -> None:
        """校验同一命名空间内的 embedding 指纹。"""
        batch: dict[str, str] = {}
        for chunk, _ in prepared:
            fingerprint = chunk.embedding_fingerprint
            if not fingerprint:
                continue
            previous = batch.get(chunk.namespace)
            if previous is not None and previous != fingerprint:
                raise EmbeddingMismatchError(
                    f"命名空间 '{chunk.namespace}' 同批次 embedding fingerprint 不一致"
                )
            batch[chunk.namespace] = fingerprint
        for namespace, fingerprint in batch.items():
            existing = self.fingerprint(namespace)
            if existing and existing != fingerprint:
                raise EmbeddingMismatchError(
                    f"命名空间 '{namespace}' 已有 fingerprint {existing}，"
                    f"与当前 {fingerprint} 不一致；请重建索引"
                )

    # ------------------------------------------------------------------ 检索
    def search(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        """按余弦相似度检索。"""
        if top_k <= 0:
            return []
        query = [float(value) for value in vector]
        if not query:
            return []
        client = self._ensure_client()
        if not self._collection_ready:
            if not client.collection_exists(self.collection):
                return []
            actual = self._collection_vector_size(client)
            if actual is not None and actual != len(query):
                raise EmbeddingMismatchError(
                    f"查询向量维度 {len(query)} 与 Qdrant 集合维度 {actual} 不一致"
                )
            self.dimension = actual or len(query)
            self._collection_ready = True
        # 与 SQLite 后端一致：过滤器由调用方完整给出，这里不补默认命名空间。
        query_filter = _build_filter(filters)
        with self._lock:
            if hasattr(client, "query_points"):
                response = client.query_points(
                    collection_name=self.collection,
                    query=query,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                )
                points = response.points
            else:  # pragma: no cover - 兼容旧版 qdrant-client
                points = client.search(
                    collection_name=self.collection,
                    query_vector=query,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                )
        hits: list[RetrievalHit] = []
        for point in points:
            payload = dict(point.payload or {})
            metadata = dict(payload.get("metadata") or {})
            chunk = Chunk(
                chunk_id=str(payload.get("chunk_id", point.id)),
                doc_id=str(payload.get("doc_id", "")),
                namespace=str(payload.get("namespace", "")),
                text=str(payload.get("text", "")),
                heading_path=str(payload.get("heading_path", "")),
                page=payload.get("page"),
                token_count=int(payload.get("token_count", 0) or 0),
                content_hash=str(payload.get("content_hash", "")),
                metadata=metadata,
                embedding_fingerprint=str(payload.get("embedding_fingerprint", "")),
            )
            hits.append(RetrievalHit.from_chunk(chunk, float(point.score)))
        return hits

    def content_hashes(
        self, namespace: str | None = None, doc_id: str | None = None
    ) -> set[str]:
        """通过 scroll 读取内容哈希集合。"""
        client = self._ensure_client()
        if not client.collection_exists(self.collection):
            return set()
        filters: dict[str, Any] = {}
        if namespace is not None:
            filters["namespace"] = namespace
        if doc_id is not None:
            filters["doc_id"] = doc_id
        query_filter = _build_filter(filters)
        hashes: set[str] = set()
        offset: Any = None
        while True:
            records, offset = client.scroll(
                collection_name=self.collection,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                value = (record.payload or {}).get("content_hash")
                if value:
                    hashes.add(str(value))
            if offset is None:
                break
        return hashes

    def fingerprint(self, namespace: str | None = None) -> str | None:
        """读取命名空间内第一条记录的 embedding 指纹。"""
        client = self._ensure_client()
        if not client.collection_exists(self.collection):
            return None
        filters = {"namespace": namespace} if namespace else {}
        records, _ = client.scroll(
            collection_name=self.collection,
            scroll_filter=_build_filter(filters),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        value = (records[0].payload or {}).get("embedding_fingerprint")
        return str(value) if value else None

    # ------------------------------------------------------------------ 删除
    def delete(self, doc_id: str | None = None, namespace: str | None = None) -> int:
        """删除指定文档/命名空间；两者都为空时删除整个集合。"""
        client = self._ensure_client()
        if not client.collection_exists(self.collection):
            return 0
        if doc_id is None and namespace is None:
            total = self.count()
            client.delete_collection(self.collection)
            self._collection_ready = False
            return total
        filters: dict[str, Any] = {}
        if doc_id is not None:
            filters["doc_id"] = doc_id
        if namespace is not None:
            filters["namespace"] = namespace
        query_filter = _build_filter(filters)
        total = self.count(namespace=namespace)
        if doc_id is not None:
            total = self._count_filtered(client, query_filter)
        from qdrant_client.models import FilterSelector

        client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(filter=query_filter),
            wait=True,
        )
        return total

    def _count_filtered(self, client: Any, query_filter: Any) -> int:
        """按过滤条件精确计数。"""
        try:
            return int(
                client.count(
                    collection_name=self.collection,
                    count_filter=query_filter,
                    exact=True,
                ).count
            )
        except Exception:
            return 0

    def count(self, namespace: str | None = None) -> int:
        """统计 chunk 数量。"""
        client = self._ensure_client()
        if not client.collection_exists(self.collection):
            return 0
        if namespace is None:
            query_filter = None
        else:
            query_filter = _build_filter({"namespace": namespace})
        try:
            result = client.count(
                collection_name=self.collection,
                count_filter=query_filter,
                exact=True,
            )
            return int(result.count)
        except Exception:
            return 0

    def clear(self, namespace: str | None = None) -> None:
        """清空命名空间；为空时删除整个集合。"""
        self.delete(namespace=namespace)

    def close(self) -> None:
        """关闭客户端（可重复调用）。"""
        if self._client is None:
            return
        try:
            self._client.close()
        finally:
            self._client = None
            self._collection_ready = False

    def __enter__(self) -> "QdrantVectorStore":
        """支持 ``with`` 上下文。"""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """退出上下文时关闭客户端。"""
        self.close()


def _point_id(chunk_id: str) -> str:
    """把任意 chunk_id 映射为 Qdrant 接受的 UUID。"""
    try:
        return str(uuid.UUID(chunk_id))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _chunk_payload(chunk: Chunk) -> dict[str, Any]:
    """构造 Qdrant payload。"""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "namespace": chunk.namespace,
        "text": chunk.text,
        "heading_path": chunk.heading_path,
        "page": chunk.page,
        "token_count": chunk.token_count,
        "content_hash": chunk.content_hash,
        "metadata": dict(chunk.metadata or {}),
        "embedding_fingerprint": chunk.embedding_fingerprint,
    }


def _build_filter(filters: dict[str, Any] | None) -> Any:
    """把通用过滤字典转换为 Qdrant Filter。"""
    if not filters:
        return None
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        MatchAny,
        MatchValue,
    )

    conditions = []
    for key, expected in filters.items():
        field = key if key in {"namespace", "doc_id"} else f"metadata.{key}"
        if isinstance(expected, (list, tuple, set, frozenset)):
            condition = FieldCondition(
                key=field, match=MatchAny(any=list(expected))
            )
        else:
            condition = FieldCondition(key=field, match=MatchValue(value=expected))
        conditions.append(condition)
    return Filter(must=conditions)
