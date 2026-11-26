"""retrieval —— 检索层：加载/分块、Embedder、BM25、SQLite/Qdrant 向量库、混合检索与 RAG 管线。"""

from agentflow.retrieval.bm25 import BM25Index
from agentflow.retrieval.chunker import Chunker
from agentflow.retrieval.embedder import Embedder, FastEmbedEmbedder, HashingEmbedder, tokenize
from agentflow.retrieval.entities import PLACE_NAMES, detect_place_candidates, detect_places
from agentflow.retrieval.loaders import load_document
from agentflow.retrieval.pipeline import RAGPipeline
from agentflow.retrieval.retriever import HybridRetriever
from agentflow.retrieval.store_qdrant import QdrantVectorStore
from agentflow.retrieval.store_sqlite import SQLiteVectorStore, pack_vector, unpack_vector
from agentflow.retrieval.types import (
    Chunk,
    Document,
    IngestResult,
    RAGAnswer,
    RetrievalHit,
    VectorStore,
    citation_for_chunk,
    matches_filters,
)

__all__ = [
    "BM25Index",
    "Chunker",
    "Chunk",
    "Document",
    "Embedder",
    "FastEmbedEmbedder",
    "HashingEmbedder",
    "HybridRetriever",
    "IngestResult",
    "PLACE_NAMES",
    "QdrantVectorStore",
    "RAGAnswer",
    "RAGPipeline",
    "RetrievalHit",
    "SQLiteVectorStore",
    "VectorStore",
    "citation_for_chunk",
    "detect_place_candidates",
    "detect_places",
    "load_document",
    "matches_filters",
    "pack_vector",
    "tokenize",
    "unpack_vector",
]
