"""
RAG 知识库演示 - 核心模块

导出主要组件供外部使用
"""

from .config import config, Config
from .embedder import embedder, Embedder
from .vector_store import (
    vector_store,
    VectorStore,
    DimensionMismatchError,
    IndexNotFoundError
)
from .document_manager import (
    document_manager,
    DocumentManager,
    Document,
    SearchResult,
    EmptyDocumentError
)

__all__ = [
    # Config
    "config",
    "Config",
    # Embedder
    "embedder",
    "Embedder",
    # VectorStore
    "vector_store",
    "VectorStore",
    "DimensionMismatchError",
    "IndexNotFoundError",
    # DocumentManager
    "document_manager",
    "DocumentManager",
    "Document",
    "SearchResult",
    "EmptyDocumentError"
]

__version__ = "1.0.0"
