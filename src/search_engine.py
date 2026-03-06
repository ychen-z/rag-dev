"""
搜索引擎模块 - 封装语义搜索功能
"""
from typing import List, Dict, Any, Optional
from functools import lru_cache
import time

from .embedder import Embedder, embedder as default_embedder
from .vector_store import VectorStore, vector_store as default_vector_store
from .document_manager import DocumentManager, document_manager as default_document_manager


class SearchEngine:
    """语义搜索引擎，集成嵌入、向量存储和文档管理"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        document_manager: Optional[DocumentManager] = None
    ):
        if self._initialized:
            return
        
        # 使用全局单例或传入的实例
        self._embedder = embedder or default_embedder
        self._vector_store = vector_store or default_vector_store
        self._document_manager = document_manager or default_document_manager
        self._query_cache = {}
        self._cache_max_size = 100
        self._initialized = True
    
    def search(
        self,
        query_text: str,
        k: int = 3,
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        执行语义搜索
        
        Args:
            query_text: 查询文本
            k: 返回结果数量
            threshold: 相似度阈值 (0-1)，低于此值的结果被过滤
            filters: 元数据过滤条件，例如 {"category": "技术"}
            
        Returns:
            包含搜索结果和元数据的字典
        """
        start_time = time.time()
        
        if not query_text or not query_text.strip():
            return {
                "results": [],
                "query": query_text,
                "elapsed_ms": 0,
                "total_found": 0
            }
        
        # 生成查询向量（使用缓存）
        query_vector = self._get_cached_embedding(query_text.strip())
        
        # 执行向量搜索
        search_results = self._vector_store.search(query_vector, top_k=k * 2 if filters else k)
        
        # 处理结果
        results = []
        for vec_id, doc_id, distance in search_results:
            # 转换 L2 距离为相似度分数 (0-1)
            similarity = self._distance_to_similarity(distance)
            
            # 应用相似度阈值
            if threshold is not None and similarity < threshold:
                continue
            
            # 获取文档
            doc = self._document_manager.get_document(doc_id)
            
            if doc is None:
                continue
            
            # 应用元数据过滤
            if filters and not self._match_filters(doc.metadata or {}, filters):
                continue
            
            # 获取匹配的分块
            chunk_idx = vec_id % len(doc.chunks) if doc.chunks else 0
            chunk_text = doc.chunks[chunk_idx] if chunk_idx < len(doc.chunks) else doc.content[:200]
            
            results.append({
                "doc_id": doc_id,
                "chunk_id": chunk_idx,
                "text": chunk_text,
                "score": round(similarity, 4),
                "distance": round(distance, 4),
                "metadata": doc.metadata or {},
                "rank": len(results) + 1
            })
            
            if len(results) >= k:
                break
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            "results": results,
            "query": query_text,
            "elapsed_ms": round(elapsed_ms, 2),
            "total_found": len(results)
        }
    
    def batch_search(
        self,
        queries: List[str],
        k: int = 3,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        批量搜索多个查询
        
        Args:
            queries: 查询文本列表
            k: 每个查询返回的结果数量
            threshold: 相似度阈值
            
        Returns:
            搜索结果列表
        """
        return [
            self.search(query, k=k, threshold=threshold)
            for query in queries
        ]
    
    def _get_cached_embedding(self, text: str):
        """获取缓存的嵌入向量"""
        if text in self._query_cache:
            return self._query_cache[text]
        
        # 清理缓存（如果超过最大大小）
        if len(self._query_cache) >= self._cache_max_size:
            # 移除最早的一半缓存
            keys_to_remove = list(self._query_cache.keys())[:self._cache_max_size // 2]
            for key in keys_to_remove:
                del self._query_cache[key]
        
        # 生成并缓存
        embedding = self._embedder.embed_text(text)
        self._query_cache[text] = embedding
        return embedding
    
    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        """
        将 L2 距离转换为相似度分数 (0-1)
        使用公式: similarity = 1 / (1 + distance)
        """
        return 1.0 / (1.0 + distance)
    
    @staticmethod
    def _match_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """检查元数据是否匹配过滤条件"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True
    
    def clear_cache(self):
        """清除查询缓存"""
        self._query_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取搜索引擎统计信息"""
        return {
            "cache_size": len(self._query_cache),
            "cache_max_size": self._cache_max_size,
            "vector_store_stats": self._vector_store.get_stats(),
            "document_count": len(self._document_manager.list_documents())
        }
