"""
嵌入模块

使用 sentence-transformers 将文本转换为向量表示
"""

import threading
from functools import lru_cache
from typing import List, Optional, Union

import numpy as np

from .config import config


class Embedder:
    """
    文本嵌入器
    
    使用 sentence-transformers 模型将文本转换为稠密向量。
    支持单例模式和模型缓存，避免重复加载。
    """
    
    _instance: Optional["Embedder"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "Embedder":
        """线程安全的单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """初始化嵌入器，加载 sentence-transformers 模型"""
        if self._initialized:
            return
        
        self._model = None
        self._model_name = config.embedding_model
        self._dimension = config.embedding_dimension
        self._cache_size = config.cache_size
        self._initialized = True
    
    @property
    def model(self):
        """
        延迟加载模型
        
        Returns:
            SentenceTransformer 模型实例
        """
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self) -> None:
        """加载 sentence-transformers 模型"""
        try:
            from sentence_transformers import SentenceTransformer
            
            print(f"正在加载嵌入模型: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            print(f"模型加载完成，维度: {self._dimension}")
            
        except ImportError:
            raise ImportError(
                "请安装 sentence-transformers: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
    
    @property
    def dimension(self) -> int:
        """嵌入向量维度"""
        return self._dimension
    
    @property
    def model_name(self) -> str:
        """模型名称"""
        return self._model_name
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        将单个文本转换为嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 形状为 (dimension,) 的向量
        """
        if not text or not text.strip():
            return np.zeros(self._dimension, dtype=np.float32)
        
        # 使用缓存的嵌入
        return self._cached_embed(text.strip())
    
    @lru_cache(maxsize=100)
    def _cached_embed(self, text: str) -> np.ndarray:
        """
        带缓存的单文本嵌入
        
        Args:
            text: 输入文本（已去除首尾空白）
            
        Returns:
            np.ndarray: 嵌入向量
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.astype(np.float32)
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        批量将文本转换为嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条
            
        Returns:
            np.ndarray: 形状为 (len(texts), dimension) 的矩阵
        """
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        
        # 预处理：去除空白，替换空文本
        processed_texts = []
        empty_indices = []
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                processed_texts.append("")
                empty_indices.append(i)
            else:
                processed_texts.append(text.strip())
        
        # 批量编码非空文本
        non_empty_texts = [t for t in processed_texts if t]
        
        if non_empty_texts:
            embeddings = self.model.encode(
                non_empty_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress
            ).astype(np.float32)
        else:
            embeddings = np.zeros((0, self._dimension), dtype=np.float32)
        
        # 构建完整结果矩阵
        result = np.zeros((len(texts), self._dimension), dtype=np.float32)
        
        non_empty_idx = 0
        for i, text in enumerate(processed_texts):
            if text:
                result[i] = embeddings[non_empty_idx]
                non_empty_idx += 1
            # 空文本保持为零向量
        
        return result
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        计算查询向量与文档向量的余弦相似度
        
        Args:
            query_embedding: 查询向量，形状 (dimension,)
            document_embeddings: 文档向量矩阵，形状 (n, dimension)
            
        Returns:
            np.ndarray: 相似度分数，形状 (n,)
        """
        # 由于已经归一化，点积就是余弦相似度
        if len(document_embeddings.shape) == 1:
            document_embeddings = document_embeddings.reshape(1, -1)
        
        return np.dot(document_embeddings, query_embedding)
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        将多个文本转换为嵌入向量列表
        
        Args:
            texts: 文本列表
            
        Returns:
            List[np.ndarray]: 嵌入向量列表
        """
        embeddings = self.embed_batch(texts)
        return [embeddings[i] for i in range(len(texts))]
    
    def warmup(self) -> None:
        """
        模型预热
        
        通过执行一次嵌入来预加载模型并预热 GPU/CPU 缓存
        """
        _ = self.embed_text("warmup")
    
    def clear_cache(self) -> None:
        """清除嵌入缓存"""
        self._cached_embed.cache_clear()
    
    def get_cache_info(self) -> dict:
        """
        获取缓存统计信息
        
        Returns:
            dict: 包含 hits, misses, size, maxsize
        """
        info = self._cached_embed.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "size": info.currsize,
            "maxsize": info.maxsize
        }


# 全局嵌入器实例
embedder = Embedder()
