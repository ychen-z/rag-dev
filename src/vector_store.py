"""
向量存储模块

基于 FAISS 实现向量的存储、检索和持久化
"""

import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from .config import config


# 自定义异常类
class DimensionMismatchError(Exception):
    """向量维度不匹配异常"""
    pass


class IndexNotFoundError(Exception):
    """索引文件未找到异常"""
    pass


class VectorStore:
    """
    FAISS 向量存储
    
    使用 IndexFlatL2 实现精确的 L2 距离搜索。
    支持向量的增删改查和持久化。
    """
    
    _instance: Optional["VectorStore"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "VectorStore":
        """线程安全的单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """初始化向量存储"""
        if self._initialized:
            return
        
        self._dimension = config.embedding_dimension
        self._data_dir = config.data_dir
        self._index_file = self._data_dir / config.get("storage.index_file", "index.faiss")
        self._id_mapping_file = self._data_dir / config.get("storage.id_mapping_file", "id_mapping.json")
        
        # 初始化 FAISS 索引
        self._index: faiss.IndexFlatL2 = faiss.IndexFlatL2(self._dimension)
        
        # ID 映射：FAISS 内部 ID -> 文档 ID
        self._id_to_doc: Dict[int, str] = {}
        self._doc_to_ids: Dict[str, List[int]] = {}
        self._next_id: int = 0
        
        # 确保数据目录存在
        self._data_dir.mkdir(parents=True, exist_ok=True)
        
        # 尝试加载已有数据
        self._load_if_exists()
        
        self._initialized = True
    
    @property
    def dimension(self) -> int:
        """向量维度"""
        return self._dimension
    
    @property
    def total_vectors(self) -> int:
        """存储的向量总数"""
        return self._index.ntotal
    
    @property
    def total_documents(self) -> int:
        """关联的文档总数"""
        return len(self._doc_to_ids)
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        doc_id: str
    ) -> List[int]:
        """
        添加向量到索引
        
        Args:
            vectors: 向量矩阵，形状 (n, dimension)
            doc_id: 关联的文档 ID
            
        Returns:
            List[int]: 分配的内部 ID 列表
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        if vectors.shape[1] != self._dimension:
            raise ValueError(
                f"向量维度不匹配: 期望 {self._dimension}, 实际 {vectors.shape[1]}"
            )
        
        # 确保数据类型正确
        vectors = vectors.astype(np.float32)
        
        # 分配 ID
        n_vectors = vectors.shape[0]
        assigned_ids = list(range(self._next_id, self._next_id + n_vectors))
        self._next_id += n_vectors
        
        # 更新 ID 映射
        for vid in assigned_ids:
            self._id_to_doc[vid] = doc_id
        
        if doc_id not in self._doc_to_ids:
            self._doc_to_ids[doc_id] = []
        self._doc_to_ids[doc_id].extend(assigned_ids)
        
        # 添加到 FAISS 索引
        self._index.add(vectors)
        
        return assigned_ids
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = None
    ) -> List[Tuple[int, str, float]]:
        """
        相似度搜索
        
        Args:
            query_vector: 查询向量，形状 (dimension,)
            top_k: 返回的最大结果数
            
        Returns:
            List[Tuple[int, str, float]]: 
                (内部ID, 文档ID, 距离) 的列表，按距离升序排列
        """
        if top_k is None:
            top_k = config.default_top_k
        
        if self._index.ntotal == 0:
            return []
        
        # 确保查询向量形状正确
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)
        
        # 限制 k 不超过总数
        k = min(top_k, self._index.ntotal)
        
        # FAISS 搜索
        distances, indices = self._index.search(query_vector, k)
        
        # 构建结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx in self._id_to_doc:
                doc_id = self._id_to_doc[idx]
                distance = float(distances[0][i])
                results.append((int(idx), doc_id, distance))
        
        return results
    
    def search_by_doc(
        self,
        query_vector: np.ndarray,
        top_k: int = None
    ) -> List[Tuple[str, float]]:
        """
        按文档搜索，去重并返回最相关的文档
        
        Args:
            query_vector: 查询向量
            top_k: 返回的最大文档数
            
        Returns:
            List[Tuple[str, float]]: (文档ID, 最小距离) 列表
        """
        if top_k is None:
            top_k = config.default_top_k
        
        # 搜索更多结果以确保文档去重后仍有足够结果
        raw_results = self.search(query_vector, top_k * 5)
        
        # 按文档 ID 去重，保留最小距离
        doc_distances: Dict[str, float] = {}
        for _, doc_id, distance in raw_results:
            if doc_id not in doc_distances:
                doc_distances[doc_id] = distance
            else:
                doc_distances[doc_id] = min(doc_distances[doc_id], distance)
        
        # 排序并截取
        sorted_docs = sorted(doc_distances.items(), key=lambda x: x[1])
        return sorted_docs[:top_k]
    
    def delete_by_doc(self, doc_id: str) -> int:
        """
        删除指定文档的所有向量
        
        注意：FAISS IndexFlatL2 不支持直接删除，需要重建索引
        
        Args:
            doc_id: 文档 ID
            
        Returns:
            int: 删除的向量数量
        """
        if doc_id not in self._doc_to_ids:
            return 0
        
        deleted_ids = set(self._doc_to_ids[doc_id])
        deleted_count = len(deleted_ids)
        
        # 收集需要保留的向量
        vectors_to_keep = []
        new_id_mapping = {}
        new_doc_to_ids: Dict[str, List[int]] = {}
        new_id = 0
        
        # 从索引中提取所有向量
        if self._index.ntotal > 0:
            all_vectors = faiss.rev_swig_ptr(
                self._index.get_xb(), 
                self._index.ntotal * self._dimension
            ).reshape(self._index.ntotal, self._dimension).copy()
            
            for old_id in range(len(all_vectors)):
                if old_id not in deleted_ids and old_id in self._id_to_doc:
                    vectors_to_keep.append(all_vectors[old_id])
                    old_doc_id = self._id_to_doc[old_id]
                    new_id_mapping[new_id] = old_doc_id
                    
                    if old_doc_id not in new_doc_to_ids:
                        new_doc_to_ids[old_doc_id] = []
                    new_doc_to_ids[old_doc_id].append(new_id)
                    
                    new_id += 1
        
        # 重建索引
        self._index = faiss.IndexFlatL2(self._dimension)
        if vectors_to_keep:
            vectors_array = np.array(vectors_to_keep, dtype=np.float32)
            self._index.add(vectors_array)
        
        # 更新映射
        self._id_to_doc = new_id_mapping
        self._doc_to_ids = new_doc_to_ids
        self._next_id = new_id
        
        return deleted_count
    
    def clear(self) -> None:
        """清空所有数据"""
        self._index = faiss.IndexFlatL2(self._dimension)
        self._id_to_doc.clear()
        self._doc_to_ids.clear()
        self._next_id = 0
    
    def save(self) -> None:
        """保存索引和映射到磁盘"""
        # 保存 FAISS 索引
        faiss.write_index(self._index, str(self._index_file))
        
        # 保存 ID 映射
        mapping_data = {
            "id_to_doc": {str(k): v for k, v in self._id_to_doc.items()},
            "doc_to_ids": self._doc_to_ids,
            "next_id": self._next_id
        }
        with open(self._id_mapping_file, "w", encoding="utf-8") as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    
    def load(self) -> bool:
        """
        从磁盘加载索引和映射
        
        Returns:
            bool: 是否成功加载
        """
        if not self._index_file.exists() or not self._id_mapping_file.exists():
            return False
        
        try:
            # 加载 FAISS 索引
            self._index = faiss.read_index(str(self._index_file))
            
            # 加载 ID 映射
            with open(self._id_mapping_file, "r", encoding="utf-8") as f:
                mapping_data = json.load(f)
            
            self._id_to_doc = {int(k): v for k, v in mapping_data["id_to_doc"].items()}
            self._doc_to_ids = mapping_data["doc_to_ids"]
            self._next_id = mapping_data["next_id"]
            
            return True
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            # 重置为空状态
            self.clear()
            return False
    
    def _load_if_exists(self) -> None:
        """如果存在已保存的数据，则加载"""
        if self._index_file.exists() and self._id_mapping_file.exists():
            self.load()
    
    def get_stats(self) -> dict:
        """
        获取存储统计信息
        
        Returns:
            dict: 统计信息
        """
        return {
            "total_vectors": self.total_vectors,
            "total_documents": self.total_documents,
            "dimension": self._dimension,
            "index_type": "IndexFlatL2",
            "data_dir": str(self._data_dir)
        }


# 全局向量存储实例
vector_store = VectorStore()
