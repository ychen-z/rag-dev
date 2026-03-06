"""
文档管理模块

负责文档的分块、嵌入、存储和检索的协调工作
"""

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import config
from .embedder import embedder
from .vector_store import vector_store


# 自定义异常类
class EmptyDocumentError(Exception):
    """空文档异常"""
    pass


@dataclass
class Document:
    """文档数据结构"""
    id: str
    title: str
    content: str
    chunks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        """从字典创建"""
        return cls(**data)


@dataclass
class SearchResult:
    """搜索结果"""
    doc_id: str
    title: str
    content: str
    chunk: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentManager:
    """
    文档管理器
    
    整合文本分块、嵌入生成和向量存储，
    提供统一的文档管理接口。
    """
    
    _instance: Optional["DocumentManager"] = None
    
    def __new__(cls) -> "DocumentManager":
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """初始化文档管理器"""
        if self._initialized:
            return
        
        self._chunk_size = config.chunk_size
        self._chunk_overlap = config.chunk_overlap
        self._data_dir = config.data_dir
        self._documents_file = self._data_dir / config.get(
            "storage.documents_file", "documents.json"
        )
        
        # 文档存储：doc_id -> Document
        self._documents: Dict[str, Document] = {}
        
        # 分块映射：chunk_index -> (doc_id, chunk_text)
        self._chunk_mapping: Dict[int, tuple] = {}
        
        # 加载已有文档
        self._load_if_exists()
        
        self._initialized = True
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = None,
        overlap: int = None
    ) -> List[str]:
        """
        将文本分割为固定大小的块
        
        Args:
            text: 输入文本
            chunk_size: 块大小（字符数）
            overlap: 块之间的重叠（字符数）
            
        Returns:
            List[str]: 文本块列表
        """
        if chunk_size is None:
            chunk_size = self._chunk_size
        if overlap is None:
            overlap = self._chunk_overlap
        
        if not text or not text.strip():
            return []
        
        # 预处理：规范化空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 如果文本较短，直接返回
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 尝试在句子边界处截断
            if end < len(text):
                # 查找最近的句子结束符
                boundary = self._find_sentence_boundary(text, end)
                if boundary > start:
                    end = boundary
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 移动到下一个位置
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, position: int) -> int:
        """
        在指定位置附近查找句子边界
        
        Args:
            text: 文本
            position: 参考位置
            
        Returns:
            int: 句子边界位置
        """
        # 句子结束符
        endings = ['。', '！', '？', '.', '!', '?', '\n']
        
        # 向后查找（不超过 50 个字符）
        search_end = min(position + 50, len(text))
        for i in range(position, search_end):
            if text[i] in endings:
                return i + 1
        
        # 向前查找（不超过 50 个字符）
        search_start = max(position - 50, 0)
        for i in range(position, search_start, -1):
            if text[i] in endings:
                return i + 1
        
        # 未找到，返回原位置
        return position
    
    def add_document(
        self,
        content: str,
        title: str = None,
        metadata: Dict[str, Any] = None
    ) -> Document:
        """
        添加文档到知识库
        
        Args:
            content: 文档内容
            title: 文档标题（可选）
            metadata: 元数据（可选）
            
        Returns:
            Document: 创建的文档对象
        """
        # 生成文档 ID
        doc_id = str(uuid.uuid4())[:8]
        
        # 默认标题
        if not title:
            # 取前 30 个字符作为标题
            title = content[:30].strip()
            if len(content) > 30:
                title += "..."
        
        # 分块
        chunks = self.chunk_text(content)
        
        if not chunks:
            raise ValueError("文档内容为空")
        
        # 创建文档对象
        doc = Document(
            id=doc_id,
            title=title,
            content=content,
            chunks=chunks,
            metadata=metadata or {}
        )
        
        # 生成嵌入并添加到向量存储
        embeddings = embedder.embed_batch(chunks)
        vector_store.add_vectors(embeddings, doc_id)
        
        # 保存文档
        self._documents[doc_id] = doc
        self.save()
        
        return doc
    
    def search_documents(
        self,
        query: str,
        top_k: int = None
    ) -> List[SearchResult]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        if top_k is None:
            top_k = config.default_top_k
        
        if not query or not query.strip():
            return []
        
        # 查询嵌入
        query_embedding = embedder.embed_text(query)
        
        # 向量搜索
        search_results = vector_store.search(query_embedding, top_k * 3)
        
        if not search_results:
            return []
        
        # 构建结果，按文档去重
        seen_docs = set()
        results = []
        
        for vec_id, doc_id, distance in search_results:
            if doc_id in seen_docs:
                continue
            
            if doc_id not in self._documents:
                continue
            
            doc = self._documents[doc_id]
            
            # 找到匹配的分块
            chunk_idx = vec_id % len(doc.chunks) if doc.chunks else 0
            chunk = doc.chunks[chunk_idx] if chunk_idx < len(doc.chunks) else ""
            
            # L2 距离转相似度分数（距离越小越相似）
            # 使用简单的转换：score = 1 / (1 + distance)
            score = 1.0 / (1.0 + distance)
            
            results.append(SearchResult(
                doc_id=doc_id,
                title=doc.title,
                content=doc.content,
                chunk=chunk,
                score=score,
                metadata=doc.metadata
            ))
            
            seen_docs.add(doc_id)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def delete_document(self, doc_id: str) -> bool:
        """
        删除文档
        
        Args:
            doc_id: 文档 ID
            
        Returns:
            bool: 是否成功删除
        """
        if doc_id not in self._documents:
            return False
        
        # 从向量存储中删除
        vector_store.delete_by_doc(doc_id)
        
        # 从文档存储中删除
        del self._documents[doc_id]
        
        # 保存更改
        self.save()
        vector_store.save()
        
        return True
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """获取文档"""
        return self._documents.get(doc_id)
    
    def list_documents(self) -> List[Document]:
        """列出所有文档"""
        return list(self._documents.values())
    
    def clear_all(self) -> None:
        """清空所有文档"""
        self._documents.clear()
        vector_store.clear()
        self.save()
        vector_store.save()
    
    def save(self) -> None:
        """保存文档到磁盘"""
        data = {
            doc_id: doc.to_dict()
            for doc_id, doc in self._documents.items()
        }
        
        with open(self._documents_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self) -> bool:
        """
        从磁盘加载文档
        
        Returns:
            bool: 是否成功加载
        """
        if not self._documents_file.exists():
            return False
        
        try:
            with open(self._documents_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self._documents = {
                doc_id: Document.from_dict(doc_data)
                for doc_id, doc_data in data.items()
            }
            return True
            
        except Exception as e:
            print(f"加载文档失败: {e}")
            return False
    
    def _load_if_exists(self) -> None:
        """如果存在已保存的数据，则加载"""
        if self._documents_file.exists():
            self.load()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        total_chunks = sum(len(doc.chunks) for doc in self._documents.values())
        total_chars = sum(len(doc.content) for doc in self._documents.values())
        
        return {
            "total_documents": len(self._documents),
            "total_chunks": total_chunks,
            "total_characters": total_chars,
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
            "vector_stats": vector_store.get_stats()
        }


# 全局文档管理器实例
document_manager = DocumentManager()
