"""
VectorStore 模块单元测试
"""
import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):
    """测试 VectorStore 类"""
    
    def setUp(self):
        """每个测试前重置单例"""
        VectorStore._instance = None
        self.store = VectorStore()
        self.store.clear()  # 确保干净状态
    
    def tearDown(self):
        """清理"""
        self.store.clear()
        VectorStore._instance = None
    
    def test_add_vector(self):
        """测试添加单个向量"""
        vector = np.random.rand(384).astype(np.float32)
        ids = self.store.add_vectors(vector, "doc_1")
        
        self.assertEqual(len(ids), 1)
        stats = self.store.get_stats()
        self.assertEqual(stats["total_vectors"], 1)
    
    def test_add_vectors_batch(self):
        """测试批量添加向量"""
        vectors = np.random.rand(5, 384).astype(np.float32)
        
        ids = self.store.add_vectors(vectors, "doc_batch")
        
        self.assertEqual(len(ids), 5)
        stats = self.store.get_stats()
        self.assertEqual(stats["total_vectors"], 5)
    
    def test_search(self):
        """测试搜索功能"""
        # 添加一些向量
        vectors = np.random.rand(10, 384).astype(np.float32)
        self.store.add_vectors(vectors, "doc_search")
        
        # 使用第一个向量进行搜索
        query = vectors[0]
        results = self.store.search(query, top_k=3)
        
        self.assertEqual(len(results), 3)
        # 结果格式为 (vec_id, doc_id, distance)
        self.assertEqual(results[0][1], "doc_search")
        self.assertAlmostEqual(results[0][2], 0.0, places=5)
    
    def test_search_k_larger_than_total(self):
        """测试 k 大于总向量数的情况"""
        vectors = np.random.rand(3, 384).astype(np.float32)
        self.store.add_vectors(vectors, "doc_small")
        
        query = vectors[0]
        results = self.store.search(query, top_k=10)
        
        self.assertEqual(len(results), 3)
    
    def test_delete_vector(self):
        """测试删除向量"""
        vectors = np.random.rand(5, 384).astype(np.float32)
        self.store.add_vectors(vectors, "doc_delete")
        
        deleted = self.store.delete_by_doc("doc_delete")
        
        self.assertEqual(deleted, 5)
        stats = self.store.get_stats()
        self.assertEqual(stats["total_vectors"], 0)
    
    def test_save_and_load(self):
        """测试保存和加载"""
        vectors = np.random.rand(5, 384).astype(np.float32)
        self.store.add_vectors(vectors, "doc_persist")
        
        # 保存
        self.store.save()
        
        # 重置单例并重新加载
        VectorStore._instance = None
        new_store = VectorStore()
        
        stats = new_store.get_stats()
        self.assertEqual(stats["total_vectors"], 5)
    
    def test_dimension_mismatch(self):
        """测试维度不匹配异常"""
        wrong_dim_vector = np.random.rand(256).astype(np.float32)
        
        with self.assertRaises(ValueError):
            self.store.add_vectors(wrong_dim_vector, "doc_wrong")
    
    def test_load_nonexistent(self):
        """测试加载不存在的索引返回 False"""
        VectorStore._instance = None
        store = VectorStore()
        store.clear()
        
        # 删除可能存在的索引文件
        index_file = store._index_file
        mapping_file = store._id_mapping_file
        
        if os.path.exists(index_file):
            os.remove(index_file)
        if os.path.exists(mapping_file):
            os.remove(mapping_file)
        
        result = store.load()
        self.assertFalse(result)
    
    def test_get_stats(self):
        """测试获取统计信息"""
        vectors = np.random.rand(3, 384).astype(np.float32)
        self.store.add_vectors(vectors, "doc_stats")
        
        stats = self.store.get_stats()
        
        self.assertIn("total_vectors", stats)
        self.assertIn("dimension", stats)
        self.assertEqual(stats["total_vectors"], 3)
        self.assertEqual(stats["dimension"], 384)
    
    def test_search_empty_index(self):
        """测试空索引搜索"""
        query = np.random.rand(384).astype(np.float32)
        results = self.store.search(query, top_k=5)
        
        self.assertEqual(len(results), 0)
    
    def test_total_documents(self):
        """测试文档计数"""
        vectors1 = np.random.rand(3, 384).astype(np.float32)
        vectors2 = np.random.rand(2, 384).astype(np.float32)
        
        self.store.add_vectors(vectors1, "doc_a")
        self.store.add_vectors(vectors2, "doc_b")
        
        self.assertEqual(self.store.total_documents, 2)
        self.assertEqual(self.store.total_vectors, 5)


if __name__ == "__main__":
    unittest.main()
