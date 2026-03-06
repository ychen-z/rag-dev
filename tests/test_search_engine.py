"""
SearchEngine 模块单元测试
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search_engine import SearchEngine
from src.document_manager import DocumentManager
from src.vector_store import VectorStore


class TestSearchEngine(unittest.TestCase):
    """测试 SearchEngine 类"""
    
    def setUp(self):
        """每个测试前重置单例"""
        VectorStore._instance = None
        DocumentManager._instance = None
        SearchEngine._instance = None
        self.engine = SearchEngine()
        
        # 添加测试文档
        self.engine._document_manager.add_document(
            content="机器学习是人工智能的一个重要分支，通过数据驱动的方法让计算机学习。",
            metadata={"category": "技术", "title": "机器学习简介"}
        )
        self.engine._document_manager.add_document(
            content="深度学习使用多层神经网络来处理复杂的模式识别任务。",
            metadata={"category": "技术", "title": "深度学习"}
        )
        self.engine._document_manager.add_document(
            content="今天天气晴朗，适合户外运动和野餐。",
            metadata={"category": "生活", "title": "天气预报"}
        )
    
    def tearDown(self):
        """清理"""
        VectorStore._instance = None
        DocumentManager._instance = None
        SearchEngine._instance = None
    
    def test_search_returns_results(self):
        """测试搜索返回结果"""
        result = self.engine.search("什么是机器学习？", k=3)
        
        self.assertIn("results", result)
        self.assertIn("query", result)
        self.assertIn("elapsed_ms", result)
        self.assertGreater(len(result["results"]), 0)
    
    def test_search_relevance(self):
        """测试搜索相关性"""
        result = self.engine.search("人工智能和机器学习", k=3)
        
        # 第一个结果应该是关于机器学习的文档
        if result["results"]:
            top_result = result["results"][0]
            self.assertIn("机器学习", top_result["text"])
    
    def test_search_with_threshold(self):
        """测试相似度阈值过滤"""
        result = self.engine.search("机器学习", k=10, threshold=0.3)
        
        for r in result["results"]:
            self.assertGreaterEqual(r["score"], 0.3)
    
    def test_search_with_filters(self):
        """测试元数据过滤"""
        result = self.engine.search("学习", k=10, filters={"category": "技术"})
        
        for r in result["results"]:
            self.assertEqual(r["metadata"]["category"], "技术")
    
    def test_search_empty_query(self):
        """测试空查询"""
        result = self.engine.search("", k=3)
        
        self.assertEqual(len(result["results"]), 0)
    
    def test_batch_search(self):
        """测试批量搜索"""
        queries = ["机器学习", "天气", "神经网络"]
        results = self.engine.batch_search(queries, k=2)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn("results", result)
    
    def test_distance_to_similarity(self):
        """测试距离到相似度转换"""
        # 距离为 0 应该返回相似度 1
        sim_0 = SearchEngine._distance_to_similarity(0)
        self.assertAlmostEqual(sim_0, 1.0)
        
        # 距离越大，相似度越小
        sim_1 = SearchEngine._distance_to_similarity(1)
        self.assertLess(sim_1, 1.0)
        self.assertGreater(sim_1, 0.0)
    
    def test_cache_functionality(self):
        """测试查询缓存"""
        query = "测试缓存查询"
        
        # 第一次查询
        self.engine.search(query, k=1)
        
        # 应该被缓存
        self.assertIn(query, self.engine._query_cache)
    
    def test_clear_cache(self):
        """测试清除缓存"""
        self.engine.search("缓存测试", k=1)
        self.assertGreater(len(self.engine._query_cache), 0)
        
        self.engine.clear_cache()
        self.assertEqual(len(self.engine._query_cache), 0)
    
    def test_get_stats(self):
        """测试获取统计信息"""
        stats = self.engine.get_stats()
        
        self.assertIn("cache_size", stats)
        self.assertIn("vector_store_stats", stats)
        self.assertIn("document_count", stats)
    
    def test_result_structure(self):
        """测试结果结构完整性"""
        result = self.engine.search("机器学习", k=1)
        
        if result["results"]:
            r = result["results"][0]
            self.assertIn("doc_id", r)
            self.assertIn("text", r)
            self.assertIn("score", r)
            self.assertIn("distance", r)
            self.assertIn("metadata", r)
            self.assertIn("rank", r)


class TestMatchFilters(unittest.TestCase):
    """测试元数据过滤逻辑"""
    
    def test_single_filter_match(self):
        """测试单个过滤条件匹配"""
        metadata = {"category": "技术", "level": "高级"}
        filters = {"category": "技术"}
        
        self.assertTrue(SearchEngine._match_filters(metadata, filters))
    
    def test_single_filter_no_match(self):
        """测试单个过滤条件不匹配"""
        metadata = {"category": "技术"}
        filters = {"category": "生活"}
        
        self.assertFalse(SearchEngine._match_filters(metadata, filters))
    
    def test_multiple_filters_match(self):
        """测试多个过滤条件匹配"""
        metadata = {"category": "技术", "level": "高级", "author": "张三"}
        filters = {"category": "技术", "level": "高级"}
        
        self.assertTrue(SearchEngine._match_filters(metadata, filters))
    
    def test_filter_key_not_in_metadata(self):
        """测试过滤键不存在于元数据"""
        metadata = {"category": "技术"}
        filters = {"level": "高级"}
        
        self.assertFalse(SearchEngine._match_filters(metadata, filters))
    
    def test_list_filter_match(self):
        """测试列表过滤条件匹配"""
        metadata = {"category": "技术"}
        filters = {"category": ["技术", "科学"]}
        
        self.assertTrue(SearchEngine._match_filters(metadata, filters))
    
    def test_list_filter_no_match(self):
        """测试列表过滤条件不匹配"""
        metadata = {"category": "生活"}
        filters = {"category": ["技术", "科学"]}
        
        self.assertFalse(SearchEngine._match_filters(metadata, filters))


if __name__ == "__main__":
    unittest.main()
