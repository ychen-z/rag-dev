"""
Embedder 模块单元测试
"""
import unittest
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedder import Embedder


class TestEmbedder(unittest.TestCase):
    """测试 Embedder 类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化，加载模型（只执行一次）"""
        cls.embedder = Embedder()
    
    def test_embed_text_returns_numpy_array(self):
        """测试 embed_text 返回 numpy 数组"""
        result = self.embedder.embed_text("测试文本")
        self.assertIsInstance(result, np.ndarray)
    
    def test_embed_text_correct_dimension(self):
        """测试嵌入向量维度正确 (384)"""
        result = self.embedder.embed_text("测试文本")
        self.assertEqual(result.shape[0], 384)
    
    def test_embed_text_normalized(self):
        """测试嵌入向量已归一化"""
        result = self.embedder.embed_text("测试文本")
        norm = np.linalg.norm(result)
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_embed_texts_batch(self):
        """测试批量嵌入"""
        texts = ["文本一", "文本二", "文本三"]
        results = self.embedder.embed_texts(texts)
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result.shape[0], 384)
    
    def test_similar_texts_have_high_similarity(self):
        """测试相似文本具有较高的相似度"""
        vec1 = self.embedder.embed_text("机器学习是人工智能的一个分支")
        vec2 = self.embedder.embed_text("人工智能包括机器学习技术")
        
        # 计算余弦相似度（由于已归一化，点积即为余弦相似度）
        similarity = np.dot(vec1, vec2)
        self.assertGreater(similarity, 0.5)
    
    def test_different_texts_have_lower_similarity(self):
        """测试不同主题文本具有较低的相似度"""
        vec1 = self.embedder.embed_text("机器学习是人工智能的一个分支")
        vec2 = self.embedder.embed_text("今天的天气非常好，适合户外运动")
        
        similarity = np.dot(vec1, vec2)
        self.assertLess(similarity, 0.5)
    
    def test_empty_text(self):
        """测试空文本"""
        result = self.embedder.embed_text("")
        self.assertEqual(result.shape[0], 384)
    
    def test_warmup(self):
        """测试模型预热"""
        # warmup 不应该抛出异常
        self.embedder.warmup()


class TestEmbedderCaching(unittest.TestCase):
    """测试 Embedder 缓存功能"""
    
    def test_same_text_returns_same_vector(self):
        """测试相同文本返回相同向量"""
        embedder = Embedder()
        text = "测试缓存功能"
        
        vec1 = embedder.embed_text(text)
        vec2 = embedder.embed_text(text)
        
        np.testing.assert_array_equal(vec1, vec2)


if __name__ == "__main__":
    unittest.main()
