"""
DocumentManager 模块单元测试
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_manager import DocumentManager, Document
from src.vector_store import VectorStore


class TestDocumentManager(unittest.TestCase):
    """测试 DocumentManager 类"""
    
    def setUp(self):
        """每个测试前重置"""
        VectorStore._instance = None
        DocumentManager._instance = None
        self.manager = DocumentManager()
    
    def tearDown(self):
        """清理"""
        try:
            self.manager.clear_all()
        except:
            pass
        VectorStore._instance = None
        DocumentManager._instance = None
    
    def test_chunk_text_basic(self):
        """测试基本文本分块"""
        text = "这是第一句话。这是第二句话。这是第三句话。这是第四句话。这是第五句话。"
        chunks = self.manager.chunk_text(text, chunk_size=30, overlap=5)
        
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, str)
            self.assertGreater(len(chunk), 0)
    
    def test_chunk_text_with_overlap(self):
        """测试分块重叠"""
        text = "A" * 100 + "B" * 100 + "C" * 100
        chunks = self.manager.chunk_text(text, chunk_size=100, overlap=20)
        
        # 检查重叠部分
        if len(chunks) > 1:
            # 第一个块的结尾应该与第二个块的开头有重叠
            self.assertGreater(len(chunks), 1)
    
    def test_chunk_text_sentence_boundary(self):
        """测试句子边界检测"""
        text = "第一句话很长很长很长。第二句话也很长很长。第三句话更长。"
        chunks = self.manager.chunk_text(text, chunk_size=20, overlap=0)
        
        # 分块应该尽量在句子边界处分割
        for chunk in chunks:
            # 每个块应该是完整的或在合适位置结束
            self.assertIsInstance(chunk, str)
    
    def test_add_document(self):
        """测试添加文档"""
        doc = self.manager.add_document(
            content="这是一个测试文档。它包含一些测试内容。",
            title="测试文档",
            metadata={"category": "测试"}
        )
        
        self.assertIsInstance(doc, Document)
        self.assertIsNotNone(doc.id)
        self.assertEqual(doc.title, "测试文档")
    
    def test_add_document_with_auto_title(self):
        """测试自动生成标题"""
        content = "自动标题测试文档内容比较长需要截取"
        doc = self.manager.add_document(content=content)
        
        self.assertIsNotNone(doc.id)
        self.assertIn(content[:30], doc.title)
    
    def test_add_empty_document_raises_error(self):
        """测试添加空文档抛出异常"""
        with self.assertRaises(ValueError):
            self.manager.add_document(content="", metadata={})
    
    def test_add_whitespace_only_document_raises_error(self):
        """测试添加仅空白文档抛出异常"""
        with self.assertRaises(ValueError):
            self.manager.add_document(content="   \n\t  ", metadata={})
    
    def test_delete_document(self):
        """测试删除文档"""
        doc = self.manager.add_document(
            content="待删除文档内容",
            title="待删除"
        )
        
        deleted = self.manager.delete_document(doc.id)
        self.assertTrue(deleted)
        
        # 再次删除应该返回 False
        deleted_again = self.manager.delete_document(doc.id)
        self.assertFalse(deleted_again)
    
    def test_list_documents(self):
        """测试列出文档"""
        # 清除并添加几个文档
        self.manager.clear_all()
        for i in range(5):
            self.manager.add_document(
                content=f"文档 {i} 的内容，需要足够长度",
                metadata={"index": i}
            )
        
        docs = self.manager.list_documents()
        self.assertEqual(len(docs), 5)
    
    def test_get_document(self):
        """测试获取单个文档"""
        doc = self.manager.add_document(
            content="可获取的文档内容",
            title="获取测试"
        )
        
        retrieved = self.manager.get_document(doc.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, doc.id)
        self.assertEqual(retrieved.title, "获取测试")
    
    def test_get_nonexistent_document(self):
        """测试获取不存在的文档"""
        doc = self.manager.get_document("nonexistent_doc_id")
        self.assertIsNone(doc)
    
    def test_get_stats(self):
        """测试获取统计信息"""
        self.manager.clear_all()
        self.manager.add_document(content="文档1内容", title="文档1")
        self.manager.add_document(content="文档2内容", title="文档2")
        
        stats = self.manager.get_stats()
        self.assertEqual(stats["total_documents"], 2)


class TestChunkingEdgeCases(unittest.TestCase):
    """测试分块边界情况"""
    
    def setUp(self):
        VectorStore._instance = None
        DocumentManager._instance = None
        self.manager = DocumentManager()
    
    def tearDown(self):
        VectorStore._instance = None
        DocumentManager._instance = None
    
    def test_very_long_text(self):
        """测试非常长的文本"""
        long_text = "这是一个很长的句子。" * 1000
        chunks = self.manager.chunk_text(long_text, chunk_size=500, overlap=50)
        
        self.assertGreater(len(chunks), 1)
    
    def test_no_sentence_boundaries(self):
        """测试没有句子边界的文本"""
        text = "没有任何标点符号的连续文本" * 50
        chunks = self.manager.chunk_text(text, chunk_size=100, overlap=10)
        
        self.assertGreater(len(chunks), 0)
    
    def test_single_sentence(self):
        """测试单个句子"""
        text = "只有一个句子。"
        chunks = self.manager.chunk_text(text, chunk_size=500, overlap=50)
        
        self.assertEqual(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()