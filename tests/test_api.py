"""
API 集成测试
"""
import unittest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import create_app
from src.document_manager import DocumentManager
from src.vector_store import VectorStore
from src.search_engine import SearchEngine


class TestAPIIntegration(unittest.TestCase):
    """API 集成测试"""
    
    def setUp(self):
        """设置测试客户端"""
        # 重置单例
        VectorStore._instance = None
        DocumentManager._instance = None
        SearchEngine._instance = None
        
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def tearDown(self):
        """清理"""
        VectorStore._instance = None
        DocumentManager._instance = None
        SearchEngine._instance = None
    
    def test_health_check(self):
        """测试健康检查端点"""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ok')
    
    def test_add_document(self):
        """测试添加文档 API"""
        response = self.client.post(
            '/api/documents',
            data=json.dumps({
                'content': '这是一个测试文档，用于验证 API 功能。',
                'title': 'API 测试',
                'metadata': {'category': '测试'}
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('document', data)
        self.assertIn('id', data['document'])
        self.assertEqual(data['success'], True)
    
    def test_add_document_empty_text(self):
        """测试添加空文档返回错误"""
        response = self.client.post(
            '/api/documents',
            data=json.dumps({
                'text': '',
                'metadata': {}
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
    
    def test_list_documents(self):
        """测试列出文档 API"""
        # 先添加一些文档
        for i in range(3):
            self.client.post(
                '/api/documents',
                data=json.dumps({
                    'content': f'测试文档 {i}，这是一段足够长的测试内容。',
                    'title': f'测试标题 {i}',
                    'metadata': {'index': i}
                }),
                content_type='application/json'
            )
        
        response = self.client.get('/api/documents')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('documents', data)
        self.assertGreaterEqual(len(data['documents']), 3)
    
    def test_list_documents_pagination(self):
        """测试文档列表（API 当前不支持分页，测试基本列表功能）"""
        # 添加 5 个文档
        for i in range(5):
            self.client.post(
                '/api/documents',
                data=json.dumps({
                    'content': f'分页测试文档 {i}，这是一段足够长的测试内容。',
                    'title': f'分页测试 {i}',
                    'metadata': {}
                }),
                content_type='application/json'
            )
        
        # 获取所有（当前 API 不支持分页参数）
        response = self.client.get('/api/documents')
        data = json.loads(response.data)
        self.assertGreaterEqual(len(data['documents']), 5)
    
    def test_delete_document(self):
        """测试删除文档 API"""
        # 先添加文档
        add_response = self.client.post(
            '/api/documents',
            data=json.dumps({
                'content': '待删除的文档，这是一段测试内容。',
                'title': '待删除',
                'metadata': {}
            }),
            content_type='application/json'
        )
        doc_id = json.loads(add_response.data)['document']['id']
        
        # 删除文档
        delete_response = self.client.delete(f'/api/documents/{doc_id}')
        self.assertEqual(delete_response.status_code, 200)
        
        data = json.loads(delete_response.data)
        self.assertEqual(data['success'], True)
    
    def test_delete_nonexistent_document(self):
        """测试删除不存在的文档"""
        response = self.client.delete('/api/documents/nonexistent_id')
        self.assertEqual(response.status_code, 404)
    
    def test_search(self):
        """测试搜索 API"""
        # 添加测试文档
        self.client.post(
            '/api/documents',
            data=json.dumps({
                'content': '机器学习是人工智能的一个重要分支，通过数据驱动学习。',
                'title': '机器学习简介',
                'metadata': {'category': '技术'}
            }),
            content_type='application/json'
        )
        
        # 搜索
        response = self.client.post(
            '/api/search',
            data=json.dumps({
                'query': '什么是人工智能？',
                'top_k': 3
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('results', data)
        self.assertIn('total', data)
    
    def test_search_empty_query(self):
        """测试空查询返回错误"""
        response = self.client.post(
            '/api/search',
            data=json.dumps({
                'query': '',
                'top_k': 3
            }),
            content_type='application/json'
        )
        
        # 空查询返回 400
        self.assertEqual(response.status_code, 400)
    
    def test_get_stats(self):
        """测试统计信息 API"""
        response = self.client.get('/api/stats')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('stats', data)
        self.assertIn('total_documents', data['stats'])
    
    def test_load_samples(self):
        """测试加载示例数据 API"""
        response = self.client.post('/api/samples')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['success'], True)
        self.assertIn('added_count', data)
    
    def test_clear_all(self):
        """测试清空所有数据 API"""
        # 先添加一些文档
        self.client.post(
            '/api/documents',
            data=json.dumps({
                'content': '待清空的文档，这是测试内容。',
                'title': '待清空',
                'metadata': {}
            }),
            content_type='application/json'
        )
        
        # 清空
        response = self.client.post('/api/clear')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['success'], True)


class TestAPIErrorHandling(unittest.TestCase):
    """API 错误处理测试"""
    
    def setUp(self):
        VectorStore._instance = None
        DocumentManager._instance = None
        SearchEngine._instance = None
        
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def tearDown(self):
        VectorStore._instance = None
        DocumentManager._instance = None
        SearchEngine._instance = None
    
    def test_invalid_json(self):
        """测试无效 JSON"""
        response = self.client.post(
            '/api/documents',
            data='not valid json',
            content_type='application/json'
        )
        
        self.assertIn(response.status_code, [400, 415])
    
    def test_missing_required_field(self):
        """测试缺少必填字段"""
        response = self.client.post(
            '/api/documents',
            data=json.dumps({
                'metadata': {}
                # 缺少 'text' 字段
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
    
    def test_404_route(self):
        """测试 404 路由"""
        response = self.client.get('/api/nonexistent')
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
