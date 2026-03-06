"""
Flask API 路由定义

提供 RAG 知识库的 REST API 接口
"""

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.document_manager import document_manager, SearchResult
from src.vector_store import vector_store


def create_app() -> Flask:
    """创建 Flask 应用"""
    app = Flask(
        __name__,
        static_folder=str(Path(__file__).parent.parent / "demo"),
        static_url_path="/demo"
    )
    
    # 启用 CORS
    CORS(app)
    
    # 注册路由
    register_routes(app)
    
    return app


def register_routes(app: Flask) -> None:
    """注册所有路由"""
    
    @app.errorhandler(400)
    def bad_request(error):
        """处理 400 错误"""
        return jsonify({
            "success": False,
            "message": "请求格式错误"
        }), 400
    
    @app.before_request
    def validate_json():
        """验证 JSON 请求"""
        if request.method in ['POST', 'PUT', 'PATCH']:
            content_type = request.content_type or ''
            if 'application/json' in content_type:
                try:
                    request.get_json(force=True)
                except Exception:
                    from flask import abort
                    abort(400)
    
    @app.route("/")
    def index():
        """重定向到演示页面"""
        return send_from_directory(app.static_folder, "index.html")
    
    @app.route("/api/health")
    def health():
        """健康检查"""
        return jsonify({
            "status": "ok",
            "message": config.ui_messages.get("success", "操作成功")
        })
    
    @app.route("/api/documents", methods=["POST"])
    def add_document():
        """
        添加文档
        
        请求体:
            {
                "content": "文档内容",
                "title": "可选标题",
                "metadata": {}  // 可选元数据
            }
        """
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    "success": False,
                    "message": config.ui_messages.get("empty_input", "请输入内容")
                }), 400
            
            content = data.get("content", "").strip()
            
            if not content:
                return jsonify({
                    "success": False,
                    "message": config.ui_messages.get("empty_input", "请输入内容")
                }), 400
            
            title = data.get("title")
            metadata = data.get("metadata", {})
            
            doc = document_manager.add_document(
                content=content,
                title=title,
                metadata=metadata
            )
            
            return jsonify({
                "success": True,
                "message": config.ui_messages.get("document_added", "文档已添加"),
                "document": {
                    "id": doc.id,
                    "title": doc.title,
                    "chunks_count": len(doc.chunks),
                    "created_at": doc.created_at
                }
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"{config.ui_messages.get('error', '操作失败')}: {str(e)}"
            }), 500
    
    @app.route("/api/documents", methods=["GET"])
    def list_documents():
        """列出所有文档"""
        try:
            docs = document_manager.list_documents()
            
            return jsonify({
                "success": True,
                "documents": [
                    {
                        "id": doc.id,
                        "title": doc.title,
                        "content_preview": doc.content[:100] + "..." if len(doc.content) > 100 else doc.content,
                        "chunks_count": len(doc.chunks),
                        "created_at": doc.created_at,
                        "metadata": doc.metadata
                    }
                    for doc in docs
                ],
                "total": len(docs)
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "message": str(e)
            }), 500
    
    @app.route("/api/documents/<doc_id>", methods=["GET"])
    def get_document(doc_id: str):
        """获取单个文档"""
        try:
            doc = document_manager.get_document(doc_id)
            
            if not doc:
                return jsonify({
                    "success": False,
                    "message": "文档不存在"
                }), 404
            
            return jsonify({
                "success": True,
                "document": {
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                    "chunks": doc.chunks,
                    "chunks_count": len(doc.chunks),
                    "created_at": doc.created_at,
                    "metadata": doc.metadata
                }
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "message": str(e)
            }), 500
    
    @app.route("/api/documents/<doc_id>", methods=["DELETE"])
    def delete_document(doc_id: str):
        """删除文档"""
        try:
            success = document_manager.delete_document(doc_id)
            
            if success:
                return jsonify({
                    "success": True,
                    "message": config.ui_messages.get("document_deleted", "文档已删除")
                })
            else:
                return jsonify({
                    "success": False,
                    "message": "文档不存在"
                }), 404
                
        except Exception as e:
            return jsonify({
                "success": False,
                "message": str(e)
            }), 500
    
    @app.route("/api/search", methods=["POST"])
    def search():
        """
        搜索文档
        
        请求体:
            {
                "query": "搜索关键词",
                "top_k": 3  // 可选，默认 3
            }
        """
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    "success": False,
                    "message": config.ui_messages.get("empty_input", "请输入内容")
                }), 400
            
            query = data.get("query", "").strip()
            
            if not query:
                return jsonify({
                    "success": False,
                    "message": config.ui_messages.get("empty_input", "请输入内容")
                }), 400
            
            top_k = data.get("top_k", config.default_top_k)
            
            results = document_manager.search_documents(query, top_k)
            
            if not results:
                return jsonify({
                    "success": True,
                    "message": config.ui_messages.get("no_results", "未找到相关信息"),
                    "results": [],
                    "total": 0
                })
            
            return jsonify({
                "success": True,
                "results": [
                    {
                        "doc_id": r.doc_id,
                        "title": r.title,
                        "chunk": r.chunk,
                        "score": round(r.score, 4),
                        "content_preview": r.content[:200] + "..." if len(r.content) > 200 else r.content
                    }
                    for r in results
                ],
                "total": len(results)
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "message": str(e)
            }), 500
    
    @app.route("/api/stats", methods=["GET"])
    def get_stats():
        """获取统计信息"""
        try:
            stats = document_manager.get_stats()
            
            return jsonify({
                "success": True,
                "stats": stats
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "message": str(e)
            }), 500
    
    @app.route("/api/clear", methods=["POST"])
    def clear_all():
        """清空所有数据"""
        try:
            document_manager.clear_all()
            
            return jsonify({
                "success": True,
                "message": config.ui_messages.get("success", "操作成功")
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "message": str(e)
            }), 500
    
    @app.route("/api/samples", methods=["POST"])
    def load_samples():
        """加载示例数据"""
        try:
            samples_dir = Path(__file__).parent.parent / "data" / "samples"
            
            # 示例文档
            sample_documents = [
                {
                    "title": "人工智能简介",
                    "content": """人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
                    旨在创建能够执行通常需要人类智能的任务的系统。这些任务包括视觉感知、
                    语音识别、决策制定和语言翻译。AI技术正在快速发展，并在医疗、金融、
                    交通等多个领域得到广泛应用。机器学习是AI的一个重要子领域，
                    它使计算机能够从数据中学习并改进其性能，而无需明确编程。
                    深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的工作方式。"""
                },
                {
                    "title": "向量数据库介绍",
                    "content": """向量数据库是一种专门设计用于存储和查询高维向量数据的数据库系统。
                    它们特别适用于处理嵌入向量，这些向量是通过机器学习模型从文本、图像或
                    其他数据类型中提取的数值表示。向量数据库的核心功能是相似性搜索，
                    即找到与给定查询向量最相似的向量。常见的向量数据库包括FAISS、Milvus、
                    Pinecone和Weaviate等。这些数据库使用各种索引算法来加速搜索，
                    如IVF（倒排文件）、HNSW（分层可导航小世界图）和PQ（乘积量化）等。"""
                },
                {
                    "title": "RAG技术详解",
                    "content": """RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合
                    信息检索和文本生成的技术。它首先从知识库中检索相关文档，然后将这些文档
                    作为上下文提供给大语言模型，以生成更准确、更相关的回答。RAG的优势在于
                    能够利用外部知识库，减少大语言模型的幻觉问题，并提供可追溯的信息来源。
                    RAG系统通常包含以下组件：文档处理器（用于分块和预处理文档）、
                    嵌入模型（将文本转换为向量）、向量数据库（存储和检索向量）、
                    以及生成模型（根据检索结果生成回答）。"""
                },
                {
                    "title": "Python编程基础",
                    "content": """Python是一种高级编程语言，以其简洁易读的语法和强大的功能而闻名。
                    Python支持多种编程范式，包括面向对象、函数式和过程式编程。它拥有丰富的
                    标准库和第三方包生态系统，使开发者能够快速构建各种应用。Python在数据科学、
                    机器学习、Web开发和自动化脚本等领域都有广泛应用。Python的设计哲学强调
                    代码的可读性，使用缩进来定义代码块，而不是使用大括号。
                    Python的主要特点包括动态类型、自动内存管理和解释执行。"""
                }
            ]
            
            added_count = 0
            for sample in sample_documents:
                try:
                    document_manager.add_document(
                        content=sample["content"],
                        title=sample["title"],
                        metadata={"source": "sample"}
                    )
                    added_count += 1
                except Exception:
                    pass  # 忽略重复添加错误
            
            return jsonify({
                "success": True,
                "message": f"已加载 {added_count} 个示例文档",
                "added_count": added_count
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "message": str(e)
            }), 500
    
    @app.route("/api/config", methods=["GET"])
    def get_config():
        """获取前端配置（UI文本等）"""
        return jsonify({
            "success": True,
            "config": {
                "title": config.ui_title,
                "messages": config.ui_messages,
                "language": config.language
            }
        })
