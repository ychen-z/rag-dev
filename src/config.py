"""
配置加载模块

从 config.yaml 读取配置并提供访问接口
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """配置管理类"""
    
    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls) -> "Config":
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> None:
        """加载配置文件"""
        config_path = self._find_config_file()
        if config_path and config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = self._default_config()
    
    def _find_config_file(self) -> Optional[Path]:
        """查找配置文件路径"""
        # 从当前文件位置向上查找
        current = Path(__file__).parent.parent
        config_file = current / "config.yaml"
        if config_file.exists():
            return config_file
        
        # 从工作目录查找
        cwd_config = Path.cwd() / "config.yaml"
        if cwd_config.exists():
            return cwd_config
        
        return None
    
    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """默认配置"""
        return {
            "language": "zh_CN",
            "embedding": {
                "model": "all-MiniLM-L6-v2",
                "dimension": 384
            },
            "chunking": {
                "chunk_size": 500,
                "overlap": 50
            },
            "search": {
                "default_top_k": 3,
                "cache_size": 100,
                "min_score": 0.0
            },
            "server": {
                "host": "0.0.0.0",
                "port": 5001,
                "debug": True
            },
            "storage": {
                "data_dir": "data",
                "index_file": "index.faiss",
                "documents_file": "documents.json",
                "id_mapping_file": "id_mapping.json"
            },
            "ui": {
                "theme": "light",
                "title": "RAG 知识库演示",
                "messages": {
                    "add_document": "添加文档",
                    "search": "搜索",
                    "delete": "删除",
                    "clear_all": "清空知识库",
                    "load_samples": "加载示例数据",
                    "no_results": "未找到相关信息",
                    "loading": "加载中...",
                    "success": "操作成功",
                    "error": "操作失败",
                    "confirm_clear": "确定要清空所有数据吗？",
                    "empty_input": "请输入内容",
                    "document_added": "文档已添加",
                    "document_deleted": "文档已删除"
                }
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔的路径
        
        Args:
            key: 配置键，如 "embedding.model"
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def language(self) -> str:
        """语言设置"""
        return self.get("language", "zh_CN")
    
    @property
    def embedding_model(self) -> str:
        """嵌入模型名称"""
        return self.get("embedding.model", "all-MiniLM-L6-v2")
    
    @property
    def embedding_dimension(self) -> int:
        """嵌入维度"""
        return self.get("embedding.dimension", 384)
    
    @property
    def chunk_size(self) -> int:
        """分块大小"""
        return self.get("chunking.chunk_size", 500)
    
    @property
    def chunk_overlap(self) -> int:
        """分块重叠"""
        return self.get("chunking.overlap", 50)
    
    @property
    def default_top_k(self) -> int:
        """默认返回结果数"""
        return self.get("search.default_top_k", 3)
    
    @property
    def cache_size(self) -> int:
        """缓存大小"""
        return self.get("search.cache_size", 100)
    
    @property
    def server_host(self) -> str:
        """服务器主机"""
        return self.get("server.host", "0.0.0.0")
    
    @property
    def server_port(self) -> int:
        """服务器端口"""
        return self.get("server.port", 5001)
    
    @property
    def server_debug(self) -> bool:
        """调试模式"""
        return self.get("server.debug", True)
    
    @property
    def data_dir(self) -> Path:
        """数据目录"""
        base = Path(__file__).parent.parent
        return base / self.get("storage.data_dir", "data")
    
    @property
    def ui_title(self) -> str:
        """UI 标题"""
        return self.get("ui.title", "RAG 知识库演示")
    
    @property
    def ui_messages(self) -> Dict[str, str]:
        """UI 消息"""
        return self.get("ui.messages", {})
    
    def reload(self) -> None:
        """重新加载配置"""
        self._load_config()


# 全局配置实例
config = Config()
