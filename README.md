# RAG 知识库演示

基于 FAISS 向量数据库的 RAG (Retrieval-Augmented Generation) 知识库演示项目。

## 功能特性

- 🔍 **语义搜索**: 基于向量相似度的智能检索
- 📄 **文档管理**: 支持文档的增删改查
- 🧠 **文本嵌入**: 使用 MiniLM-L6-v2 模型生成向量
- 🎨 **现代界面**: Tailwind CSS 构建的响应式 Demo 页面

## 快速开始

### 1. 安装依赖

```bash
cd vector-data
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python main
```

### 3. 访问 Demo

打开浏览器访问: http://localhost:5001/demo

## 项目结构

```
vector-data/
├── src/                    # 核心模块
│   ├── config.py           # 配置加载
│   ├── embedder.py         # 文本嵌入
│   ├── vector_store.py     # 向量存储
│   ├── document_manager.py # 文档管理
│   └── search_engine.py    # 搜索引擎
├── api/
│   └── app.py              # Flask API
├── demo/
│   └── index.html          # Demo 界面
├── data/
│   └── samples/            # 示例数据
├── config.yaml             # 配置文件
└── requirements.txt        # 依赖
```

## API 文档

| 方法   | 路径                  | 描述         |
| ------ | --------------------- | ------------ |
| POST   | `/api/documents`      | 添加文档     |
| GET    | `/api/documents`      | 列出文档     |
| DELETE | `/api/documents/<id>` | 删除文档     |
| POST   | `/api/search`         | 搜索         |
| GET    | `/api/stats`          | 统计信息     |
| POST   | `/api/load-samples`   | 加载示例数据 |

## 配置说明

编辑 `config.yaml` 自定义配置:

```yaml
language: zh_CN # 默认语言
embedding:
  model: all-MiniLM-L6-v2
  dimension: 384
chunking:
  chunk_size: 500
  overlap: 50
```

## 许可证

MIT License
