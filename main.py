#!/usr/bin/env python3
"""
RAG 知识库演示 - 启动脚本

使用方法:
    python main.py [--host HOST] [--port PORT] [--debug]
"""

import argparse
import sys
from pathlib import Path

# 添加项目目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="RAG 知识库演示服务器"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="服务器主机地址 (默认: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="服务器端口 (默认: 5001)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    args = parser.parse_args()
    
    # 导入配置和应用
    from src.config import config
    from api import create_app
    
    # 使用命令行参数或配置文件中的值
    host = args.host or config.server_host
    port = args.port or config.server_port
    debug = args.debug or config.server_debug
    
    # 创建应用
    app = create_app()
    
    print("=" * 50)
    print("  RAG 知识库演示")
    print("=" * 50)
    print(f"  服务地址: http://{host}:{port}")
    print(f"  演示页面: http://localhost:{port}/")
    print(f"  API 文档: http://localhost:{port}/api/health")
    print(f"  调试模式: {'开启' if debug else '关闭'}")
    print("=" * 50)
    print("\n按 Ctrl+C 停止服务器\n")
    
    # 启动服务器
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )


if __name__ == "__main__":
    main()
