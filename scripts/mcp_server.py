"""启动 CampusRAG MCP Server。

用法：
    python scripts/mcp_server.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mcp_server import main


if __name__ == "__main__":
    main()
