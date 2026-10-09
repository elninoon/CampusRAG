"""建立或更新本地向量索引。

用法：
    python scripts/build_index.py
    python scripts/build_index.py --reset
"""
import argparse
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_settings
from src.indexer import IndexBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="建立 CampusRAG 的本地 Chroma 索引")
    parser.add_argument("--strategy", choices=["structured", "naive"], default="structured")
    parser.add_argument("--reset", action="store_true", help="删除旧索引后重建")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--index-dir", help="指定索引目录，便于建立隔离实验索引")
    parser.add_argument("--max-chars", type=int, default=800, help="结构化切块最大字符数")
    parser.add_argument("--overlap", type=int, default=40, help="结构化长段切块重叠字符数")
    parser.add_argument("--chunk-size", type=int, default=500, help="固定长度切块长度")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="固定长度切块重叠字符数")
    args = parser.parse_args()

    try:
        settings = get_settings()
        if args.index_dir:
            settings = replace(settings, index_dir=os.path.abspath(args.index_dir))
        result = IndexBuilder(settings=settings, batch_size=args.batch_size).build(
            strategy=args.strategy,
            reset=args.reset,
            max_chars=args.max_chars,
            overlap=args.overlap,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"索引完成：{result.document_count} 篇文档，"
        f"{result.chunk_count} 个 chunk，库内共 {result.collection_count} 条。"
    )


if __name__ == "__main__":
    main()
