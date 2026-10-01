"""建立或更新本地向量索引。

用法：
    python scripts/build_index.py
    python scripts/build_index.py --reset
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexer import IndexBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="建立 CampusRAG 的本地 Chroma 索引")
    parser.add_argument("--strategy", choices=["structured", "naive"], default="structured")
    parser.add_argument("--reset", action="store_true", help="删除旧索引后重建")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    try:
        result = IndexBuilder(batch_size=args.batch_size).build(
            strategy=args.strategy,
            reset=args.reset,
        )
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"索引完成：{result.document_count} 篇文档，"
        f"{result.chunk_count} 个 chunk，库内共 {result.collection_count} 条。"
    )


if __name__ == "__main__":
    main()
