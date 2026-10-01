"""查询已建立的本地索引。

用法：
    python scripts/search.py "奖学金材料什么时候提交" --year 2026 --category 奖学金
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="检索 CampusRAG 本地知识库")
    parser.add_argument("query", help="要检索的问题")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--year", type=int)
    parser.add_argument("--category")
    parser.add_argument("--department")
    args = parser.parse_args()

    filters = {
        key: value
        for key, value in {
            "year": args.year,
            "category": args.category,
            "department": args.department,
        }.items()
        if value is not None
    }
    try:
        results = HybridRetriever().search(args.query, top_k=args.top_k, filters=filters)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    if not results:
        print("没有找到匹配的内容。")
        return
    for position, result in enumerate(results, start=1):
        print(f"[{position}] score={result.score:.4f} source={result.metadata.get('source_path')}")
        print(result.text)
        print()


if __name__ == "__main__":
    main()
