"""向 CampusRAG 提问并显示回答及其来源。"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="向 CampusRAG 提问")
    parser.add_argument("question", help="问题")
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
        answer = RAGPipeline().ask(args.question, filters=filters)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print(answer.text)
    if not answer.citation_check.valid:
        print(f"\n引用校验警告：{'；'.join(answer.citation_check.errors)}")
    if answer.sources:
        print("\n来源：")
        for number, source in enumerate(answer.sources, start=1):
            print(
                f"[{number}] {source.metadata.get('title', '未命名文档')}"
                f" | {source.metadata.get('source_path', '未知来源')}"
            )


if __name__ == "__main__":
    main()
