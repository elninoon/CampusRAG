"""评估本地索引的检索命中率。"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator import evaluate_retrieval, load_cases
from src.retriever import HybridRetriever


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CASES = os.path.join(ROOT, "data", "eval", "retrieval_cases.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 CampusRAG 的检索质量")
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    try:
        result = evaluate_retrieval(HybridRetriever(), load_cases(args.cases), args.top_k)
    except (RuntimeError, ValueError, OSError) as exc:
        parser.error(str(exc))

    print(f"Hit@{args.top_k}: {result.hit_at_k:.1%} ({result.hit_count}/{result.case_count})")
    if result.missed_questions:
        print("未命中问题：")
        for question in result.missed_questions:
            print(f"- {question}")


if __name__ == "__main__":
    main()
