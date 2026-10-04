"""显示查询在向量召回、BM25、RRF 和 Rerank 各阶段的排名。"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_settings
from src.pipeline import expand_retrieval_query
from src.reranker import Reranker
from src.retriever import HybridRetriever, SearchResult


def _label(result: SearchResult) -> str:
    title = result.metadata.get("title", "未命名文档")
    heading = result.metadata.get("heading", "")
    document_id = result.metadata.get("document_id", "")
    return f"{title} | {heading} | {document_id}"


def _print_stage(name: str, results: list[SearchResult]) -> None:
    print(f"\n=== {name} ({len(results)}) ===")
    for rank, result in enumerate(results, start=1):
        print(
            f"[{rank:02d}] score={result.score:.4f} "
            f"vector={result.vector_rank or '-'} bm25={result.bm25_rank or '-'} "
            f"{_label(result)}"
        )
        print(f"     {result.text[:140].replace(chr(10), ' ')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="追踪 CampusRAG 查询的各级检索结果")
    parser.add_argument("question")
    parser.add_argument("--year", type=int)
    parser.add_argument("--category")
    parser.add_argument("--department")
    parser.add_argument("--retrieval-k", type=int, default=20)
    parser.add_argument("--rerank-k", type=int, default=5)
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
    retrieval_query = expand_retrieval_query(args.question)
    print(f"原始问题：{args.question}")
    print(f"检索问题：{retrieval_query}")
    print(f"Metadata filters：{filters or '{}'}")

    retriever = HybridRetriever()
    trace = retriever.search_with_trace(
        retrieval_query,
        top_k=args.retrieval_k,
        filters=filters,
        vector_k=args.retrieval_k,
        bm25_k=args.retrieval_k,
    )
    _print_stage("Vector", trace.vector_results)
    _print_stage("BM25", trace.bm25_results)
    _print_stage("RRF fused", trace.fused_results)

    reranked = Reranker(get_settings().reranker).rerank(
        retrieval_query,
        trace.fused_results,
        top_n=args.rerank_k,
    )
    _print_stage("Rerank", reranked)


if __name__ == "__main__":
    main()
