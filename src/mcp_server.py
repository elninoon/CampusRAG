"""CampusRAG MCP Server.

把成熟的 RAG 能力暴露成 MCP tools，供外部 Agent 调用。

工具：
- campus_rag_search: 只做混合检索，不调用 reranker / LLM
- campus_rag_ask: 走完整 RAGPipeline，返回受资料约束回答和来源
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List

from src.pipeline import RAGPipeline
from src.retriever import HybridRetriever, SearchResult

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("缺少 mcp 依赖，请先执行：python -m pip install -r requirements.txt") from exc


mcp = FastMCP("CampusRAG")
logger = logging.getLogger(__name__)


def _filters(
    year: int | None = None,
    category: str = "",
    department: str = "",
) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    if year is not None:
        filters["year"] = year
    if category.strip():
        filters["category"] = category.strip()
    if department.strip():
        filters["department"] = department.strip()
    return filters


def _source_payload(result: SearchResult) -> Dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "text": result.text,
        "metadata": result.metadata,
        "score": result.score,
        "vector_rank": result.vector_rank,
        "bm25_rank": result.bm25_rank,
    }


@lru_cache(maxsize=1)
def _retriever() -> HybridRetriever:
    return HybridRetriever()


@lru_cache(maxsize=1)
def _pipeline() -> RAGPipeline:
    return RAGPipeline()


@mcp.tool()
def campus_rag_search(
    query: str,
    top_k: int = 5,
    year: int | None = None,
    category: str = "",
    department: str = "",
) -> List[Dict[str, Any]]:
    """检索校园资料，返回 RRF 融合后的 chunk 列表。

    这个工具只做检索，不调用 reranker 或 LLM，适合 Agent 先查证据。
    """
    logger.info("MCP campus_rag_search started: top_k=%s", top_k)
    retriever = _retriever()
    logger.info("MCP campus_rag_search retriever ready")
    results = retriever.search(
        query=query,
        top_k=top_k,
        filters=_filters(year=year, category=category, department=department),
        vector_k=max(top_k, 5),
        bm25_k=max(top_k, 5),
    )
    payload = [_source_payload(result) for result in results]
    logger.info("MCP campus_rag_search completed: results=%s", len(payload))
    return payload


@mcp.tool()
def campus_rag_ask(
    question: str,
    history: List[Dict[str, str]] | None = None,
    year: int | None = None,
    category: str = "",
    department: str = "",
    retrieval_k: int = 8,
    rerank_k: int = 3,
) -> Dict[str, Any]:
    """基于校园资料回答问题，并返回答案、来源和引用校验结果。

    这个工具会调用完整 RAGPipeline，包括混合检索、rerank 和 LLM 生成。
    """
    answer = _pipeline().ask(
        question=question,
        history=history,
        filters=_filters(year=year, category=category, department=department),
        retrieval_k=retrieval_k,
        rerank_k=rerank_k,
    )
    return {
        "answer": answer.text,
        "citation_valid": answer.citation_check.valid,
        "citation_errors": answer.citation_check.errors,
        "sources": [_source_payload(source) for source in answer.sources],
    }


def main() -> None:
    logger.info("Initializing CampusRAG retriever before starting MCP transport")
    _retriever()
    mcp.run()


if __name__ == "__main__":
    main()
