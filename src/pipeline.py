"""阶段 7：串联混合检索、精排和受约束回答生成。"""
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping

from config import Settings, get_settings
from src.generator import AnswerGenerator, GeneratedAnswer
from src.reranker import Reranker
from src.retriever import HybridRetriever, SearchResult


LEADERSHIP_TERMS = ("副校长", "校长", "书记", "学校领导", "领导班子")


@dataclass(frozen=True)
class QueryTrace:
    original_query: str
    retrieval_query: str
    filters: Dict[str, Any]
    vector_results: List[SearchResult]
    bm25_results: List[SearchResult]
    fused_results: List[SearchResult]
    reranked_results: List[SearchResult]


def expand_retrieval_query(question: str) -> str:
    """为需要权威名录的领导类问题补充检索语义，不改变最终用户问题。"""
    if any(term in question for term in LEADERSHIP_TERMS):
        return f"{question} 学校领导 现任 完整名单"
    return question


class RAGPipeline:
    """CampusRAG 的单问题问答入口。"""

    def __init__(
        self,
        settings: Settings | None = None,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        generator: AnswerGenerator | None = None,
    ):
        settings = settings or get_settings()
        self.retriever = retriever or HybridRetriever(settings)
        self.reranker = reranker or Reranker(settings.reranker)
        self.generator = generator or AnswerGenerator(settings.llm)

    def ask(
        self,
        question: str,
        filters: Mapping[str, Any] | None = None,
        retrieval_k: int = 12,
        rerank_k: int = 5,
    ) -> GeneratedAnswer:
        retrieval_query = expand_retrieval_query(question)
        retrieval_trace = self.retriever.search_with_trace(
            retrieval_query,
            top_k=retrieval_k,
            filters=filters,
            vector_k=retrieval_k,
            bm25_k=retrieval_k,
        )
        ranked = self.reranker.rerank(
            retrieval_query,
            retrieval_trace.fused_results,
            top_n=rerank_k,
        )
        answer = self.generator.generate(question, ranked)
        trace = QueryTrace(
            original_query=question,
            retrieval_query=retrieval_query,
            filters=dict(filters or {}),
            vector_results=retrieval_trace.vector_results,
            bm25_results=retrieval_trace.bm25_results,
            fused_results=retrieval_trace.fused_results,
            reranked_results=ranked,
        )
        return replace(answer, trace=trace)
