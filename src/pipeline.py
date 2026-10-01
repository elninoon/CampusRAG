"""阶段 7：串联混合检索、精排和受约束回答生成。"""
from typing import Any, Mapping

from config import Settings, get_settings
from src.generator import AnswerGenerator, GeneratedAnswer
from src.reranker import Reranker
from src.retriever import HybridRetriever


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
        candidates = self.retriever.search(
            question,
            top_k=retrieval_k,
            filters=filters,
            vector_k=retrieval_k,
            bm25_k=retrieval_k,
        )
        ranked = self.reranker.rerank(question, candidates, top_n=rerank_k)
        return self.generator.generate(question, ranked)
