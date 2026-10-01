"""阶段 4：向量检索与 BM25 检索的混合召回。"""
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from config import Settings, get_settings
from src.embeddings import EmbeddingClient
from src.indexer import COLLECTION_NAME


@dataclass(frozen=True)
class SearchResult:
    """一条可交给重排器或回答生成器处理的检索结果。"""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    score: float
    vector_rank: int | None
    bm25_rank: int | None


def _tokenize(text: str) -> List[str]:
    """为中英文混合校园文本生成轻量 BM25 词项。

    中文按单字匹配，英文和数字按连续词匹配。它无需额外分词模型，适合
    小规模本地知识库；语义相近但字面不同的问题由向量召回来补足。
    """
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def _matches_filters(metadata: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    return all(metadata.get(key) == value for key, value in filters.items())


def _rrf_fuse(
    vector_results: Sequence[SearchResult],
    bm25_results: Sequence[SearchResult],
    top_k: int,
    rrf_k: int = 60,
) -> List[SearchResult]:
    """使用 Reciprocal Rank Fusion 合并两份排序，避免比较不同分数尺度。"""
    if top_k < 1:
        raise ValueError("top_k 必须大于 0。")

    merged: Dict[str, SearchResult] = {}
    scores: Dict[str, float] = {}
    vector_ranks: Dict[str, int] = {}
    bm25_ranks: Dict[str, int] = {}

    for rank, result in enumerate(vector_results, start=1):
        merged[result.chunk_id] = result
        vector_ranks[result.chunk_id] = rank
        scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    for rank, result in enumerate(bm25_results, start=1):
        merged.setdefault(result.chunk_id, result)
        bm25_ranks[result.chunk_id] = rank
        scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    fused = [
        SearchResult(
            chunk_id=chunk_id,
            text=result.text,
            metadata=result.metadata,
            score=scores[chunk_id],
            vector_rank=vector_ranks.get(chunk_id),
            bm25_rank=bm25_ranks.get(chunk_id),
        )
        for chunk_id, result in merged.items()
    ]
    return sorted(fused, key=lambda item: item.score, reverse=True)[:top_k]


class HybridRetriever:
    """针对已建立的 Chroma 索引提供混合检索。"""

    def __init__(self, settings: Settings | None = None):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "缺少 chromadb 依赖，请先执行：python -m pip install -r requirements.txt"
            ) from exc
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "缺少 rank-bm25 依赖，请先执行：python -m pip install -r requirements.txt"
            ) from exc

        self.settings = settings or get_settings()
        self.embedder = EmbeddingClient(self.settings.embedding)
        client = chromadb.PersistentClient(path=self.settings.index_dir)
        existing = {collection.name for collection in client.list_collections()}
        if COLLECTION_NAME not in existing:
            raise RuntimeError("未找到本地索引，请先运行：python scripts/build_index.py")
        self.collection = client.get_collection(COLLECTION_NAME)

        raw = self.collection.get(include=["documents", "metadatas"])
        self._records = [
            SearchResult(
                chunk_id=chunk_id,
                text=document,
                metadata=metadata or {},
                score=0.0,
                vector_rank=None,
                bm25_rank=None,
            )
            for chunk_id, document, metadata in zip(
                raw["ids"], raw["documents"], raw["metadatas"]
            )
        ]
        if not self._records:
            raise RuntimeError("本地索引为空，请先运行：python scripts/build_index.py --reset")
        self._bm25 = BM25Okapi([_tokenize(record.text) for record in self._records])

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
        vector_k: int = 20,
        bm25_k: int = 20,
    ) -> List[SearchResult]:
        """执行两路召回并返回 RRF 融合后的结果。"""
        if not query.strip():
            raise ValueError("查询不能为空。")
        if top_k < 1 or vector_k < 1 or bm25_k < 1:
            raise ValueError("top_k、vector_k 和 bm25_k 必须大于 0。")
        active_filters = dict(filters or {})
        vector_results = self._vector_search(query, vector_k, active_filters)
        bm25_results = self._bm25_search(query, bm25_k, active_filters)
        return _rrf_fuse(vector_results, bm25_results, top_k)

    def _vector_search(
        self,
        query: str,
        limit: int,
        filters: Mapping[str, Any],
    ) -> List[SearchResult]:
        count = self.collection.count()
        if count == 0:
            return []
        response = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(limit, count),
            where=dict(filters) or None,
            include=["documents", "metadatas"],
        )
        return [
            SearchResult(
                chunk_id=chunk_id,
                text=document,
                metadata=metadata or {},
                score=0.0,
                vector_rank=rank,
                bm25_rank=None,
            )
            for rank, (chunk_id, document, metadata) in enumerate(
                zip(response["ids"][0], response["documents"][0], response["metadatas"][0]),
                start=1,
            )
        ]

    def _bm25_search(
        self,
        query: str,
        limit: int,
        filters: Mapping[str, Any],
    ) -> List[SearchResult]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        results: List[SearchResult] = []
        for index in ranked:
            if scores[index] <= 0:
                break
            record = self._records[index]
            if _matches_filters(record.metadata, filters):
                results.append(record)
            if len(results) == limit:
                break
        return results
