"""阶段 5：调用 SiliconFlow Rerank API 精排混合召回结果。"""
import json
from dataclasses import replace
from typing import List, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import RerankerConfig
from src.retriever import SearchResult


class Reranker:
    """使用查询与全文相关性分数，对候选 chunk 再排序。"""

    def __init__(self, config: RerankerConfig, timeout_seconds: int = 30):
        if not config.api_key:
            raise ValueError("未配置 SILICONFLOW_API_KEY，请复制 .env.example 为 .env 后填写密钥。")
        self._api_key = config.api_key
        self._model = config.model
        self._endpoint = f"{config.base_url.rstrip('/')}/rerank"
        self._timeout_seconds = timeout_seconds

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        top_n: int = 3,
    ) -> List[SearchResult]:
        if not query.strip():
            raise ValueError("查询不能为空。")
        if top_n < 1:
            raise ValueError("top_n 必须大于 0。")
        if not candidates:
            return []

        payload = {
            "model": self._model,
            "query": query,
            "documents": [candidate.text for candidate in candidates],
            "top_n": min(top_n, len(candidates)),
            "return_documents": False,
        }
        request = Request(
            self._endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Rerank API 请求失败 ({exc.code}): {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"无法连接 Rerank API: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Rerank API 返回了无法解析的响应。") from exc

        reranked: List[SearchResult] = []
        for item in data.get("results", []):
            index = item.get("index")
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                raise RuntimeError("Rerank API 返回了无效的候选文档索引。")
            score = item.get("relevance_score")
            if not isinstance(score, (int, float)):
                raise RuntimeError("Rerank API 返回了无效的相关性分数。")
            reranked.append(replace(candidates[index], score=float(score)))
        return sorted(reranked, key=lambda result: result.score, reverse=True)
