"""阶段 3：通过 OpenAI 兼容接口生成文本向量。"""
from typing import List, Sequence

from config import EmbeddingConfig


class EmbeddingClient:
    """对 Embedding API 的轻量封装，负责批处理与基本响应校验。"""

    def __init__(self, config: EmbeddingConfig, batch_size: int = 32):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "缺少 openai 依赖，请先执行：python -m pip install -r requirements.txt"
            ) from exc
        if not config.api_key:
            raise ValueError("未配置 SILICONFLOW_API_KEY，请复制 .env.example 为 .env 后填写密钥。")
        if batch_size < 1:
            raise ValueError("batch_size 必须大于 0。")

        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self._model = config.model
        self._batch_size = batch_size

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """按输入顺序返回文本向量，空字符串不会被静默送入 API。"""
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("待向量化的文本不能是空字符串。")

        vectors: List[List[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start:start + self._batch_size])
            response = self._client.embeddings.create(model=self._model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            if len(ordered) != len(batch):
                raise RuntimeError("Embedding API 返回的向量数量与输入数量不一致。")
            vectors.extend([item.embedding for item in ordered])
        return vectors

    def embed_query(self, query: str) -> List[float]:
        """为单个查询生成向量，供后续检索模块使用。"""
        if not query.strip():
            raise ValueError("查询不能为空。")
        return self.embed_documents([query])[0]
