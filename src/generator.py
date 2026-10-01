"""阶段 6：基于带来源的检索上下文生成受约束回答。"""
from dataclasses import dataclass
from typing import List, Sequence

from config import LLMConfig
from src.citations import CitationCheck, validate_citations
from src.retriever import SearchResult


SYSTEM_PROMPT = """你是校园信息助手。只可依据提供的资料回答问题。
资料不足以支持回答时，直接说明“提供的资料不足以回答这个问题”，不要猜测。
每一个包含事实的信息后都必须标注对应的来源编号，例如 [1]。
不要编造来源编号，不要把资料中的内容说成最新政策。"""


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    sources: List[SearchResult]
    citation_check: CitationCheck


def build_context(results: Sequence[SearchResult], max_chars: int = 10000) -> str:
    """将精排结果编号后组合为上下文，并限制发送给模型的长度。"""
    if max_chars < 1:
        raise ValueError("max_chars 必须大于 0。")

    parts: List[str] = []
    used = 0
    for number, result in enumerate(results, start=1):
        title = result.metadata.get("title", "未命名文档")
        heading = result.metadata.get("heading", "")
        prefix = f"[来源 {number}] {title} {heading}\n"
        remaining = max_chars - used - len(prefix)
        if remaining <= 0:
            break
        text = result.text[:remaining]
        parts.append(f"{prefix}{text}")
        used += len(prefix) + len(text)
    return "\n\n".join(parts)


class AnswerGenerator:
    """DeepSeek 的 OpenAI 兼容调用封装。"""

    def __init__(self, config: LLMConfig):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "缺少 openai 依赖，请先执行：python -m pip install -r requirements.txt"
            ) from exc
        if not config.api_key:
            raise ValueError("未配置 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 后填写密钥。")
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self._model = config.model

    def generate(self, question: str, results: Sequence[SearchResult]) -> GeneratedAnswer:
        if not question.strip():
            raise ValueError("问题不能为空。")
        if not results:
            text = "提供的资料不足以回答这个问题。"
            return GeneratedAnswer(text, [], validate_citations(text, 0))

        context = build_context(results)
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"资料：\n{context}\n\n问题：{question}",
                },
            ],
        )
        text = response.choices[0].message.content
        if not text or not text.strip():
            raise RuntimeError("LLM 未返回有效回答。")
        text = text.strip()
        return GeneratedAnswer(text, list(results), validate_citations(text, len(results)))
