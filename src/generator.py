"""阶段 6：基于带来源的检索上下文生成受约束回答。"""
from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence

from config import LLMConfig
from src.citations import CitationCheck, validate_citations
from src.retriever import SearchResult


SYSTEM_PROMPT = """你是校园信息助手。只可依据提供的资料回答问题。
资料不足以支持回答时，直接说明“提供的资料不足以回答这个问题”，不要猜测。
每一个包含事实的信息后都必须标注对应的来源编号，例如 [1]。
不要编造来源编号，不要把资料中的内容说成最新政策。
当问题询问现任人员、完整名单、数量或当前状态时：
1. 优先使用官网名录、专门信息页或明确给出完整集合的资料。
2. 新闻中偶然提到的个别人不能证明完整名单，不可据此回答“只有”该人员。
3. 如果资料互相冲突，说明冲突；如果没有完整名录，明确说明资料不足。"""


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    sources: List[SearchResult]
    citation_check: CitationCheck
    trace: Any | None = None


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

    @staticmethod
    def _history_text(history: Sequence[Mapping[str, str]] | None) -> str:
        """Format a bounded conversation history for reference resolution only."""
        if not history:
            return ""
        turns = []
        for message in history[-8:]:
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            content = content.strip()
            if content:
                turns.append(f"{role}: {content[:1200]}")
        return "\n".join(turns)

    def rewrite_question(
        self,
        question: str,
        history: Sequence[Mapping[str, str]],
    ) -> str:
        """Resolve references in a follow-up without adding facts."""
        history_text = self._history_text(history)
        if not history_text:
            return question.strip()

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你负责把用户的追问改写成可独立检索的问题。结合对话历史解析“它、这个、那项”等指代，"
                        "保留当前问题询问的内容，不要回答问题，不要补充历史中没有的事实。"
                        "如果历史无法明确指代对象，原样返回当前问题。只输出一条问题，不要解释或加引号。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"对话历史：\n{history_text}\n\n"
                        f"当前追问：{question.strip()}\n\n独立问题："
                    ),
                },
            ],
        )
        standalone = response.choices[0].message.content
        if not standalone or not standalone.strip():
            raise RuntimeError("追问改写未返回有效问题，请重试。")
        return standalone.strip().strip('"“”')

    def generate(
        self,
        question: str,
        results: Sequence[SearchResult],
        history: Sequence[Mapping[str, str]] | None = None,
        standalone_question: str | None = None,
    ) -> GeneratedAnswer:
        if not question.strip():
            raise ValueError("问题不能为空。")
        if not results:
            text = "提供的资料不足以回答这个问题。"
            return GeneratedAnswer(text, [], validate_citations(text, 0))

        context = build_context(results)
        history_text = self._history_text(history)
        history_context = (
            f"对话历史（只用于理解当前问题的指代，不作为事实依据）：\n{history_text}\n\n"
            if history_text
            else ""
        )
        resolved_question = standalone_question or question
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"{history_context}资料：\n{context}\n\n"
                        f"用户当前问题：{question}\n"
                        f"用于检索的独立问题：{resolved_question}\n"
                        "请回答用户当前问题。"
                    ),
                },
            ],
        )
        text = response.choices[0].message.content
        if not text or not text.strip():
            raise RuntimeError("LLM 未返回有效回答。")
        text = text.strip()
        return GeneratedAnswer(text, list(results), validate_citations(text, len(results)))
