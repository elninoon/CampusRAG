"""阶段 8：检索质量的离线评测。"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Protocol, Sequence

from src.retriever import SearchResult


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_document_id: str
    filters: Dict[str, Any]


@dataclass(frozen=True)
class EvaluationResult:
    case_count: int
    hit_count: int
    hit_at_k: float
    missed_questions: List[str]


class Searcher(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> List[SearchResult]: ...


def load_cases(path: str | Path) -> List[EvaluationCase]:
    """从 JSON 文件读取人工标注的查询、过滤条件与目标文档。"""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("评测集必须是 JSON 数组。")

    cases: List[EvaluationCase] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 条评测样本不是对象。")
        question = item.get("question")
        document_id = item.get("expected_document_id")
        filters = item.get("filters", {})
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"第 {index} 条评测样本缺少 question。")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"第 {index} 条评测样本缺少 expected_document_id。")
        if not isinstance(filters, dict):
            raise ValueError(f"第 {index} 条评测样本的 filters 必须是对象。")
        cases.append(EvaluationCase(question, document_id, filters))
    return cases


def evaluate_retrieval(searcher: Searcher, cases: Sequence[EvaluationCase], top_k: int = 5) -> EvaluationResult:
    """计算 Hit@K：前 K 个结果中是否出现人工标注的目标文档。"""
    if top_k < 1:
        raise ValueError("top_k 必须大于 0。")
    misses: List[str] = []
    for case in cases:
        results = searcher.search(case.question, top_k=top_k, filters=case.filters)
        hit = any(
            result.metadata.get("document_id") == case.expected_document_id
            for result in results
        )
        if not hit:
            misses.append(case.question)
    count = len(cases)
    hits = count - len(misses)
    return EvaluationResult(
        case_count=count,
        hit_count=hits,
        hit_at_k=hits / count if count else 0.0,
        missed_questions=misses,
    )
