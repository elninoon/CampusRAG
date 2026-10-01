"""回答引用编号的基础校验。"""
import re
from dataclasses import dataclass
from typing import List


CITATION_RE = re.compile(r"\[(\d+)]")


@dataclass(frozen=True)
class CitationCheck:
    cited_numbers: List[int]
    errors: List[str]

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_citations(text: str, source_count: int) -> CitationCheck:
    """检查回答里的 [n] 是否存在且能映射到展示的来源列表。"""
    if source_count < 0:
        raise ValueError("source_count 不能小于 0。")

    cited_numbers = sorted({int(value) for value in CITATION_RE.findall(text)})
    errors: List[str] = []
    if source_count and not cited_numbers:
        errors.append("回答没有引用任何来源。")
    invalid = [number for number in cited_numbers if not 1 <= number <= source_count]
    if invalid:
        errors.append(f"回答引用了不存在的来源编号: {invalid}")
    if not source_count and cited_numbers:
        errors.append("回答没有可用来源，却包含引用编号。")
    return CitationCheck(cited_numbers=cited_numbers, errors=errors)
