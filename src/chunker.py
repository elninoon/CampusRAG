"""阶段2：Chunking（本项目第一个可对比实验）。

提供两种切分策略：

1. naive_chunk：固定长度 + 重叠 —— 绝大多数玩具 RAG 的做法。
   问题：会把"四、申请时间"和它下面的具体截止日期拆到两个 chunk，
   导致问"什么时候交材料"时召回不到完整答案。

2. structured_chunk：按中文公文层级标题（一、二、三…）切分，
   并把"文档标题"拼进每个 chunk，保证每个 chunk 都自带语境。
   若某一段过长，再按二级标题（（一）（二）…）细分。

后面评测阶段会用 recall 数字证明 structured 为什么更好。
"""
import re
from typing import List, Tuple

from src.schema import Document

# 一级标题：一、 二、 三、 …（兼容 "一." "一．"）
LEVEL1 = re.compile(r"^[一二三四五六七八九十]+[、.．]\s*")
# 二级标题：（一）（二）…
LEVEL2 = re.compile(r"^[（(][一二三四五六七八九十]+[)）]\s*")
SPLIT_MARKS = ("\n", "。", "！", "？", "；", ";")


def _split_sections(text: str) -> List[Tuple[str, str]]:
    """按一级标题把正文切成 [(标题, 正文)]。

    第一个一级标题之前的文字（"各单位：…"这类引言）归入 '引言' 段。
    """
    lines = text.splitlines()
    sections: List[Tuple[str, str]] = []
    cur_heading = "引言"
    cur_body: List[str] = []

    for line in lines:
        stripped = line.strip()
        if LEVEL1.match(stripped):
            sections.append((cur_heading, "\n".join(cur_body).strip()))
            cur_heading = stripped
            cur_body = []
        else:
            cur_body.append(line)

    sections.append((cur_heading, "\n".join(cur_body).strip()))
    # 丢掉空的"引言"段（有些文档没有引言）
    return [(h, b) for h, b in sections if b or h != "引言"]


def _sub_split(heading: str, body: str) -> List[Tuple[str, str]]:
    """段内二级切分：按（一）（二）…进一步切，避免单段过长。"""
    if not LEVEL2.search(body, re.MULTILINE):
        return [(heading, body)]

    lines = body.splitlines()
    parts: List[Tuple[str, str]] = []
    cur_head = heading
    cur_body: List[str] = []
    for line in lines:
        stripped = line.strip()
        if LEVEL2.match(stripped):
            parts.append((cur_head, "\n".join(cur_body).strip()))
            cur_head = f"{heading} > {stripped}"
            cur_body = []
        else:
            cur_body.append(line)
    parts.append((cur_head, "\n".join(cur_body).strip()))
    return [(h, b) for h, b in parts if b]


def _bounded_parts(text: str, max_chars: int, overlap: int = 80) -> List[str]:
    """优先在段落或句末切分，并保证每一段都不超过 max_chars。"""
    if max_chars < 1:
        raise ValueError("max_chars 太小，无法容纳文档标题和章节标题。")
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        end = hard_end
        if hard_end < len(text):
            search_start = start + max_chars // 2
            candidates = [text.rfind(mark, search_start, hard_end) for mark in SPLIT_MARKS]
            boundary = max(candidates)
            if boundary >= search_start:
                end = boundary + 1

        part = text[start:end].strip()
        if part:
            parts.append(part)
        if end >= len(text):
            break
        start = max(end - min(overlap, end - start - 1), start + 1)
    return parts


def structured_chunk(doc: Document, max_chars: int = 1200) -> List[Document]:
    """结构化切分：每个 chunk = 文档标题 + 一级标题 + 正文。"""
    title = doc.metadata.get("title") or doc.text.splitlines()[0].strip()
    sections = _split_sections(doc.text)
    chunks: List[Document] = []

    for heading, body in sections:
        for sub_head, sub_body in _sub_split(heading, body):
            prefix = f"{title}\n\n{sub_head}\n"
            body_limit = max_chars - len(prefix)
            for part_index, part in enumerate(_bounded_parts(sub_body, body_limit)):
                chunks.append(Document(
                    text=f"{prefix}{part}".strip(),
                    metadata={
                        **doc.metadata,
                        "heading": sub_head,
                        "section_chunk_index": part_index,
                    },
                ))

    return chunks


def naive_chunk(doc: Document, chunk_size: int = 500, overlap: int = 50) -> List[Document]:
    """固定长度 + 重叠的 baseline 切分，用于对照实验。"""
    text = doc.text
    chunks: List[Document] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(Document(
            text=text[start:end],
            metadata={**doc.metadata, "chunk_index": len(chunks)},
        ))
        if end == len(text):
            break
        start = end - overlap
    return chunks


CHUNKERS = {
    "structured": structured_chunk,
    "naive": naive_chunk,
}


def chunk_documents(docs: List[Document], strategy: str = "structured", **kw) -> List[Document]:
    """对一批文档统一切分。strategy: 'structured' | 'naive'。"""
    fn = CHUNKERS[strategy]
    out: List[Document] = []
    for d in docs:
        out.extend(fn(d, **kw))
    return out
