"""最小数据单元：Document。

这是整个系统里流通的"货币"：
- 解析器（parser）产出 Document
- chunker 把一个 Document 切成多个 Document
- 向量库 / BM25 存的都是 chunk 级别的 Document

metadata 是本项目的灵魂 —— 通知的时效性、部门、分类，都靠它支撑
后面的过滤检索和 bad case 修复。
"""
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Document:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
