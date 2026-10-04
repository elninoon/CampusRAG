"""阶段 3：把切分后的文档写入本地 Chroma 向量库。"""
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from config import Settings, get_settings
from src.chunker import chunk_documents
from src.embeddings import EmbeddingClient
from src.parser import load_directory
from src.schema import Document


COLLECTION_NAME = "campus_documents"
INDEX_VERSION_FILE = ".index_version"


@dataclass(frozen=True)
class IndexResult:
    document_count: int
    chunk_count: int
    collection_count: int


def _chunk_id(chunk: Document) -> str:
    """用来源和内容生成可复现 ID，支持重复执行时更新同一 chunk。"""
    source = str(chunk.metadata.get("source_path", ""))
    heading = str(chunk.metadata.get("heading", ""))
    payload = f"{source}\n{heading}\n{chunk.text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _chroma_metadata(metadata: Dict[str, Any]) -> Dict[str, str | int | float | bool]:
    """将 frontmatter 的日期、列表等值转换为 Chroma 可持久化的标量。"""
    cleaned: Dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, (date, datetime)):
            cleaned[key] = value.isoformat()
        else:
            cleaned[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return cleaned


def _batches(items: List[Document], size: int) -> Iterable[List[Document]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


class IndexBuilder:
    """协调加载、切分、向量化和 Chroma 持久化。"""

    def __init__(self, settings: Settings | None = None, batch_size: int = 32):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "缺少 chromadb 依赖，请先执行：python -m pip install -r requirements.txt"
            ) from exc
        self.settings = settings or get_settings()
        self.batch_size = batch_size
        self.embedder = EmbeddingClient(self.settings.embedding, batch_size=batch_size)
        self.client = chromadb.PersistentClient(path=self.settings.index_dir)

    def build(self, strategy: str = "structured", reset: bool = False) -> IndexResult:
        docs = load_directory(self.settings.data_dir)
        chunks = chunk_documents(docs, strategy=strategy)
        if not chunks:
            raise RuntimeError(f"数据目录中没有可索引的内容: {self.settings.data_dir}")

        if reset:
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except ValueError:
                pass
        collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        for batch in _batches(chunks, self.batch_size):
            vectors = self.embedder.embed_documents([chunk.text for chunk in batch])
            ids = [_chunk_id(chunk) for chunk in batch]
            collection.upsert(
                ids=ids,
                documents=[chunk.text for chunk in batch],
                metadatas=[_chroma_metadata(chunk.metadata) for chunk in batch],
                embeddings=vectors,
            )

        version_path = Path(self.settings.index_dir) / INDEX_VERSION_FILE
        version_path.write_text(str(time.time_ns()), encoding="ascii")

        return IndexResult(
            document_count=len(docs),
            chunk_count=len(chunks),
            collection_count=collection.count(),
        )
