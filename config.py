"""集中配置：所有模型 / 路径都从这里读，改一处全改。

用法：
    from config import get_settings
    s = get_settings()
    print(s.embedding.model)
"""
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str


@dataclass(frozen=True)
class EmbeddingConfig:
    api_key: str
    base_url: str
    model: str
    dim: int  # 向量维度，建库时需要写死


@dataclass(frozen=True)
class RerankerConfig:
    api_key: str
    base_url: str
    model: str


@dataclass(frozen=True)
class Settings:
    llm: LLMConfig
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    data_dir: str
    index_dir: str


def get_settings() -> Settings:
    base = os.path.dirname(os.path.abspath(__file__))
    return Settings(
        llm=LLMConfig(
            api_key=_env("DEEPSEEK_API_KEY"),
            base_url=_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=_env("DEEPSEEK_MODEL", "deepseek-chat"),
        ),
        embedding=EmbeddingConfig(
            api_key=_env("SILICONFLOW_API_KEY"),
            base_url=_env("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
            model=_env("EMBEDDING_MODEL", "BAAI/bge-m3"),
            dim=1024,  # bge-m3 的默认输出维度
        ),
        reranker=RerankerConfig(
            api_key=_env("SILICONFLOW_API_KEY"),
            base_url=_env("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
            model=_env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        ),
        data_dir=os.path.join(base, "data", "raw"),
        index_dir=os.path.join(base, "chroma_db"),
    )
