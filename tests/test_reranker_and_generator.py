import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from config import LLMConfig, RerankerConfig
from src.generator import build_context
from src.reranker import Reranker
from src.retriever import SearchResult


def _result(chunk_id: str, text: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        text=text,
        metadata={"title": "测试通知", "heading": "一、测试"},
        score=0.0,
        vector_rank=None,
        bm25_rank=None,
    )


class RerankerAndGeneratorTests(unittest.TestCase):
    def test_build_context_numbers_sources(self) -> None:
        context = build_context([_result("a", "甲"), _result("b", "乙")])
        self.assertIn("[来源 1]", context)
        self.assertIn("[来源 2]", context)

    @patch("src.reranker.urlopen")
    def test_reranker_uses_api_indices_and_scores(self, mock_urlopen) -> None:
        response = SimpleNamespace(read=lambda: json.dumps({
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.4},
            ]
        }).encode("utf-8"))
        mock_urlopen.return_value.__enter__.return_value = response
        reranker = Reranker(RerankerConfig("key", "https://example.com/v1", "model"))

        results = reranker.rerank("问题", [_result("a", "甲"), _result("b", "乙")])

        self.assertEqual([item.chunk_id for item in results], ["b", "a"])
        self.assertEqual(results[0].score, 0.9)


if __name__ == "__main__":
    unittest.main()
