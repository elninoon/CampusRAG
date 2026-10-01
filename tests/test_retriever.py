import unittest

from src.retriever import SearchResult, _rrf_fuse, _tokenize


def _result(chunk_id: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        text=chunk_id,
        metadata={},
        score=0.0,
        vector_rank=None,
        bm25_rank=None,
    )


class RetrieverUnitTests(unittest.TestCase):
    def test_tokenize_supports_chinese_and_english(self) -> None:
        self.assertEqual(_tokenize("2026 Scholarship 奖学金"), ["2026", "scholarship", "奖", "学", "金"])

    def test_rrf_rewards_results_found_by_both_retrievers(self) -> None:
        results = _rrf_fuse(
            vector_results=[_result("vector-only"), _result("shared")],
            bm25_results=[_result("shared"), _result("bm25-only")],
            top_k=3,
        )
        self.assertEqual(results[0].chunk_id, "shared")
        self.assertEqual(results[0].vector_rank, 2)
        self.assertEqual(results[0].bm25_rank, 1)


if __name__ == "__main__":
    unittest.main()
