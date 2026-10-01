import unittest

from src.citations import validate_citations
from src.evaluator import EvaluationCase, evaluate_retrieval
from src.retriever import SearchResult


class FakeSearcher:
    def search(self, query, top_k=5, filters=None):
        return [
            SearchResult(
                chunk_id="chunk",
                text="内容",
                metadata={"document_id": "target.md"},
                score=1.0,
                vector_rank=1,
                bm25_rank=1,
            )
        ]


class EvaluatorAndCitationTests(unittest.TestCase):
    def test_citation_check_rejects_invalid_number(self) -> None:
        check = validate_citations("答案 [1] [3]", source_count=2)
        self.assertFalse(check.valid)
        self.assertIn("[3]", check.errors[0])

    def test_citation_check_requires_source_when_context_exists(self) -> None:
        check = validate_citations("答案", source_count=1)
        self.assertFalse(check.valid)

    def test_evaluate_retrieval_reports_hit_at_k(self) -> None:
        result = evaluate_retrieval(
            FakeSearcher(),
            [EvaluationCase("问题", "target.md", {})],
            top_k=3,
        )
        self.assertEqual(result.hit_at_k, 1.0)
        self.assertEqual(result.missed_questions, [])


if __name__ == "__main__":
    unittest.main()
