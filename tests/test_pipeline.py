import unittest

from src.citations import validate_citations
from src.generator import GeneratedAnswer
from src.pipeline import RAGPipeline, expand_retrieval_query
from src.retriever import RetrievalTrace, SearchResult


RESULT = SearchResult("chunk", "奖学金材料应按时提交。", {}, 1.0, 1, 1)


class FakeRetriever:
    def search_with_trace(self, query, top_k, filters, vector_k, bm25_k):
        self.query = query
        self.filters = filters
        return RetrievalTrace([RESULT], [RESULT], [RESULT])


class FakeReranker:
    def rerank(self, query, candidates, top_n):
        self.query = query
        return list(candidates)


class FakeGenerator:
    def generate(self, question, results):
        return GeneratedAnswer("请按时提交 [1]", list(results), validate_citations("请按时提交 [1]", 1))


class PipelineTests(unittest.TestCase):
    def test_pipeline_passes_filtered_results_to_generator(self) -> None:
        retriever = FakeRetriever()
        pipeline = RAGPipeline(
            retriever=retriever,
            reranker=FakeReranker(),
            generator=FakeGenerator(),
        )

        answer = pipeline.ask("什么时候提交", filters={"year": 2026})

        self.assertEqual(answer.text, "请按时提交 [1]")
        self.assertEqual(retriever.filters, {"year": 2026})
        self.assertEqual(answer.trace.original_query, "什么时候提交")
        self.assertEqual(answer.trace.reranked_results, [RESULT])

    def test_leadership_question_is_expanded_for_retrieval(self) -> None:
        expanded = expand_retrieval_query("华东师范大学的副校长是谁")

        self.assertIn("学校领导", expanded)
        self.assertIn("完整名单", expanded)

    def test_unrelated_question_is_not_expanded(self) -> None:
        question = "奖学金什么时候提交"

        self.assertEqual(expand_retrieval_query(question), question)


if __name__ == "__main__":
    unittest.main()
