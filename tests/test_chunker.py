import unittest

from src.chunker import structured_chunk
from src.schema import Document


class ChunkerTests(unittest.TestCase):
    def test_structured_chunk_enforces_max_chars_without_subheadings(self):
        body = "这是一段很长的规定。" * 200
        document = Document(
            text=f"一、适用范围\n{body}",
            metadata={"title": "研究生手册", "document_id": "handbook.pdf"},
        )

        chunks = structured_chunk(document, max_chars=200)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 200 for chunk in chunks))
        self.assertTrue(all(chunk.metadata["heading"] == "一、适用范围" for chunk in chunks))
        self.assertEqual(
            [chunk.metadata["section_chunk_index"] for chunk in chunks],
            list(range(len(chunks))),
        )

    def test_structured_chunk_preserves_short_section(self):
        document = Document(
            text="一、提交要求\n请在规定日期前提交材料。",
            metadata={"title": "奖学金通知"},
        )

        chunks = structured_chunk(document, max_chars=200)

        self.assertEqual(len(chunks), 1)
        self.assertIn("请在规定日期前提交材料。", chunks[0].text)


if __name__ == "__main__":
    unittest.main()
