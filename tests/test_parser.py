import tempfile
import unittest
from pathlib import Path

from src.parser import load_directory, load_file


class ParserTests(unittest.TestCase):
    def test_load_file_merges_sidecar_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "handbook.txt"
            source.write_text("研究生手册正文", encoding="utf-8")
            source.with_suffix(".meta.yaml").write_text(
                "title: 2026全日制研究生手册\n"
                "department: 研究生院\n"
                "category: 研究生手册\n"
                "year: 2026\n"
                "document_id: cannot-override.txt\n",
                encoding="utf-8",
            )

            document = load_file(str(source))

            self.assertEqual(document.text, "研究生手册正文")
            self.assertEqual(document.metadata["title"], "2026全日制研究生手册")
            self.assertEqual(document.metadata["year"], 2026)
            self.assertEqual(document.metadata["document_id"], "handbook.txt")
            self.assertEqual(document.metadata["format"], "txt")

    def test_load_directory_does_not_index_sidecar_as_document(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notice.txt").write_text("通知正文", encoding="utf-8")
            (root / "notice.meta.yaml").write_text("category: 通知\n", encoding="utf-8")

            documents = load_directory(directory)

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].metadata["category"], "通知")

    def test_sidecar_must_contain_yaml_object(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "notice.txt"
            source.write_text("通知正文", encoding="utf-8")
            source.with_suffix(".meta.yaml").write_text("- invalid\n- metadata\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "必须是 YAML 对象"):
                load_file(str(source))


if __name__ == "__main__":
    unittest.main()
