import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.parser import load_file
from src.web_ingestor import ingest_webpage


class WebIngestorTests(unittest.TestCase):
    def test_ingest_webpage_writes_clean_markdown_with_metadata(self):
        html = """
        <html><head><title>校情简介-华东师范大学</title></head><body>
          <nav>不应进入正文</nav>
          <div class="m2atcRcon">
            <p>华东师范大学成立于1951年。</p>
            <p>这是用于测试的学校简介正文，内容需要足够长，以验证正文识别、来源记录、文本清洗和元数据写入功能。</p>
            <p>学校坚持立德树人，持续推进人才培养、科学研究、社会服务、文化传承创新和国际交流合作。</p>
            <script>ignore()</script>
          </div>
        </body></html>
        """
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "src.web_ingestor._fetch_html",
                return_value=("https://www.ecnu.edu.cn/about.htm", html),
            ):
                result = ingest_webpage(
                    "https://www.ecnu.edu.cn/about.htm",
                    directory,
                    department="华东师范大学",
                    category="学校概况",
                )

            document = load_file(result.markdown_path)
            self.assertEqual(document.metadata["title"], "校情简介")
            self.assertEqual(document.metadata["category"], "学校概况")
            self.assertEqual(document.metadata["department"], "华东师范大学")
            self.assertEqual(
                document.metadata["source_url"],
                "https://www.ecnu.edu.cn/about.htm",
            )
            self.assertIn("成立于1951年", document.text)
            self.assertNotIn("不应进入正文", document.text)
            self.assertNotIn("ignore", document.text)
            self.assertTrue(Path(result.markdown_path).exists())

    def test_ingest_webpage_rejects_directory_page_without_content(self):
        html = "<html><head><title>学校概况</title></head><body><nav>目录</nav></body></html>"
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "src.web_ingestor._fetch_html",
                return_value=("https://example.edu/about.htm", html),
            ):
                with self.assertRaisesRegex(ValueError, "正文容器"):
                    ingest_webpage("https://example.edu/about.htm", directory)


if __name__ == "__main__":
    unittest.main()
