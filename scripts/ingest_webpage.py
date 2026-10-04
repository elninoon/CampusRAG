"""抓取单个公开网页并保存为可直接建库的 Markdown。"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_settings
from src.web_ingestor import ingest_webpage


def main() -> None:
    default_output = os.path.join(get_settings().data_dir, "web")
    parser = argparse.ArgumentParser(description="提取公开网页正文并保存为 Markdown")
    parser.add_argument("url", help="具体内容页 URL，不建议传栏目目录页")
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--title", default="", help="覆盖网页标题")
    parser.add_argument("--department", default="")
    parser.add_argument("--category", default="网页资料")
    parser.add_argument("--year", type=int)
    parser.add_argument("--selector", default="", help="正文容器 CSS 选择器")
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()

    try:
        result = ingest_webpage(
            args.url,
            args.output_dir,
            title=args.title,
            department=args.department,
            category=args.category,
            year=args.year,
            selector=args.selector,
            timeout=args.timeout,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"网页提取完成：{result.title}")
    print(f"正文字符数：{result.text_length}")
    print(f"来源：{result.source_url}")
    print(f"Markdown：{os.path.abspath(result.markdown_path)}")


if __name__ == "__main__":
    main()
