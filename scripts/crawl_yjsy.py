"""爬取华东师范大学研究生院公开网页到 data/raw/yjsy。

用法：
    python scripts/crawl_yjsy.py --max-pages 100
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_settings
from src.crawler import DEFAULT_START_URL, crawl_yjsy


def main() -> None:
    settings = get_settings()
    default_output = os.path.join(settings.data_dir, "yjsy")

    parser = argparse.ArgumentParser(description="爬取华东师范大学研究生院公开网页")
    parser.add_argument(
        "--start-url",
        action="append",
        default=None,
        help="起始 URL，可重复传入；默认从研究生院首页开始",
    )
    parser.add_argument("--output-dir", default=default_output, help="Markdown 输出目录")
    parser.add_argument("--max-pages", type=int, default=80, help="最多访问的 HTML 页面数")
    parser.add_argument("--delay", type=float, default=0.5, help="两次请求之间的等待秒数")
    parser.add_argument("--timeout", type=int, default=10, help="单次请求超时秒数")
    args = parser.parse_args()

    result = crawl_yjsy(
        output_dir=args.output_dir,
        start_urls=args.start_url or [DEFAULT_START_URL],
        max_pages=args.max_pages,
        delay=args.delay,
        timeout=args.timeout,
    )

    print(
        "爬取完成："
        f"访问 {result.visited} 页，保存 {result.saved} 篇，"
        f"跳过 {result.skipped} 页，重复 {result.duplicates} 篇，"
        f"失败 {result.failed} 页。"
    )
    print(f"输出目录：{os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
