"""演示脚本：阶段1（解析）+ 阶段2（切分）的效果预览。

不依赖任何 API key，直接运行：
    python scripts/demo.py
"""
import os
import sys

# 让 scripts/ 下的脚本无论从哪个目录跑都能 import 到 src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import load_directory
from src.chunker import structured_chunk, naive_chunk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "raw")


def main():
    docs = load_directory(DATA_DIR)
    print(f"共加载 {len(docs)} 篇文档：\n")
    for d in docs:
        m = d.metadata
        print(f"  · {m.get('title')}")
        print(f"      {m.get('department')} | {m.get('publish_date')} | "
              f"category={m.get('category')} | year={m.get('year')}")

    # 挑 2026 奖学金通知，演示两种切分
    sch = [d for d in docs
           if d.metadata.get("category") == "奖学金" and d.metadata.get("year") == 2026][0]

    print("\n" + "=" * 60)
    print("结构化切分（structured）：每个 chunk 自带标题 + 一级标题\n")
    for c in structured_chunk(sch):
        print(f"  [{c.metadata['heading']}]  长度={len(c.text)}")
        print(f"    开头: {c.text.splitlines()[0]}")

    print("\n" + "=" * 60)
    print("naive 切分（500 字 / 重叠 50）：只看长度，不管结构\n")
    for c in naive_chunk(sch, 500, 50):
        print(f"  [chunk{c.metadata['chunk_index']}] 长度={len(c.text)}  开头: {c.text[:28]!r}")


if __name__ == "__main__":
    main()
