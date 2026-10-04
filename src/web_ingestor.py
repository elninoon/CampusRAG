"""将单个公开网页清洗为带 YAML frontmatter 的 Markdown。"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml
from bs4 import BeautifulSoup, Tag


USER_AGENT = "CampusRAG/0.1 (+local academic webpage ingestor)"
CONTENT_SELECTORS = (
    ".wp_articlecontent",
    ".m2atcRcon",
    ".article-content",
    ".article_content",
    ".article",
    "article",
    "main",
)


@dataclass(frozen=True)
class WebIngestResult:
    source_url: str
    title: str
    markdown_path: str
    text_length: int


def ingest_webpage(
    url: str,
    output_dir: str,
    *,
    title: str = "",
    department: str = "",
    category: str = "网页资料",
    year: int | None = None,
    selector: str = "",
    timeout: int = 15,
) -> WebIngestResult:
    resolved_url, html = _fetch_html(url, timeout=timeout)
    soup = BeautifulSoup(html, "lxml")
    page_title = title.strip() or _extract_title(soup)
    content = _select_content(soup, selector=selector)
    text = _clean_content(content)
    if len(text) < 80:
        raise ValueError(
            "网页正文过短，可能传入了栏目目录页或页面依赖 JavaScript 渲染。"
            "请传入具体内容页 URL，或用 --selector 指定正文容器。"
        )

    metadata: dict[str, object] = {
        "title": page_title,
        "category": category.strip() or "网页资料",
        "source_url": resolved_url,
        "source_site": urlparse(resolved_url).netloc,
        "retrieved_date": date.today().isoformat(),
    }
    if department.strip():
        metadata["department"] = department.strip()
    if year is not None:
        metadata["year"] = year

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{_safe_filename(page_title, resolved_url)}.md"
    frontmatter = yaml.safe_dump(
        metadata,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).strip()
    path.write_text(f"---\n{frontmatter}\n---\n\n{text}\n", encoding="utf-8")
    return WebIngestResult(
        source_url=resolved_url,
        title=page_title,
        markdown_path=str(path),
        text_length=len(text),
    )


def _fetch_html(url: str, *, timeout: int) -> tuple[str, str]:
    candidates = [url]
    parsed = urlparse(url)
    if not Path(parsed.path).suffix and not parsed.path.endswith("/"):
        candidates.append(f"{url}.htm")

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            request = Request(candidate, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                if "html" not in content_type:
                    raise ValueError(f"URL 返回的不是 HTML: {content_type}")
                raw = response.read()
                resolved_url = response.geturl()
            return resolved_url, raw.decode(_guess_encoding(raw), errors="replace")
        except HTTPError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError(f"无法读取网页: {url}")


def _guess_encoding(raw: bytes) -> str:
    head = raw[:4096].decode("ascii", errors="ignore")
    match = re.search(r"charset=[\"']?([\w-]+)", head, flags=re.I)
    return match.group(1) if match else "utf-8"


def _extract_title(soup: BeautifulSoup) -> str:
    for selector in ("h1", ".arti_title", ".article-title", "title"):
        node = soup.select_one(selector)
        if node:
            title = node.get_text(" ", strip=True)
            if title:
                return re.sub(r"\s*[-|_]\s*华东师范大学\s*$", "", title).strip()
    return "未命名网页"


def _select_content(soup: BeautifulSoup, *, selector: str = "") -> Tag:
    if selector:
        selected = soup.select_one(selector)
        if selected is None:
            raise ValueError(f"未找到指定的正文容器: {selector}")
        return selected
    for candidate in CONTENT_SELECTORS:
        selected = soup.select_one(candidate)
        if selected and len(selected.get_text(" ", strip=True)) >= 80:
            return selected
    raise ValueError(
        "无法自动识别网页正文容器，请检查是否为栏目目录页，或用 --selector 指定 CSS 选择器。"
    )


def _clean_content(content: Tag) -> str:
    for tag in content.select("script, style, iframe, form, nav, aside"):
        tag.decompose()
    text = content.get_text("\n", strip=True)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _safe_filename(title: str, url: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", title).strip("._")[:70]
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{name or 'webpage'}_{digest}"
