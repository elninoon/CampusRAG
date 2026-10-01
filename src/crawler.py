"""华东师范大学研究生院网站爬虫。

爬取公开网页内容，清洗成带 YAML frontmatter 的 Markdown 文件，
直接落到 data/raw 下供现有 parser/indexer 使用。
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from html import unescape
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


DEFAULT_START_URL = "https://yjsy.ecnu.edu.cn/"
ALLOWED_HOST = "yjsy.ecnu.edu.cn"
USER_AGENT = "CampusRAG/0.1 (+local academic crawler)"
LOGIN_TITLES = ("统一身份认证平台", "统一身份认证")


@dataclass(frozen=True)
class CrawledPage:
    url: str
    title: str
    text: str
    metadata: dict


@dataclass
class CrawlResult:
    saved: int = 0
    visited: int = 0
    skipped: int = 0
    duplicates: int = 0
    failed: int = 0
    output_files: list[str] = field(default_factory=list)


def crawl_yjsy(
    output_dir: str,
    start_urls: Iterable[str] | None = None,
    *,
    max_pages: int = 80,
    delay: float = 0.5,
    timeout: int = 10,
) -> CrawlResult:
    """爬取研究生院公开页面并保存为 Markdown。

    只跟随 yjsy.ecnu.edu.cn 域名下的 HTML 页面。PDF 等附件会保留在正文链接中，
    但不会下载，避免把二进制文件混入当前 Markdown 资料流。
    """
    os.makedirs(output_dir, exist_ok=True)
    queue = list(start_urls or [DEFAULT_START_URL])
    seen: set[str] = set()
    seen_pages: set[str] = set()
    result = CrawlResult()

    while queue and result.visited < max_pages:
        url = _normalize_url(queue.pop(0))
        if not url or url in seen or not _is_allowed_url(url):
            continue
        seen.add(url)

        try:
            html = _fetch_html(url, timeout=timeout)
        except (HTTPError, URLError, TimeoutError, UnicodeDecodeError, OSError, ValueError):
            result.failed += 1
            continue

        result.visited += 1
        soup = BeautifulSoup(html, "lxml")
        for link in _extract_links(soup, url):
            if link not in seen and link not in queue:
                queue.append(link)

        page = _extract_page(soup, url)
        if page is None:
            result.skipped += 1
        elif _page_signature(page) in seen_pages:
            result.duplicates += 1
        else:
            seen_pages.add(_page_signature(page))
            path = _write_markdown(page, output_dir)
            result.saved += 1
            result.output_files.append(path)

        if delay > 0:
            time.sleep(delay)

    return result


def _fetch_html(url: str, *, timeout: int) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            raise ValueError(f"不是 HTML 页面: {content_type}")
        raw = resp.read()
    return raw.decode(_guess_encoding(raw), errors="ignore")


def _guess_encoding(raw: bytes) -> str:
    head = raw[:2048].decode("ascii", errors="ignore")
    match = re.search(r"charset=[\"']?([\w-]+)", head, flags=re.I)
    return match.group(1) if match else "utf-8"


def _extract_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    links: list[str] = []
    for tag in soup.select("a[href]"):
        href = tag.get("href", "").strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:")):
            continue
        url = _normalize_url(urljoin(base_url, href))
        if url and _is_allowed_url(url):
            links.append(url)
    return sorted(dict.fromkeys(links), key=_crawl_priority)


def _crawl_priority(url: str) -> tuple[int, str]:
    path = urlparse(url).path.lower()
    if path.endswith("/page.htm") or "/page." in path:
        return (0, url)
    if path.endswith("/list.htm") or re.search(r"/list\d+\.htm$", path):
        return (1, url)
    return (2, url)


def _extract_page(soup: BeautifulSoup, url: str) -> CrawledPage | None:
    title = _clean_text(_first_text(soup, [".arti_title", "h1", "title"]))
    if not title or any(marker in title for marker in LOGIN_TITLES):
        return None

    content = soup.select_one(".wp_articlecontent")
    if content is None:
        content = soup.select_one(".article")
    if content is None:
        return None

    for tag in content.select("script, style, iframe, form"):
        tag.decompose()

    text = _clean_text(content.get_text("\n", strip=True))
    if len(text) < 80:
        return None

    publish_date = _extract_publish_date(soup.get_text(" ", strip=True))
    year = int(publish_date[:4]) if publish_date else _extract_year(title + " " + text)
    category = _guess_category(title, url)

    metadata = {
        "title": title,
        "department": "研究生院",
        "category": category,
        "source_url": url,
        "source_site": "华东师范大学研究生院",
    }
    if publish_date:
        metadata["publish_date"] = publish_date
    if year:
        metadata["year"] = year

    return CrawledPage(url=url, title=title, text=text, metadata=metadata)


def _first_text(soup: BeautifulSoup, selectors: Iterable[str]) -> str:
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            text = node.get_text(" ", strip=True)
            if text:
                return text
    return ""


def _extract_publish_date(text: str) -> str:
    match = re.search(r"发布时间[:：]\s*(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})", text)
    if not match:
        match = re.search(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?", text)
    if not match:
        return ""
    year, month, day = (int(part) for part in match.groups())
    return f"{year:04d}-{month:02d}-{day:02d}"


def _extract_year(text: str) -> int | None:
    match = re.search(r"(20\d{2})", text)
    return int(match.group(1)) if match else None


def _guess_category(title: str, url: str) -> str:
    rules = [
        ("国际交流", ("国际交流", "出国", "境外", "留学", "联合培养")),
        ("学籍管理", ("学籍", "新生报到", "结业", "肄业")),
        ("培养", ("培养", "课程", "调停课", "教学", "专业")),
        ("学位", ("学位", "答辩", "论文", "学位申请")),
        ("规章制度", ("规定", "办法", "章程", "细则", "制度")),
        ("招生", ("招生", "推免", "博士", "硕士")),
    ]
    for category, keywords in rules:
        if any(keyword in title for keyword in keywords):
            return category
    if "42087" in url:
        return "规章制度"
    return "通知公告"


def _write_markdown(page: CrawledPage, output_dir: str) -> str:
    filename = _safe_filename(page.title, page.url) + ".md"
    path = os.path.join(output_dir, filename)
    frontmatter = "\n".join(f"{key}: {_yaml_scalar(value)}" for key, value in page.metadata.items())
    body = f"---\n{frontmatter}\n---\n\n{page.text}\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    return path


def _page_signature(page: CrawledPage) -> str:
    digest = hashlib.sha1(page.text.encode("utf-8")).hexdigest()
    return f"{page.title}|{digest}"


def _yaml_scalar(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _safe_filename(title: str, url: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", title).strip("._")
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    if len(name) > 70:
        name = name[:70].rstrip("._")
    return f"{name}_{digest}"


def _normalize_url(url: str) -> str:
    url = urldefrag(unescape(url.strip()))[0]
    parsed = urlparse(url)
    if parsed.scheme == "http" and parsed.netloc == ALLOWED_HOST:
        parsed = parsed._replace(scheme="https")
    return parsed.geturl()


def _is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.netloc != ALLOWED_HOST:
        return False
    path = parsed.path.lower()
    if path.endswith((".pdf", ".doc", ".docx", ".xls", ".xlsx", ".zip", ".rar", ".jpg", ".png")):
        return False
    return path in {"", "/"} or path.endswith((".htm", ".html"))


def _clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
