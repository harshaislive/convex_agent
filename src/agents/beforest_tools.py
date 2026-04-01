import json
import re
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from langchain_core.tools import tool

from core.settings import settings

EXPERIENCES_BASE_URL = "https://experiences.beforest.co/"
LIVE_SEARCH_SEEDS = [
    "https://beforest.co/",
    "https://experiences.beforest.co/",
    "https://hospitality.beforest.co/",
    "https://10percent.beforest.co/",
]
LIVE_SEARCH_ALLOWED_HOSTS = {
    "beforest.co",
    "experiences.beforest.co",
    "hospitality.beforest.co",
    "10percent.beforest.co",
    "bewild.life",
}
DEFAULT_HTTP_TIMEOUT_SECONDS = 10.0
OUTLINE_PAGE_SIZE = 100
LIVE_SEARCH_CACHE_TTL_SECONDS = 300
LIVE_SEARCH_MAX_PAGES = 24
_OUTLINE_TAG_RE = re.compile(r"<[^>]+>")
_COMMON_QUERY_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "beforest",
    "by",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "our",
    "the",
    "their",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}
_OUTLINE_CACHE: dict[str, object] = {
    "docs": [],
    "chunks": [],
    "fetched_at": 0.0,
    "error": "",
}
_LIVE_SEARCH_CACHE: dict[str, object] = {
    "pages": [],
    "fetched_at": 0.0,
}


def _knowledge_trace_enabled() -> bool:
    return settings.TRACE_KNOWLEDGE_CALLS


def _log_knowledge_trace(message: str) -> None:
    if _knowledge_trace_enabled():
        print(f"[knowledge-trace] {message}", flush=True)


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _normalize_block(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _strip_html_tags(text: str) -> str:
    return _OUTLINE_TAG_RE.sub("", text)


_SEMANTIC_TERM_VARIANTS = {
    "founder": ("founder", "founders", "founded", "founding", "found"),
    "overview": ("overview", "about", "introduction", "summary", "story"),
    "visit": ("visit", "visiting", "book", "booking", "stay", "stays"),
    "hospitality": ("hospitality", "stay", "stays", "accommodation"),
    "experience": ("experience", "experiences", "retreat", "retreats", "workshop", "workshops"),
}
_MONTH_NAME_TO_NUMBER = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_MONTH_DATE_RE = re.compile(
    r"\b("
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?"
    r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?(?:\s+(\d{4}))?\b",
    flags=re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_CURRENT_EXPERIENCE_QUERY_HINTS = (
    "current",
    "currently",
    "live",
    "next",
    "next one",
    "right now",
    "now",
    "today",
    "upcoming",
    "available",
)


def _term_variants(term: str) -> list[str]:
    variants = {term}
    if term in _SEMANTIC_TERM_VARIANTS:
        variants.update(_SEMANTIC_TERM_VARIANTS[term])

    if term.endswith("er") and len(term) > 4:
        stem = term[:-2]
        variants.update({stem, f"{stem}ed", f"{stem}ing"})
    if term.endswith("ed") and len(term) > 4:
        stem = term[:-2]
        variants.update({stem, f"{stem}ing", f"{stem}er"})
    if term.endswith("ing") and len(term) > 5:
        stem = term[:-3]
        variants.update({stem, f"{stem}ed", f"{stem}er"})
    if term.endswith("y") and len(term) > 3:
        variants.add(f"{term[:-1]}ies")
    elif not term.endswith("s") and len(term) > 3:
        variants.add(f"{term}s")

    return sorted((variant for variant in variants if variant), key=lambda item: (-len(item), item))


def _query_terms(query: str) -> list[str]:
    raw_terms = re.findall(r"\b[\w-]+\b", query.lower())
    filtered = [
        term
        for term in raw_terms
        if len(term) > 2 or term.isdigit()
        if term not in _COMMON_QUERY_TERMS
    ]
    expanded: list[str] = []
    for term in filtered or [term for term in raw_terms if term]:
        expanded.extend(_term_variants(term))
    ordered = sorted(dict.fromkeys(expanded), key=lambda term: (-len(term), term))
    return ordered


def _score_text(query: str, text: str) -> int:
    terms = _query_terms(query)
    haystack = text.lower()
    score = 0
    for term in terms:
        matches = haystack.count(term)
        if not matches:
            continue
        weight = 1
        if term in _SEMANTIC_TERM_VARIANTS:
            weight = 3
        elif any(term in variants for variants in _SEMANTIC_TERM_VARIANTS.values()):
            weight = 2
        score += matches * weight
    return score


def _split_text_blocks(text: str) -> list[str]:
    raw_blocks = re.split(r"\n\s*\n", text)
    blocks = [_normalize_block(block) for block in raw_blocks]
    return [block for block in blocks if block]


def _heading_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("#"):
        return stripped.lstrip("#").strip()
    if re.fullmatch(r"\*\*[^*]+\*\*", stripped):
        return stripped.strip("*").strip()
    return ""


def _chunk_section(section_text: str, *, target_chars: int = 1200) -> list[str]:
    blocks = _split_text_blocks(section_text)
    if not blocks:
        return []

    chunks: list[str] = []
    current = ""
    for block in blocks:
        candidate = block if not current else f"{current}\n\n{block}"
        if current and len(candidate) > target_chars:
            chunks.append(current)
            current = block
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _extract_outline_sections(text: str) -> list[str]:
    lines = text.splitlines()
    sections: list[str] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if current_heading and body:
            sections.extend(_chunk_section(f"{current_heading}\n\n{body}"))
        elif body:
            sections.extend(_chunk_section(body))
        current_lines = []

    for line in lines:
        heading = _heading_line(line)
        if heading:
            flush()
            current_heading = heading
            continue
        current_lines.append(line)

    flush()
    return [section for section in sections if section.strip()]


def _best_matching_block(text: str, query: str) -> str:
    blocks = _split_text_blocks(text)
    if not blocks:
        return _normalize_text(text)

    scored_blocks: list[tuple[int, str]] = []
    for block in blocks:
        score = _score_text(query, block)
        if score > 0:
            scored_blocks.append((score, block))

    if not scored_blocks:
        return _normalize_text(text)

    scored_blocks.sort(key=lambda item: item[0], reverse=True)
    return scored_blocks[0][1]


def _build_snippet(text: str, query: str, limit: int = 900) -> str:
    best_block = _best_matching_block(text, query)
    compact = _normalize_text(best_block)
    if len(compact) <= limit:
        return compact

    lower_text = compact.lower()
    for term in _query_terms(query):
        index = lower_text.find(term)
        if index >= 0:
            start = max(index - limit // 3, 0)
            end = min(start + limit, len(compact))
            snippet = compact[start:end].strip()
            return snippet if start == 0 else f"...{snippet}"
    return compact[:limit].strip()


def _is_current_experiences_query(query: str) -> bool:
    lowered = query.lower()
    if "experience" not in lowered and "retreat" not in lowered and "workshop" not in lowered:
        return False
    return any(hint in lowered for hint in _CURRENT_EXPERIENCE_QUERY_HINTS)


def _extract_dates_from_text(text: str, *, today: date) -> list[date]:
    parsed_dates: list[date] = []
    for match in _ISO_DATE_RE.finditer(text):
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            parsed_dates.append(date(year, month, day))
        except ValueError:
            continue

    for match in _MONTH_DATE_RE.finditer(text):
        month_key = match.group(1).lower()
        month = _MONTH_NAME_TO_NUMBER.get(month_key)
        if month is None:
            continue
        day = int(match.group(2))
        raw_year = match.group(3)
        year = int(raw_year) if raw_year else today.year
        try:
            parsed_date = date(year, month, day)
        except ValueError:
            continue
        if raw_year is None and parsed_date < today:
            try:
                parsed_date = date(year + 1, month, day)
            except ValueError:
                continue
        parsed_dates.append(parsed_date)
    return parsed_dates


def _page_has_upcoming_experience_date(page: dict[str, object], *, today: date) -> bool | None:
    page_text = "\n".join(
        [
            str(page.get("title", "")),
            str(page.get("text", "")),
            str(page.get("markdown", "")),
            str(page.get("url", "")),
        ]
    )
    parsed_dates = _extract_dates_from_text(page_text, today=today)
    if not parsed_dates:
        return None
    return any(parsed_date >= today for parsed_date in parsed_dates)


def _outline_request(path: str, payload: dict[str, object]) -> dict[str, object]:
    api_url = settings.OUTLINE_API_URL
    token = settings.OUTLINE_API_TOKEN
    if not api_url or not token:
        raise RuntimeError("Outline URL or token is missing.")

    request = Request(
        f"{str(api_url).rstrip('/')}/api/{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token.get_secret_value()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:  # noqa: S310
            raw = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        try:
            error_body = exc.read().decode("utf-8", errors="replace")
        except OSError:
            error_body = ""
        raise RuntimeError(
            f"Outline API {path} failed with HTTP {exc.code}: {error_body or exc.reason}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Outline API {path} failed with network error: {exc}") from exc

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Outline API {path} returned unexpected payload.")
    return parsed


def _fetch_outline_documents() -> list[dict[str, object]]:
    docs: list[dict[str, object]] = []
    offset = 0
    collection_id = settings.OUTLINE_COLLECTION_ID

    while True:
        payload: dict[str, object] = {"limit": OUTLINE_PAGE_SIZE, "offset": offset}
        if collection_id:
            payload["collectionId"] = collection_id

        parsed = _outline_request("documents.list", payload)
        items = parsed.get("data", [])
        if not isinstance(items, list):
            break

        page_docs = [item for item in items if isinstance(item, dict)]
        docs.extend(page_docs)

        pagination = parsed.get("pagination", {})
        total = pagination.get("total", 0) if isinstance(pagination, dict) else 0
        offset += len(page_docs)
        if not page_docs or not isinstance(total, int) or offset >= total:
            break

    return docs


def _build_outline_chunks(documents: list[dict[str, object]]) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for document in documents:
        title = str(document.get("title", "Untitled Outline document"))
        text = _strip_html_tags(str(document.get("text", "") or ""))
        if not text.strip():
            continue
        sections = _extract_outline_sections(text) or _chunk_section(text)
        for index, section in enumerate(sections):
            chunks.append(
                {
                    "doc_id": str(document.get("id", "")),
                    "title": title,
                    "chunk_index": index,
                    "text": section,
                }
            )
    return chunks


def _load_outline_index() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    now = datetime.now(UTC).timestamp()
    fetched_at = float(_OUTLINE_CACHE.get("fetched_at", 0.0))
    cached_docs = _OUTLINE_CACHE.get("docs", [])
    cached_chunks = _OUTLINE_CACHE.get("chunks", [])

    if (
        now - fetched_at < settings.OUTLINE_CACHE_TTL_SECONDS
        and isinstance(cached_docs, list)
        and isinstance(cached_chunks, list)
        and cached_chunks
    ):
        return (
            [item for item in cached_docs if isinstance(item, dict)],
            [item for item in cached_chunks if isinstance(item, dict)],
        )

    documents = _fetch_outline_documents()
    chunks = _build_outline_chunks(documents)
    _OUTLINE_CACHE["docs"] = documents
    _OUTLINE_CACHE["chunks"] = chunks
    _OUTLINE_CACHE["fetched_at"] = now
    _OUTLINE_CACHE["error"] = ""
    _log_knowledge_trace(f"Outline index refreshed docs={len(documents)} chunks={len(chunks)}")
    return documents, chunks


def _is_allowed_beforest_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and (
        parsed.netloc in LIVE_SEARCH_ALLOWED_HOSTS
        or parsed.netloc.endswith(".beforest.co")
    )


class _MarkdownHTMLParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[str] = []
        self.blocks: list[str] = []
        self.title_parts: list[str] = []
        self._current_parts: list[str] = []
        self._skip_stack: list[str] = []
        self._in_title = False
        self._list_stack: list[str] = []
        self._active_link: str | None = None
        self._active_link_parts: list[str] = []
        self._block_prefix = ""

    def _append_inline(self, text: str) -> None:
        normalized = re.sub(r"\s+", " ", text)
        if not normalized:
            return
        if self._current_parts and not str(self._current_parts[-1]).endswith(
            (" ", "\n", "(", "[", "/")
        ) and not normalized.startswith((".", ",", "!", "?", ":", ";", ")", "]")):
            self._current_parts.append(" ")
        self._current_parts.append(normalized)

    def _flush_block(self) -> None:
        raw = "".join(self._current_parts).strip()
        if raw:
            self.blocks.append(f"{self._block_prefix}{raw}".rstrip())
        self._current_parts = []
        self._block_prefix = ""

    def _begin_block(self, prefix: str = "") -> None:
        self._flush_block()
        self._block_prefix = prefix

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag in {"script", "style", "noscript"}:
            self._skip_stack.append(tag)
            return
        if self._skip_stack:
            return
        if tag == "title":
            self._in_title = True
            return
        if tag == "a":
            href = attr_map.get("href")
            self._active_link = urljoin(self.base_url, href) if href else None
            self._active_link_parts = []
            return
        if tag in {"p", "div", "section", "article", "main", "header", "footer", "aside", "nav"}:
            self._flush_block()
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag[1])
            self._begin_block("#" * level + " ")
            return
        if tag in {"ul", "ol"}:
            self._flush_block()
            self._list_stack.append(tag)
            return
        if tag == "li":
            bullet = "-" if not self._list_stack or self._list_stack[-1] == "ul" else "1."
            indent = "  " * max(len(self._list_stack) - 1, 0)
            self._begin_block(f"{indent}{bullet} ")
            return
        if tag == "br":
            self._flush_block()

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_stack:
            self._skip_stack.pop()
            return
        if self._skip_stack:
            return
        if tag == "title":
            self._in_title = False
            return
        if tag == "a":
            link_text = _normalize_text("".join(self._active_link_parts))
            href = self._active_link
            if link_text:
                if href and href.startswith(("http://", "https://")):
                    self._append_inline(f"[{link_text}]({href})")
                    if _is_allowed_beforest_url(href) and href not in self.links:
                        self.links.append(href)
                else:
                    self._append_inline(link_text)
            elif href and _is_allowed_beforest_url(href) and href not in self.links:
                self.links.append(href)
            self._active_link = None
            self._active_link_parts = []
            return
        if tag in {"p", "div", "section", "article", "main", "header", "footer", "aside", "nav", "li"}:
            self._flush_block()
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush_block()
            return
        if tag in {"ul", "ol"}:
            self._flush_block()
            if self._list_stack:
                self._list_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            return
        if self._in_title:
            self.title_parts.append(data)
        if self._active_link is not None:
            self._active_link_parts.append(data)
            return
        self._append_inline(data)


def _markdown_to_text(markdown: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", markdown)
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    return _normalize_block(text)


def _fetch_beforest_page(url: str) -> dict[str, object]:
    if not _is_allowed_beforest_url(url):
        raise ValueError("URL must be on beforest.co, a beforest.co subdomain, or bewild.life.")
    request = Request(
        url,
        headers={"User-Agent": "BeforestAgent/1.0 (+https://beforest.co)"},
    )
    try:
        with urlopen(request, timeout=DEFAULT_HTTP_TIMEOUT_SECONDS) as response:  # noqa: S310
            charset = response.headers.get_content_charset() or "utf-8"
            html = response.read().decode(charset, errors="replace")
    except HTTPError as exc:
        raise RuntimeError(f"Could not fetch {url}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not fetch {url}: {exc.reason}") from exc

    parser = _MarkdownHTMLParser(url)
    parser.feed(html)
    title = _normalize_text(" ".join(parser.title_parts)) or url
    markdown = "\n\n".join(block for block in parser.blocks if block.strip()).strip()
    if not markdown and title:
        markdown = f"# {title}"
    text = _markdown_to_text(markdown)
    return {
        "url": url,
        "host": urlparse(url).netloc,
        "title": title,
        "markdown": markdown,
        "text": text,
        "links": parser.links,
    }


def _likely_detail_link(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.strip("/").lower()
    if not path:
        return False
    if path in {"about", "contact"}:
        return False
    return len(path.split("/")) >= 1


def _load_live_search_pages() -> list[dict[str, object]]:
    now = datetime.now(UTC).timestamp()
    cached_pages = _LIVE_SEARCH_CACHE.get("pages", [])
    fetched_at = float(_LIVE_SEARCH_CACHE.get("fetched_at", 0.0))
    if (
        now - fetched_at < LIVE_SEARCH_CACHE_TTL_SECONDS
        and isinstance(cached_pages, list)
        and cached_pages
    ):
        return [page for page in cached_pages if isinstance(page, dict)]

    queue = list(LIVE_SEARCH_SEEDS)
    seen: set[str] = set()
    pages: list[dict[str, object]] = []

    while queue and len(pages) < LIVE_SEARCH_MAX_PAGES:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        try:
            page = _fetch_beforest_page(current)
        except (ValueError, RuntimeError):
            continue
        pages.append(page)

        for link in page.get("links", []):
            if not isinstance(link, str):
                continue
            if link in seen or link in queue:
                continue
            if not _is_allowed_beforest_url(link):
                continue
            if len(queue) + len(pages) >= LIVE_SEARCH_MAX_PAGES:
                break
            if _likely_detail_link(link) or link.endswith("/"):
                queue.append(link)

    _LIVE_SEARCH_CACHE["pages"] = pages
    _LIVE_SEARCH_CACHE["fetched_at"] = now
    _log_knowledge_trace(f"Live search cache refreshed pages={len(pages)}")
    return pages


@tool
def search_beforest_knowledge(
    query: str,
    max_results: int = 3,
) -> list[dict[str, str | int]]:
    """Search approved Beforest knowledge from Outline sections."""
    _log_knowledge_trace(f"search_beforest_knowledge query={query!r} max_results={max_results}")
    try:
        _, chunks = _load_outline_index()
    except RuntimeError as exc:
        _OUTLINE_CACHE["error"] = str(exc)
        return []

    terms = _query_terms(query)
    scored_chunks: list[tuple[int, dict[str, object]]] = []
    for chunk in chunks:
        title = str(chunk.get("title", ""))
        text = str(chunk.get("text", ""))
        body_score = _score_text(query, text)
        title_score = _score_text(query, title) * 3
        first_line = text.splitlines()[0].lower() if text.splitlines() else ""
        heading_bonus = sum(8 for term in terms if term in first_line)
        score = body_score + title_score + heading_bonus
        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    results: list[dict[str, str | int]] = []
    for score, chunk in scored_chunks[:max_results]:
        results.append(
            {
                "source": str(chunk.get("title", "Untitled Outline document")),
                "score": score,
                "snippet": _build_snippet(str(chunk.get("text", "")), query),
            }
        )
    _log_knowledge_trace(
        "Outline results: " + ", ".join(str(item.get("source", "")) for item in results)
    )
    return results


@tool
def fetch_beforest_markdown(url: str) -> dict[str, object]:
    """Fetch a Beforest-owned page and return clean markdown for agent grounding."""
    try:
        page = _fetch_beforest_page(url)
    except (ValueError, RuntimeError) as exc:
        return {"url": url, "title": "Unavailable page", "markdown": str(exc), "links": []}
    return {
        "url": str(page["url"]),
        "title": str(page["title"]),
        "markdown": str(page.get("markdown", "")),
        "links": page["links"][:8] if isinstance(page["links"], list) else [],
    }


@tool
def browse_beforest_page(url: str, query: str = "") -> dict[str, object]:
    """Fetch a specific Beforest page for live page-level checks."""
    try:
        page = _fetch_beforest_page(url)
    except (ValueError, RuntimeError) as exc:
        return {
            "url": url,
            "title": "Unavailable page",
            "snippet": str(exc),
            "markdown": str(exc),
            "links": [],
        }
    text = str(page["text"])
    snippet_query = query or text[:80]
    return {
        "url": str(page["url"]),
        "title": str(page["title"]),
        "snippet": _build_snippet(text, snippet_query, limit=500),
        "markdown": str(page.get("markdown", "")),
        "links": page["links"][:8] if isinstance(page["links"], list) else [],
    }

@tool
def search_beforest_live(query: str, max_results: int = 5) -> list[dict[str, str | int]]:
    """Search live Beforest-owned websites for current offerings, listings, and page-level details."""
    _log_knowledge_trace(f"search_beforest_live query={query!r} max_results={max_results}")
    pages = _load_live_search_pages()
    scored_pages: list[tuple[int, dict[str, object]]] = []
    for page in pages:
        title = str(page.get("title", ""))
        text = str(page.get("text", ""))
        url = str(page.get("url", ""))
        host = str(page.get("host", ""))
        score = _score_text(query, title) * 4 + _score_text(query, text)
        if "experiences.beforest.co" in host:
            score += 3
        if "experience" in url.lower() or "retreat" in url.lower():
            score += 2
        if score > 0:
            scored_pages.append((score, page))

    scored_pages.sort(key=lambda item: item[0], reverse=True)
    results: list[dict[str, str | int]] = []
    seen_urls: set[str] = set()
    for score, page in scored_pages:
        url = str(page.get("url", ""))
        if url in seen_urls:
            continue
        seen_urls.add(url)
        results.append(
            {
                "source": str(page.get("title", url or "Untitled Beforest page")),
                "score": score,
                "snippet": _build_snippet(str(page.get("text", "")), query, limit=500),
                "url": url,
            }
        )
        if len(results) >= max_results:
            break

    _log_knowledge_trace(
        "Live search results: " + ", ".join(str(item.get("source", "")) for item in results)
    )
    return results


@tool
def search_beforest_experiences(query: str, max_results: int = 5) -> list[dict[str, str | int]]:
    """Search live Beforest experiences content with stronger crawling and ranking."""
    requires_fresh_dates = _is_current_experiences_query(query)
    today = datetime.now(UTC).date()
    pages = [
        page
        for page in _load_live_search_pages()
        if str(page.get("host", "")) == "experiences.beforest.co"
    ]
    scored_pages: list[tuple[int, dict[str, object]]] = []
    for page in pages:
        title = str(page.get("title", ""))
        text = str(page.get("text", ""))
        url = str(page.get("url", ""))
        score = _score_text(query, title) * 5 + _score_text(query, text)
        if requires_fresh_dates:
            upcoming_status = _page_has_upcoming_experience_date(page, today=today)
            if upcoming_status is False:
                continue
            if upcoming_status is True:
                score += 8
        if any(token in url.lower() for token in ("experience", "retreat", "event", "workshop")):
            score += 2
        if score > 0:
            scored_pages.append((score, page))

    scored_pages.sort(key=lambda item: item[0], reverse=True)
    results: list[dict[str, str | int]] = []
    for score, page in scored_pages[:max_results]:
        results.append(
            {
                "source": str(page.get("title", "Untitled experience page")),
                "score": score,
                "snippet": _build_snippet(str(page.get("text", "")), query, limit=500),
                "url": str(page.get("url", "")),
            }
        )
    if requires_fresh_dates and not results:
        return [
            {
                "source": "Live experiences status",
                "score": 1,
                "snippet": (
                    "No confirmed upcoming dated experiences were found in the latest crawl. "
                    "Route users to experiences.beforest.co for the newest listings instead of "
                    "naming past events as currently live."
                ),
                "url": EXPERIENCES_BASE_URL,
            }
        ]
    _log_knowledge_trace(
        "Experience results: " + ", ".join(str(item.get("source", "")) for item in results)
    )
    return results


def get_knowledge_source_status() -> dict[str, object]:
    outline_configured = bool(settings.OUTLINE_API_URL and settings.OUTLINE_API_TOKEN)
    if not outline_configured:
        return {
            "outlineConfigured": False,
            "outlineResultCount": 0,
            "outlineError": "Missing Outline URL or token.",
        }

    results = search_beforest_knowledge.invoke({"query": "Beforest goals 2040", "max_results": 3})
    return {
        "outlineConfigured": True,
        "outlineApiBase": f"{str(settings.OUTLINE_API_URL).rstrip('/')}/api",
        "outlineCollectionId": settings.OUTLINE_COLLECTION_ID or "",
        "outlineResultCount": len(results),
        "outlineTopSources": [str(item.get("source", "")) for item in results[:3]],
        "outlineDocCount": len([item for item in _OUTLINE_CACHE.get("docs", []) if isinstance(item, dict)]),
        "outlineChunkCount": len([item for item in _OUTLINE_CACHE.get("chunks", []) if isinstance(item, dict)]),
        "outlineError": str(_OUTLINE_CACHE.get("error", "") or ""),
        "convexConfigured": bool(settings.CONVEX_HTTP_ACTION_URL and settings.AGENT_SHARED_SECRET),
    }
