import json
import os
import re
import smtplib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlparse
from urllib.request import Request, urlopen

from langchain_core.tools import tool

EXAMPLE_DIR = Path(__file__).parent.resolve()
KNOWLEDGE_DIR = EXAMPLE_DIR / "knowledge"
KNOWLEDGE_CENTER_DOCS_DIR = EXAMPLE_DIR / "knowledge_center" / "docs"
DATA_DIR = EXAMPLE_DIR / "data"
LEADS_PATH = DATA_DIR / "leads.jsonl"
ESCALATIONS_PATH = DATA_DIR / "escalations.jsonl"
NARRATIVE_PATH = EXAMPLE_DIR / "BEFOREST LIFESTYLE SOLUTIONS- NARRATIVE 2025.docx.md"
EXPERIENCES_BASE_URL = "https://experiences.beforest.co/"
MAX_EXPERIENCE_PAGES = 3
DEFAULT_HTTP_TIMEOUT_SECONDS = 3.0
EXPERIENCE_CACHE_TTL_SECONDS = 300
_EXPERIENCE_CACHE: dict[str, object] = {"pages": [], "fetched_at": 0.0}
_EXPERIENCE_PAGE_CACHE: dict[str, object] = {}
_OUTLINE_TAG_RE = re.compile(r"<[^>]+>")


def _score_text(query: str, text: str) -> int:
    terms = [term for term in query.lower().split() if term]
    haystack = text.lower()
    return sum(haystack.count(term) for term in terms)


def _normalize_text(text: str) -> str:
    """Collapse repeated whitespace for cleaner snippets."""
    return " ".join(text.split())


def _strip_html_tags(text: str) -> str:
    """Remove simple inline HTML markers returned by Outline search."""
    return _OUTLINE_TAG_RE.sub("", text)


def _knowledge_debug_enabled() -> bool:
    return os.getenv("DEBUG_KNOWLEDGE_ERRORS", "").strip().lower() == "true"


def _log_knowledge_error(message: str) -> None:
    if _knowledge_debug_enabled():
        print(f"[knowledge] {message}", flush=True)


def _get_outline_api_base() -> str | None:
    """Return the Outline API base URL when configured."""
    base = (
        os.getenv("OUTLINE_API_URL", "").strip()
        or os.getenv("OUTLINE_BASE_URL", "").strip()
        or os.getenv("OUTLINE_URL", "").strip()
    )
    if not base:
        return None
    return base.rstrip("/") + "/api"


def _get_outline_token() -> str:
    return (
        os.getenv("OUTLINE_API_TOKEN", "").strip()
        or os.getenv("OUTLINE_TOKEN", "").strip()
        or os.getenv("OUTLINE_API_KEY", "").strip()
    )


def _get_outline_collection_id() -> str:
    return os.getenv("OUTLINE_COLLECTION_ID", "").strip() or os.getenv(
        "OUTLINE_COLLECTION", ""
    ).strip()


def _load_outline_knowledge_results(
    query: str,
    *,
    max_results: int,
) -> list[dict[str, str | int]]:
    """Search Outline documents and return grounded snippets."""
    api_base = _get_outline_api_base()
    token = _get_outline_token()
    if not api_base or not token:
        _log_knowledge_error("Outline search skipped because URL or token is missing.")
        return []

    payload = {"query": query, "limit": max_results}
    collection_id = _get_outline_collection_id()
    if collection_id:
        payload["collectionId"] = collection_id

    request = Request(
        f"{api_base}/documents.search",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=10) as response:  # noqa: S310
            raw = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        try:
            error_body = exc.read().decode("utf-8", errors="replace")
        except OSError:
            error_body = exc.reason if hasattr(exc, "reason") else ""
        _log_knowledge_error(
            f"Outline search failed with HTTP {exc.code}: {error_body or exc.reason}"
        )
        return []
    except OSError as exc:
        _log_knowledge_error(f"Outline search failed with network error: {exc}")
        return []

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        _log_knowledge_error(f"Outline search returned non-JSON: {raw[:400]}")
        return []

    items = parsed.get("data", [])
    if not isinstance(items, list):
        _log_knowledge_error(f"Outline search returned unexpected payload: {raw[:400]}")
        return []

    results: list[dict[str, str | int]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        document = item.get("document", {})
        if not isinstance(document, dict):
            document = {}
        title = str(document.get("title", "Untitled Outline document"))
        context = _normalize_text(_strip_html_tags(str(item.get("context", "")).strip()))
        body = _normalize_text(str(document.get("text", "")).strip())
        snippet_source = context or body
        if not snippet_source:
            continue
        ranking = item.get("ranking", 0)
        try:
            score = int(float(ranking) * 1000)
        except (TypeError, ValueError):
            score = 0
        results.append(
            {
                "source": title,
                "score": score,
                "snippet": _build_snippet(snippet_source, query),
            }
        )

    return results


def get_knowledge_source_status() -> dict[str, object]:
    """Return non-secret knowledge source diagnostics for deployment checks."""
    outline_api_base = _get_outline_api_base()
    outline_token = _get_outline_token()
    collection_id = _get_outline_collection_id()
    status: dict[str, object] = {
        "outlineConfigured": bool(outline_api_base and outline_token),
        "outlineApiBase": outline_api_base or "",
        "outlineCollectionId": collection_id or "",
        "convexConfigured": bool(_get_convex_base() and os.getenv("AGENT_SHARED_SECRET", "").strip()),
    }

    if not outline_api_base or not outline_token:
        status["outlineReachable"] = False
        status["outlineResultCount"] = 0
        status["outlineError"] = "Missing Outline URL or token."
        return status

    probe_results = _load_outline_knowledge_results("Beforest", max_results=3)
    status["outlineReachable"] = True
    status["outlineResultCount"] = len(probe_results)
    status["outlineTopSources"] = [
        str(item.get("source", "")) for item in probe_results[:3]
    ]
    if not probe_results:
        status["outlineWarning"] = "Outline is configured but returned no search results."
    return status


def _get_convex_base() -> str | None:
    """Derive the Convex deployment base URL from the configured DM event endpoint."""
    url = os.getenv("CONVEX_HTTP_ACTION_URL", "").strip()
    if url:
        return url.replace("/instagram/store-dm-event", "")

    site_url = os.getenv("CONVEX_SITE_URL", "").strip()
    if site_url:
        return site_url.rstrip("/")

    return None


def _load_convex_knowledge_results(
    query: str,
    *,
    max_results: int,
    intent: str = "",
    audience: str = "",
) -> list[dict[str, str | int]]:
    """Search Convex-backed knowledge entries when configured."""
    base = _get_convex_base()
    secret = os.getenv("AGENT_SHARED_SECRET", "").strip()
    if not base or not secret:
        return []

    encoded_query = quote(query, safe="")
    url = f"{base}/knowledge/search?query={encoded_query}&maxResults={max_results}"
    if intent.strip():
        url += f"&intent={quote(intent.strip(), safe='')}"
    if audience.strip():
        url += f"&audience={quote(audience.strip(), safe='')}"

    request = Request(
        url,
        headers={"x-agent-secret": secret},
        method="GET",
    )
    try:
        with urlopen(request, timeout=10) as response:  # noqa: S310
            raw = response.read().decode("utf-8", errors="replace")
    except (HTTPError, OSError):
        return []

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    results: list[dict[str, str | int]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        body = str(item.get("body", ""))
        summary = str(item.get("summary", "") or "")
        snippet_source = summary or body
        results.append(
            {
                "source": str(item.get("title", "Untitled knowledge entry")),
                "score": int(item.get("score", 0)),
                "snippet": _build_snippet(snippet_source or body, query),
            }
        )
    return results


def _build_snippet(text: str, query: str, limit: int = 360) -> str:
    """Extract a short snippet around the first matched term when possible."""
    compact = _normalize_text(text)
    if len(compact) <= limit:
        return compact

    lower_text = compact.lower()
    for term in [item for item in query.lower().split() if item]:
        index = lower_text.find(term)
        if index >= 0:
            start = max(index - limit // 3, 0)
            end = min(start + limit, len(compact))
            snippet = compact[start:end].strip()
            return snippet if start == 0 else f"...{snippet}"

    return compact[:limit].strip()


class _ExperienceHTMLParser(HTMLParser):
    """Parse HTML into visible text, title, and same-page links."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip_stack: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag in {"script", "style", "noscript"}:
            self._skip_stack.append(tag)
            return
        if tag == "title":
            self._in_title = True
            return
        if tag == "a":
            href = attr_map.get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_stack:
            self._skip_stack.pop()
            return
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            return
        if self._in_title:
            self.title_parts.append(data)
        self.parts.append(data)


@dataclass(slots=True)
class _ExperiencePage:
    url: str
    title: str
    text: str
    links: list[str] = field(default_factory=list)


def _is_allowed_beforest_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and parsed.netloc.endswith("beforest.co")


def _is_same_beforest_experiences_domain(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and parsed.netloc == "experiences.beforest.co"


def _fetch_experience_page(url: str) -> _ExperiencePage:
    """Fetch and parse a single experiences page."""
    request = Request(
        url,
        headers={"User-Agent": "BeforestConcierge/1.0 (+https://beforest.co)"},
    )
    timeout_seconds = float(os.getenv("BEFOREST_HTTP_TIMEOUT_SECONDS", DEFAULT_HTTP_TIMEOUT_SECONDS))

    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            charset = response.headers.get_content_charset() or "utf-8"
            html = response.read().decode(charset, errors="replace")
    except HTTPError as exc:
        msg = f"Could not fetch {url}: HTTP {exc.code}"
        raise RuntimeError(msg) from exc
    except URLError as exc:
        msg = f"Could not fetch {url}: {exc.reason}"
        raise RuntimeError(msg) from exc

    parser = _ExperienceHTMLParser()
    parser.feed(html)
    title = _normalize_text(" ".join(parser.title_parts)) or url
    text = _normalize_text(" ".join(parser.parts))
    links: list[str] = []
    for href in parser.links:
        absolute_url = urljoin(url, href)
        if _is_same_beforest_experiences_domain(absolute_url):
            links.append(absolute_url)

    return _ExperiencePage(url=url, title=title, text=text, links=links)


def _fetch_beforest_page(url: str) -> dict[str, str | list[str]]:
    """Fetch a specific Beforest URL and return normalized text plus same-domain links."""
    if not _is_allowed_beforest_url(url):
        msg = "URL must be on beforest.co or one of its subdomains."
        raise ValueError(msg)

    request = Request(
        url,
        headers={"User-Agent": "BeforestConcierge/1.0 (+https://beforest.co)"},
    )
    timeout_seconds = float(os.getenv("BEFOREST_HTTP_TIMEOUT_SECONDS", DEFAULT_HTTP_TIMEOUT_SECONDS))

    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            charset = response.headers.get_content_charset() or "utf-8"
            html = response.read().decode(charset, errors="replace")
    except HTTPError as exc:
        msg = f"Could not fetch {url}: HTTP {exc.code}"
        raise RuntimeError(msg) from exc
    except URLError as exc:
        msg = f"Could not fetch {url}: {exc.reason}"
        raise RuntimeError(msg) from exc

    parser = _ExperienceHTMLParser()
    parser.feed(html)
    links: list[str] = []
    for href in parser.links:
        absolute_url = urljoin(url, href)
        if _is_allowed_beforest_url(absolute_url) and absolute_url not in links:
            links.append(absolute_url)

    return {
        "url": url,
        "title": _normalize_text(" ".join(parser.title_parts)) or url,
        "text": _normalize_text(" ".join(parser.parts)),
        "links": links,
    }


def _crawl_experience_pages() -> list[_ExperiencePage]:
    """Crawl a small set of same-domain experiences pages."""
    now = datetime.now(timezone.utc).timestamp()
    cached_at = float(_EXPERIENCE_CACHE.get("fetched_at", 0.0))
    cached_pages = _EXPERIENCE_CACHE.get("pages", [])
    if now - cached_at < EXPERIENCE_CACHE_TTL_SECONDS and isinstance(cached_pages, list):
        return [page for page in cached_pages if isinstance(page, _ExperiencePage)]

    queue = [EXPERIENCES_BASE_URL]
    seen: set[str] = set()
    pages: list[_ExperiencePage] = []

    while queue and len(pages) < MAX_EXPERIENCE_PAGES:
        current_url = queue.pop(0)
        if current_url in seen:
            continue
        seen.add(current_url)

        try:
            page = _fetch_experience_page(current_url)
        except RuntimeError:
            continue
        pages.append(page)

        for link in page.links:
            if link not in seen and link not in queue:
                queue.append(link)

    _EXPERIENCE_CACHE["pages"] = pages
    _EXPERIENCE_CACHE["fetched_at"] = now
    return pages


def _fetch_cached_experience_page(url: str) -> _ExperiencePage | None:
    """Fetch and cache a specific experiences page."""
    cached_page = _EXPERIENCE_PAGE_CACHE.get(url)
    if cached_page is not None:
        return cached_page
    try:
        page = _fetch_experience_page(url)
    except RuntimeError:
        return None
    _EXPERIENCE_PAGE_CACHE[url] = page
    return page


def _candidate_experience_links(pages: list[_ExperiencePage]) -> list[str]:
    """Collect likely experience detail links from crawled pages."""
    links: list[str] = []
    for page in pages:
        for link in page.links:
            parsed = urlparse(link)
            if not parsed.path.startswith("/experiences/"):
                continue
            if link not in links:
                links.append(link)
    return links


def _append_jsonl(path: Path, record: dict[str, str]) -> None:
    """Append a JSON line to a local data file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _smtp_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        msg = f"Missing required environment variable: {name}"
        raise ValueError(msg)
    return value


@tool
def search_beforest_knowledge(
    query: str,
    max_results: int = 3,
    intent: str = "",
    audience: str = "",
) -> list[dict[str, str | int]]:
    """Search Beforest knowledge with Outline as the primary source of truth.

    Args:
        query: Question or topic to search for.
        max_results: Maximum number of snippets to return.
        intent: Optional intent hint like `pricing`, `booking`, or `collectives`.
        audience: Optional audience hint like `member`, `prospect`, or `partner`.

    Returns:
        Matching knowledge snippets ordered by relevance.
    """
    outline_results = _load_outline_knowledge_results(
        query,
        max_results=max_results,
    )
    if outline_results:
        return outline_results[:max_results]

    if _get_outline_api_base() and os.getenv("OUTLINE_API_TOKEN", "").strip():
        return []

    convex_results = _load_convex_knowledge_results(
        query,
        max_results=max_results,
        intent=intent,
        audience=audience,
    )
    if convex_results:
        return convex_results[:max_results]

    results: list[dict[str, str | int]] = []

    for path in list(KNOWLEDGE_DIR.glob("*.md")) + list(KNOWLEDGE_CENTER_DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        score = _score_text(query, text)
        if score <= 0:
            continue

        results.append(
            {
                "source": path.name,
                "score": score,
                "snippet": _build_snippet(text, query),
            }
        )

    if NARRATIVE_PATH.exists():
        narrative_text = NARRATIVE_PATH.read_text(encoding="utf-8")
        narrative_score = _score_text(query, narrative_text)
        if narrative_score > 0:
            results.append(
                {
                    "source": NARRATIVE_PATH.name,
                    "score": narrative_score,
                    "snippet": _build_snippet(narrative_text, query),
                }
            )

    results.sort(key=lambda item: int(item["score"]), reverse=True)
    return results[:max_results]


@tool
def browse_beforest_page(url: str, query: str = "") -> dict[str, str | list[str]]:
    """Fetch a specific Beforest page when current, page-level information is needed.

    Args:
        url: A `beforest.co` or subdomain URL to fetch.
        query: Optional topic to focus the returned snippet on.

    Returns:
        Page title, URL, focused snippet, and a small list of discovered links.
    """
    try:
        page = _fetch_beforest_page(url)
    except (ValueError, RuntimeError) as exc:
        # Bad or unreachable URLs should not fail the whole agent run.
        return {
            "url": url,
            "title": "Unavailable page",
            "snippet": str(exc),
            "links": [],
        }
    text = str(page["text"])
    links = page["links"]
    snippet_query = query or text[:80]
    return {
        "url": str(page["url"]),
        "title": str(page["title"]),
        "snippet": _build_snippet(text, snippet_query),
        "links": links[:8] if isinstance(links, list) else [],
    }


@tool
def search_beforest_experiences(query: str, max_results: int = 5) -> list[dict[str, str | int]]:
    """Search live pages on `experiences.beforest.co` for relevant experience details.

    Args:
        query: Question or topic to search for.
        max_results: Maximum number of snippets to return.

    Returns:
        Matching live experience snippets ordered by relevance.
    """
    pages = _crawl_experience_pages()
    if not pages:
        return []
    results: list[dict[str, str | int]] = []

    for page in pages:
        combined = f"{page.title}\n{page.text}"
        score = _score_text(query, combined)
        if score <= 0:
            continue

        results.append(
            {
                "title": page.title,
                "url": page.url,
                "score": score,
                "snippet": _build_snippet(page.text, query),
            }
        )

    for link in _candidate_experience_links(pages)[:6]:
        page = _fetch_cached_experience_page(link)
        if page is None:
            continue
        combined = f"{page.title}\n{page.text}\n{page.url}"
        score = _score_text(query, combined)
        if score <= 0:
            continue
        results.append(
            {
                "title": page.title,
                "url": page.url,
                "score": score + 2,
                "snippet": _build_snippet(page.text, query),
            }
        )

    results.sort(key=lambda item: int(item["score"]), reverse=True)
    deduped_results: list[dict[str, str | int]] = []
    seen_urls: set[str] = set()
    for item in results:
        url = str(item["url"])
        if url in seen_urls:
            continue
        seen_urls.add(url)
        deduped_results.append(item)
        if len(deduped_results) >= max_results:
            break
    return deduped_results


@tool
def save_lead(
    name: str,
    email: str,
    interest: str,
    notes: str = "",
) -> str:
    """Save an interested Beforest lead for human follow-up.

    Args:
        name: Person's name.
        email: Email address for follow-up.
        interest: What they want help with.
        notes: Optional context such as city, phone, or preferred collective.

    Returns:
        Confirmation message for the agent.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "email": email,
        "interest": interest,
        "notes": notes,
    }
    _append_jsonl(LEADS_PATH, record)
    return f"Lead saved for {name}. Human follow-up can continue via hello@beforest.co."


@tool
def escalate_beforest_question(
    name: str,
    email: str,
    question: str,
    context: str = "",
    phone: str = "",
    city: str = "",
) -> str:
    """Email an unresolved visitor question to the Beforest team.

    Args:
        name: Visitor name.
        email: Visitor email for reply.
        question: The unresolved question or request.
        context: Additional conversation context to help the team respond.
        phone: Optional phone number.
        city: Optional city.

    Returns:
        Confirmation message for the agent.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "email": email,
        "phone": phone,
        "city": city,
        "question": question,
        "context": context,
    }
    _append_jsonl(ESCALATIONS_PATH, record)

    smtp_host = _smtp_env("BEFOREST_ESCALATION_SMTP_HOST")
    smtp_port = int(os.getenv("BEFOREST_ESCALATION_SMTP_PORT", "587"))
    smtp_username = _smtp_env("BEFOREST_ESCALATION_SMTP_USERNAME")
    smtp_password = _smtp_env("BEFOREST_ESCALATION_SMTP_PASSWORD")
    email_to = _smtp_env("BEFOREST_ESCALATION_EMAIL_TO")
    email_from = os.getenv("BEFOREST_ESCALATION_EMAIL_FROM", smtp_username).strip() or smtp_username
    use_starttls = os.getenv("BEFOREST_ESCALATION_SMTP_STARTTLS", "true").strip().lower() != "false"

    subject = f"Beforest concierge escalation: {name} needs follow-up"
    body = "\n".join(
        [
            "A visitor asked a question that the concierge could not answer confidently.",
            "",
            f"Name: {name}",
            f"Email: {email}",
            f"Phone: {phone or 'Not provided'}",
            f"City: {city or 'Not provided'}",
            "",
            "Question:",
            question,
            "",
            "Context:",
            context or "No extra context provided.",
        ]
    )

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = email_from
    message["To"] = email_to
    message["Reply-To"] = email
    message.set_content(body)

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as smtp:
            if use_starttls:
                smtp.starttls()
            smtp.login(smtp_username, smtp_password)
            smtp.send_message(message)
    except OSError as exc:
        msg = f"Escalation email could not be sent: {exc}"
        raise RuntimeError(msg) from exc
    except smtplib.SMTPException as exc:
        msg = f"Escalation email could not be sent: {exc}"
        raise RuntimeError(msg) from exc

    return f"Escalation sent for {name}. The Beforest team can reply to {email} directly."
