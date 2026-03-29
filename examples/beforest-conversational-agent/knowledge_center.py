import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EXAMPLE_DIR = Path(__file__).parent.resolve()
KNOWLEDGE_DIR = EXAMPLE_DIR / "knowledge"
KNOWLEDGE_CENTER_DIR = EXAMPLE_DIR / "knowledge_center"
DOCS_DIR = KNOWLEDGE_CENTER_DIR / "docs"
INDEX_PATH = KNOWLEDGE_CENTER_DIR / "index.json"
DEFAULT_HTTP_TIMEOUT_SECONDS = 8.0
SESSION_COOKIE_NAME = "knowledge_center_session"


@dataclass(slots=True)
class KnowledgeDocument:
    slug: str
    title: str
    content: str
    source_type: str
    tags: list[str]
    source_url: str | None
    updated_at: str


class _MarkdownHTMLParser(HTMLParser):
    """Extract readable markdown-ish content from a page."""

    BLOCK_TAGS = {
        "article",
        "section",
        "div",
        "p",
        "br",
        "li",
        "ul",
        "ol",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }

    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.parts: list[str] = []
        self._skip_stack: list[str] = []
        self._href_stack: list[str | None] = []
        self._in_title = False
        self._list_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_stack.append(tag)
            return
        if tag == "title":
            self._in_title = True
            return
        if tag == "a":
            self._href_stack.append(attr_map.get("href"))
        if tag in {"ul", "ol"}:
            self._list_depth += 1
        if tag in {"li"}:
            self.parts.append("\n" + "  " * max(self._list_depth - 1, 0) + "- ")
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag[1])
            self.parts.append("\n\n" + "#" * level + " ")
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_stack:
            self._skip_stack.pop()
            return
        if tag == "title":
            self._in_title = False
            return
        if tag == "a" and self._href_stack:
            href = self._href_stack.pop()
            if href:
                self.parts.append(f" ({href})")
        if tag in {"ul", "ol"} and self._list_depth > 0:
            self._list_depth -= 1
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            return
        cleaned = " ".join(data.split())
        if not cleaned:
            return
        if self._in_title:
            self.title_parts.append(cleaned)
        self.parts.append(cleaned)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or f"doc-{hashlib.sha1(value.encode('utf-8')).hexdigest()[:8]}"


def _ensure_dirs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        INDEX_PATH.write_text("[]\n", encoding="utf-8")


def _load_index() -> list[dict[str, Any]]:
    _ensure_dirs()
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def _save_index(items: list[dict[str, Any]]) -> None:
    _ensure_dirs()
    INDEX_PATH.write_text(json.dumps(items, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _doc_path(slug: str) -> Path:
    return DOCS_DIR / f"{slug}.md"


def _core_doc_slug(path: Path) -> str:
    return f"core--{path.stem}"


def _dedupe_slug(base_slug: str) -> str:
    slug = base_slug
    suffix = 2
    while _doc_path(slug).exists():
        slug = f"{base_slug}-{suffix}"
        suffix += 1
    return slug


def _normalize_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    seen: set[str] = set()
    tags: list[str] = []
    for part in raw.split(","):
        tag = part.strip().lower()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def _extract_markdown_from_html(html: str) -> tuple[str, str]:
    parser = _MarkdownHTMLParser()
    parser.feed(html)
    title = " ".join(parser.title_parts).strip() or "Imported page"
    content = "".join(parser.parts)
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r"[ \t]+\n", "\n", content)
    content = re.sub(r"\n +", "\n", content)
    return title, content.strip()


def _fetch_url_content(
    url: str,
    *,
    cookie_header: str | None = None,
    auth_header: str | None = None,
) -> tuple[str, str]:
    headers = {
        "User-Agent": "BeforestKnowledgeCenter/1.0 (+https://beforest.co)",
    }
    if cookie_header:
        headers["Cookie"] = cookie_header
    if auth_header:
        headers["Authorization"] = auth_header
    timeout_seconds = float(
        os.getenv("BEFOREST_HTTP_TIMEOUT_SECONDS", DEFAULT_HTTP_TIMEOUT_SECONDS)
    )
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            charset = response.headers.get_content_charset() or "utf-8"
            html = response.read().decode(charset, errors="replace")
    except HTTPError as exc:
        raise RuntimeError(f"Could not fetch {url}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not fetch {url}: {exc.reason}") from exc

    title, markdown = _extract_markdown_from_html(html)
    if not markdown:
        raise RuntimeError("Fetched page but could not extract readable content.")
    return title, markdown


def list_documents() -> list[dict[str, Any]]:
    items = list(_load_index())
    for path in KNOWLEDGE_DIR.glob("*.md"):
        stat = path.stat()
        items.append(
            {
                "slug": _core_doc_slug(path),
                "title": path.stem.replace("_", " ").replace("-", " ").title(),
                "source_type": "core",
                "source_url": None,
                "tags": ["built-in"],
                "updated_at": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )
    return sorted(items, key=lambda item: item.get("updated_at", ""), reverse=True)


def read_document(slug: str) -> KnowledgeDocument:
    if slug.startswith("core--"):
        name = slug.removeprefix("core--")
        path = KNOWLEDGE_DIR / f"{name}.md"
        if not path.exists():
            raise FileNotFoundError(slug)
        stat = path.stat()
        return KnowledgeDocument(
            slug=slug,
            title=path.stem.replace("_", " ").replace("-", " ").title(),
            content=path.read_text(encoding="utf-8"),
            source_type="core",
            tags=["built-in"],
            source_url=None,
            updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        )
    for item in _load_index():
        if item.get("slug") != slug:
            continue
        content = _doc_path(slug).read_text(encoding="utf-8")
        return KnowledgeDocument(
            slug=slug,
            title=str(item.get("title", slug)),
            content=content,
            source_type=str(item.get("source_type", "markdown")),
            tags=[str(tag) for tag in item.get("tags", [])],
            source_url=item.get("source_url"),
            updated_at=str(item.get("updated_at", "")),
        )
    raise FileNotFoundError(slug)


def save_markdown_document(title: str, content: str, tags: str | None = None) -> dict[str, Any]:
    _ensure_dirs()
    normalized_title = title.strip() or "Untitled document"
    normalized_content = content.strip()
    if not normalized_content:
        raise ValueError("Content is required.")
    slug = _dedupe_slug(_slugify(normalized_title))
    _doc_path(slug).write_text(normalized_content + "\n", encoding="utf-8")
    record = {
        "slug": slug,
        "title": normalized_title,
        "source_type": "markdown",
        "source_url": None,
        "tags": _normalize_tags(tags),
        "updated_at": _now_iso(),
    }
    items = _load_index()
    items.append(record)
    _save_index(items)
    return record


def ingest_url_document(
    url: str,
    *,
    title: str | None = None,
    tags: str | None = None,
    cookie_header: str | None = None,
    auth_header: str | None = None,
) -> dict[str, Any]:
    _ensure_dirs()
    fetched_title, markdown = _fetch_url_content(
        url,
        cookie_header=cookie_header,
        auth_header=auth_header,
    )
    chosen_title = (title or fetched_title).strip() or fetched_title
    slug = _dedupe_slug(_slugify(chosen_title))
    frontmatter = [
        "---",
        f'title: "{chosen_title.replace("\"", "\\\"")}"',
        f'source_url: "{url}"',
        f'fetched_at: "{_now_iso()}"',
        "---",
        "",
    ]
    _doc_path(slug).write_text("\n".join(frontmatter) + markdown + "\n", encoding="utf-8")
    record = {
        "slug": slug,
        "title": chosen_title,
        "source_type": "url",
        "source_url": url,
        "tags": _normalize_tags(tags),
        "updated_at": _now_iso(),
    }
    items = _load_index()
    items.append(record)
    _save_index(items)
    return record


def _knowledge_password() -> str:
    password = os.getenv("KNOWLEDGE_CENTER_PASSWORD", "").strip()
    if not password:
        raise RuntimeError("KNOWLEDGE_CENTER_PASSWORD is not configured.")
    return password


def _cookie_secret() -> str:
    return (
        os.getenv("KNOWLEDGE_CENTER_SESSION_SECRET", "").strip()
        or os.getenv("AGENT_SHARED_SECRET", "").strip()
        or "beforest-knowledge-center"
    )


def session_cookie_value() -> str:
    return hmac.new(
        _cookie_secret().encode("utf-8"),
        _knowledge_password().encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_password(password: str) -> bool:
    try:
        expected = _knowledge_password()
    except RuntimeError:
        return False
    return hmac.compare_digest(password, expected)


def is_authenticated(cookie_value: str | None) -> bool:
    if not cookie_value:
        return False
    try:
        expected = session_cookie_value()
    except RuntimeError:
        return False
    return hmac.compare_digest(cookie_value, expected)


def render_knowledge_center_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Knowledge Center</title>
  <style>
    :root {
      --bg: #f2ede4;
      --paper: #fffdf8;
      --panel: rgba(255, 252, 246, 0.88);
      --panel-strong: rgba(255, 252, 246, 0.96);
      --ink: #171512;
      --muted: #6c655a;
      --line: rgba(64, 49, 30, 0.12);
      --line-strong: rgba(22, 58, 44, 0.24);
      --accent: #17372d;
      --accent-2: #a56439;
      --accent-soft: rgba(23, 55, 45, 0.08);
      --danger: #a04632;
      --shadow: 0 24px 80px rgba(24, 18, 10, 0.1);
      --radius: 28px;
    }
    * { box-sizing: border-box; }
    html, body { min-height: 100%; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      background:
        radial-gradient(circle at top left, rgba(220, 229, 214, 0.8), transparent 28%),
        radial-gradient(circle at top right, rgba(243, 212, 192, 0.55), transparent 30%),
        linear-gradient(180deg, #f8f3eb 0%, var(--bg) 58%, #ebe3d8 100%);
    }
    button, input, textarea {
      font: inherit;
    }
    .shell {
      width: min(1380px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 22px 0 30px;
    }
    .hero {
      position: relative;
      display: grid;
      grid-template-columns: 1.3fr 0.9fr;
      gap: 18px;
      margin-bottom: 18px;
    }
    .hero-card, .hero-side {
      background: linear-gradient(180deg, rgba(255,255,255,0.78), rgba(255,251,245,0.72));
      border: 1px solid rgba(255,255,255,0.7);
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
      border-radius: 32px;
      padding: 26px;
    }
    .hero-card::after {
      content: "";
      position: absolute;
      inset: auto 18px 18px auto;
      width: 160px;
      height: 160px;
      background: radial-gradient(circle, rgba(165,100,57,0.18), transparent 68%);
      pointer-events: none;
      filter: blur(6px);
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 9px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.76);
      border: 1px solid var(--line);
      color: var(--muted);
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 0.78rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .hero h1 {
      margin: 14px 0 10px;
      font-size: clamp(2.4rem, 6vw, 5rem);
      line-height: 0.9;
      letter-spacing: -0.06em;
      font-weight: 600;
    }
    .hero p {
      margin: 0;
      max-width: 52rem;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.6;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    .hero-stats {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 22px;
    }
    .stat {
      border-radius: 22px;
      padding: 15px 16px;
      background: rgba(255,255,255,0.62);
      border: 1px solid var(--line);
    }
    .stat strong {
      display: block;
      font-size: 1.4rem;
      line-height: 1;
      letter-spacing: -0.04em;
    }
    .stat span {
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-size: 0.82rem;
      font-family: ui-sans-serif, system-ui, sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .hero-side {
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 16px;
      background:
        linear-gradient(180deg, rgba(19,55,45,0.95), rgba(19,55,45,0.84)),
        radial-gradient(circle at top right, rgba(255,255,255,0.14), transparent 34%);
      color: #f5efe6;
    }
    .hero-side h2 {
      margin: 0;
      font-size: 1.5rem;
      line-height: 1.05;
      letter-spacing: -0.04em;
    }
    .hero-side p {
      color: rgba(245,239,230,0.78);
      font-size: 0.95rem;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    button {
      appearance: none;
      border: none;
      border-radius: 999px;
      padding: 11px 16px;
      background: var(--accent);
      color: #f8f5ef;
      cursor: pointer;
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 0.94rem;
      transition: transform 0.18s ease, opacity 0.18s ease, box-shadow 0.18s ease;
      box-shadow: 0 8px 24px rgba(23,55,45,0.18);
    }
    button:hover {
      transform: translateY(-1px);
    }
    button.secondary {
      background: rgba(255,255,255,0.9);
      color: var(--ink);
      border: 1px solid var(--line);
      box-shadow: none;
    }
    button.ghost {
      background: rgba(255,255,255,0.12);
      color: #f8f5ef;
      border: 1px solid rgba(255,255,255,0.16);
      box-shadow: none;
    }
    .workspace {
      display: grid;
      grid-template-columns: 320px minmax(0, 1.04fr) minmax(320px, 0.96fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.72);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
      overflow: hidden;
    }
    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 18px 20px 12px;
      border-bottom: 1px solid rgba(64,49,30,0.06);
    }
    .panel-title {
      margin: 0;
      font-size: 0.78rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    .panel-body {
      padding: 18px 20px 20px;
    }
    .search {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.88);
      padding: 12px 14px;
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, sans-serif;
      margin-bottom: 14px;
    }
    .doc-list {
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: calc(100vh - 320px);
      overflow: auto;
      padding-right: 4px;
    }
    .doc-item {
      border-radius: 22px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.64);
      padding: 14px 15px;
      cursor: pointer;
      transition: transform 0.16s ease, background 0.16s ease, border-color 0.16s ease;
    }
    .doc-item:hover,
    .doc-item.active {
      transform: translateY(-1px);
      background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(227,239,233,0.82));
      border-color: var(--line-strong);
    }
    .doc-item h3 {
      margin: 0 0 7px;
      font-size: 1rem;
      line-height: 1.22;
      letter-spacing: -0.02em;
    }
    .doc-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      color: var(--muted);
      font-size: 0.79rem;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      padding: 5px 9px;
      border-radius: 999px;
      background: rgba(23,55,45,0.08);
      color: var(--accent);
      font-size: 0.72rem;
      letter-spacing: 0.02em;
      font-family: ui-sans-serif, system-ui, sans-serif;
      margin: 7px 6px 0 0;
    }
    .studio {
      display: grid;
      gap: 18px;
    }
    .form-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }
    .card-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    .mini-card {
      border-radius: 24px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.58);
      padding: 16px;
    }
    .mini-card h3 {
      margin: 0 0 6px;
      font-size: 1.18rem;
      line-height: 1.05;
      letter-spacing: -0.03em;
    }
    .mini-card p {
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.5;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    form {
      display: grid;
      gap: 10px;
    }
    label {
      color: var(--muted);
      font-size: 0.8rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.88);
      padding: 12px 14px;
      color: var(--ink);
    }
    textarea {
      resize: vertical;
      min-height: 148px;
      line-height: 1.55;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
    }
    .hint {
      color: var(--muted);
      font-size: 0.78rem;
      line-height: 1.45;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    .status {
      min-height: 1.3rem;
      color: var(--muted);
      font-size: 0.92rem;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    .status.error { color: var(--danger); }
    .preview-card {
      background: var(--panel-strong);
    }
    .preview-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 0.8rem;
      color: var(--muted);
    }
    .preview-title {
      margin: 0;
      font-size: 2rem;
      line-height: 0.95;
      letter-spacing: -0.05em;
    }
    .preview-source {
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.5;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    .preview {
      margin-top: 18px;
      min-height: calc(100vh - 360px);
      border-radius: 26px;
      border: 1px solid var(--line);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.9), rgba(249,244,236,0.96));
      padding: 24px;
      overflow: auto;
      white-space: pre-wrap;
      line-height: 1.72;
      font-size: 1.02rem;
    }
    .login-shell {
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }
    .login-card {
      width: min(520px, 100%);
      padding: 30px;
      border-radius: 34px;
      background: linear-gradient(180deg, rgba(255,255,255,0.78), rgba(255,250,243,0.76));
      border: 1px solid rgba(255,255,255,0.72);
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }
    .login-card h1 {
      margin: 14px 0 10px;
      font-size: 3rem;
      line-height: 0.92;
      letter-spacing: -0.06em;
    }
    .login-card p {
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.6;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }
    @media (max-width: 1120px) {
      .workspace {
        grid-template-columns: 1fr;
      }
      .doc-list, .preview {
        max-height: none;
        min-height: 320px;
      }
      .preview {
        min-height: 360px;
      }
    }
    @media (max-width: 860px) {
      .shell {
        width: min(100vw - 18px, 100%);
      }
      .hero {
        grid-template-columns: 1fr;
      }
      .hero-stats,
      .card-grid {
        grid-template-columns: 1fr;
      }
      .hero-card, .hero-side, .panel {
        border-radius: 24px;
      }
      .preview-title {
        font-size: 1.6rem;
      }
      .preview {
        padding: 18px;
        font-size: 0.98rem;
      }
    }
  </style>
</head>
<body>
  <div id="app"></div>
  <script>
    const state = {
      docs: [],
      selectedSlug: null,
      activeDoc: null,
      isAuthed: false,
      filter: "",
    };

    function escapeHtml(value) {
      return String(value || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function formatDate(value) {
      if (!value) return "";
      const date = new Date(value);
      return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
    }

    function setStatus(message, isError = false) {
      const el = document.querySelector("[data-status]");
      if (!el) return;
      el.textContent = message || "";
      el.classList.toggle("error", Boolean(isError));
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        credentials: "same-origin",
        headers: {
          "Content-Type": "application/json",
          ...(options.headers || {}),
        },
        ...options,
      });
      if (response.status === 401) {
        state.isAuthed = false;
        state.activeDoc = null;
        render();
        throw new Error("Unauthorized");
      }
      const contentType = response.headers.get("content-type") || "";
      const body = contentType.includes("application/json") ? await response.json() : await response.text();
      if (!response.ok) {
        const message = body && body.detail ? body.detail : response.statusText;
        throw new Error(message);
      }
      return body;
    }

    function filteredDocs() {
      const query = state.filter.trim().toLowerCase();
      if (!query) return state.docs;
      return state.docs.filter((doc) => {
        const haystack = [
          doc.title,
          doc.source_type,
          ...(doc.tags || []),
        ].join(" ").toLowerCase();
        return haystack.includes(query);
      });
    }

    function renderLogin() {
      document.getElementById("app").innerHTML = `
        <div class="login-shell">
          <div class="login-card">
            <div class="eyebrow">Private workspace</div>
            <h1>Knowledge<br>Center</h1>
            <p>One place for markdown notes, imported reference pages, and grounded material your DM agent can actually search.</p>
            <form id="login-form">
              <label>Password</label>
              <input type="password" name="password" autocomplete="current-password" required>
              <button type="submit">Enter workspace</button>
              <div class="status" data-status></div>
            </form>
          </div>
        </div>`;
      document.getElementById("login-form").addEventListener("submit", async (event) => {
        event.preventDefault();
        const form = new FormData(event.currentTarget);
        setStatus("Signing in...");
        try {
          await api("/knowledge-center/login", {
            method: "POST",
            body: JSON.stringify({ password: form.get("password") || "" }),
          });
          state.isAuthed = true;
          await loadDocs();
        } catch (error) {
          setStatus(error.message, true);
        }
      });
    }

    function renderApp() {
      const docs = filteredDocs();
      const activeDoc = state.activeDoc;
      const coreCount = state.docs.filter((doc) => doc.source_type === "core").length;
      const importedCount = state.docs.filter((doc) => doc.source_type !== "core").length;
      const docCards = docs.map((doc) => `
        <div class="doc-item ${state.selectedSlug === doc.slug ? "active" : ""}" data-slug="${escapeHtml(doc.slug)}">
          <h3>${escapeHtml(doc.title)}</h3>
          <div class="doc-meta">
            <span>${escapeHtml(doc.source_type)}</span>
            <span>${escapeHtml(formatDate(doc.updated_at))}</span>
          </div>
          <div>${(doc.tags || []).map((tag) => `<span class="pill">${escapeHtml(tag)}</span>`).join("")}</div>
        </div>`).join("");

      document.getElementById("app").innerHTML = `
        <div class="shell">
          <section class="hero">
            <div class="hero-card">
              <div class="eyebrow">Knowledge workspace</div>
              <h1>Library,<br>Studio,<br>Preview.</h1>
              <p>Keep your agent grounded with built-in notes, fresh markdown, and extracted page snapshots. The workspace is private, touch-friendly, and fast enough to use from your phone without feeling like admin software.</p>
              <div class="hero-stats">
                <div class="stat"><strong>${state.docs.length}</strong><span>Total docs</span></div>
                <div class="stat"><strong>${coreCount}</strong><span>Built-in</span></div>
                <div class="stat"><strong>${importedCount}</strong><span>Imported</span></div>
              </div>
            </div>
            <aside class="hero-side">
              <div>
                <h2>Import from notes or locked pages.</h2>
                <p>Paste markdown directly, or ingest a page URL with optional cookies or bearer auth when your team needs material from protected systems.</p>
              </div>
              <div class="toolbar">
                <button class="ghost" id="refresh-btn">Refresh</button>
                <button class="ghost" id="logout-btn">Log out</button>
              </div>
            </aside>
          </section>

          <section class="workspace">
            <div class="panel">
              <div class="panel-header">
                <p class="panel-title">Library</p>
              </div>
              <div class="panel-body">
                <input class="search" id="doc-search" placeholder="Filter by title, type, or tag" value="${escapeHtml(state.filter)}">
                <div class="doc-list">${docCards || '<div class="doc-item"><h3>No matching documents</h3><div class="doc-meta">Adjust the filter or add a new source.</div></div>'}</div>
              </div>
            </div>

            <div class="studio">
              <div class="panel">
                <div class="panel-header">
                  <p class="panel-title">Studio</p>
                </div>
                <div class="panel-body">
                  <div class="card-grid">
                    <div class="mini-card">
                      <h3>Markdown drop</h3>
                      <p>Useful for FAQs, internal notes, objections, and draft positioning material.</p>
                      <form id="markdown-form">
                        <label>Title</label>
                        <input name="title" placeholder="Collectives FAQ">
                        <label>Tags</label>
                        <input name="tags" placeholder="collectives, faq, sales">
                        <label>Markdown</label>
                        <textarea name="content" placeholder="# Notes&#10;&#10;Paste markdown here."></textarea>
                        <button type="submit">Save markdown</button>
                      </form>
                    </div>
                    <div class="mini-card">
                      <h3>URL ingest</h3>
                      <p>Fetch a page once, extract readable markdown, and keep a stable snapshot for retrieval.</p>
                      <form id="url-form">
                        <label>Page URL</label>
                        <input name="url" placeholder="https://example.com/private-page" required>
                        <label>Title override</label>
                        <input name="title" placeholder="Optional">
                        <label>Tags</label>
                        <input name="tags" placeholder="url, source">
                        <label>Cookie header</label>
                        <textarea name="cookie_header" placeholder="Optional. Example: sessionid=..."></textarea>
                        <label>Authorization header</label>
                        <textarea name="auth_header" placeholder="Optional. Example: Bearer ..."></textarea>
                        <button type="submit">Ingest URL</button>
                      </form>
                    </div>
                  </div>
                  <div class="hint" style="margin-top:14px;">Imported pages are stored as markdown snapshots. The agent searches these alongside the built-in knowledge files.</div>
                  <div class="status" data-status></div>
                </div>
              </div>
            </div>

            <div class="panel preview-card">
              <div class="panel-header">
                <p class="panel-title">Preview</p>
              </div>
              <div class="panel-body">
                <h2 class="preview-title">${escapeHtml(activeDoc && activeDoc.title ? activeDoc.title : "Select a document")}</h2>
                <div class="preview-meta">
                  ${activeDoc ? `<span>${escapeHtml(activeDoc.source_type || "")}</span>` : ""}
                  ${activeDoc ? `<span>${escapeHtml(formatDate(activeDoc.updated_at || ""))}</span>` : ""}
                  ${activeDoc && activeDoc.tags ? activeDoc.tags.map((tag) => `<span class="pill">${escapeHtml(tag)}</span>`).join("") : ""}
                </div>
                ${activeDoc && activeDoc.source_url ? `<div class="preview-source">Source: ${escapeHtml(activeDoc.source_url)}</div>` : ""}
                <div class="preview" id="preview">${escapeHtml(activeDoc && activeDoc.content ? activeDoc.content : "Pick a document from the library to preview its markdown here.")}</div>
              </div>
            </div>
          </section>
        </div>`;

      document.querySelectorAll("[data-slug]").forEach((node) => {
        node.addEventListener("click", () => loadDoc(node.getAttribute("data-slug")));
      });
      document.getElementById("refresh-btn").addEventListener("click", () => loadDocs(state.selectedSlug));
      document.getElementById("logout-btn").addEventListener("click", logout);
      document.getElementById("markdown-form").addEventListener("submit", submitMarkdown);
      document.getElementById("url-form").addEventListener("submit", submitUrl);
      document.getElementById("doc-search").addEventListener("input", (event) => {
        state.filter = event.target.value || "";
        renderApp();
      });
    }

    async function submitMarkdown(event) {
      event.preventDefault();
      const form = new FormData(event.currentTarget);
      setStatus("Saving markdown...");
      try {
        const result = await api("/knowledge-center/api/markdown", {
          method: "POST",
          body: JSON.stringify(Object.fromEntries(form.entries())),
        });
        setStatus(`Saved ${result.title}.`);
        event.currentTarget.reset();
        await loadDocs(result.slug);
      } catch (error) {
        setStatus(error.message, true);
      }
    }

    async function submitUrl(event) {
      event.preventDefault();
      const form = new FormData(event.currentTarget);
      setStatus("Fetching page and extracting markdown...");
      try {
        const result = await api("/knowledge-center/api/url", {
          method: "POST",
          body: JSON.stringify(Object.fromEntries(form.entries())),
        });
        setStatus(`Imported ${result.title}.`);
        event.currentTarget.reset();
        await loadDocs(result.slug);
      } catch (error) {
        setStatus(error.message, true);
      }
    }

    async function loadDoc(slug) {
      try {
        const doc = await api(`/knowledge-center/api/documents/${slug}`);
        state.selectedSlug = slug;
        state.activeDoc = doc;
        renderApp();
      } catch (error) {
        setStatus(error.message, true);
      }
    }

    async function loadDocs(selectSlug = null) {
      try {
        const docs = await api("/knowledge-center/api/documents");
        state.docs = docs;
        state.selectedSlug = selectSlug || state.selectedSlug || (docs[0] && docs[0].slug) || null;
        state.isAuthed = true;
        if (state.selectedSlug) {
          try {
            state.activeDoc = await api(`/knowledge-center/api/documents/${state.selectedSlug}`);
          } catch (error) {
            state.activeDoc = null;
          }
        } else {
          state.activeDoc = null;
        }
        render();
      } catch (error) {
        setStatus(error.message, true);
      }
    }

    async function logout() {
      await api("/knowledge-center/logout", { method: "POST" });
      state.isAuthed = false;
      state.activeDoc = null;
      render();
    }

    function render() {
      if (!state.isAuthed) {
        renderLogin();
        return;
      }
      renderApp();
    }

    (async function init() {
      try {
        await loadDocs();
      } catch (error) {
        renderLogin();
      }
    })();
  </script>
</body>
</html>"""
