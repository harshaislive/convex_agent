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
      --bg: #f3f1eb;
      --paper: #fffdf8;
      --ink: #1e1d1a;
      --muted: #6f6a61;
      --line: #ddd6ca;
      --accent: #163a2c;
      --accent-soft: #e3efe9;
      --danger: #a04632;
      --shadow: 0 20px 60px rgba(27, 24, 18, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(214, 224, 214, 0.75), transparent 30%),
        linear-gradient(180deg, #f7f4ee 0%, var(--bg) 100%);
      min-height: 100vh;
    }
    .shell {
      max-width: 1240px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero {
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: flex-start;
      margin-bottom: 20px;
    }
    .hero h1 {
      margin: 0;
      font-size: clamp(2rem, 5vw, 3.6rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }
    .hero p {
      max-width: 520px;
      color: var(--muted);
      margin: 10px 0 0;
      font-size: 0.98rem;
      line-height: 1.5;
    }
    .badge {
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.8);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 0.82rem;
      backdrop-filter: blur(10px);
    }
    .grid {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 20px;
    }
    .panel {
      background: rgba(255, 253, 248, 0.84);
      border: 1px solid rgba(221, 214, 202, 0.9);
      border-radius: 28px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
      overflow: hidden;
    }
    .panel-inner { padding: 20px; }
    .section-title {
      margin: 0 0 12px;
      font-size: 0.84rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }
    .doc-list {
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: 72vh;
      overflow: auto;
      padding-right: 6px;
    }
    .doc-item {
      padding: 14px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.65);
      cursor: pointer;
      transition: transform 0.16s ease, border-color 0.16s ease, background 0.16s ease;
    }
    .doc-item:hover, .doc-item.active {
      transform: translateY(-1px);
      border-color: rgba(22, 58, 44, 0.28);
      background: var(--accent-soft);
    }
    .doc-item h3 {
      margin: 0 0 6px;
      font-size: 0.98rem;
      line-height: 1.3;
    }
    .doc-meta {
      color: var(--muted);
      font-size: 0.82rem;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .editor {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }
    .stack {
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    form {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    label {
      font-size: 0.84rem;
      color: var(--muted);
    }
    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.88);
      padding: 12px 14px;
      font: inherit;
      color: var(--ink);
    }
    textarea {
      min-height: 180px;
      resize: vertical;
      line-height: 1.5;
    }
    .preview {
      min-height: 540px;
      padding: 22px;
      border-radius: 24px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(249,246,240,0.92));
      overflow: auto;
      white-space: pre-wrap;
      line-height: 1.65;
    }
    .toolbar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 14px;
    }
    button {
      appearance: none;
      border: 0;
      border-radius: 999px;
      padding: 11px 16px;
      font: inherit;
      cursor: pointer;
      background: var(--accent);
      color: #f6f4ee;
    }
    button.secondary {
      background: rgba(255,255,255,0.9);
      color: var(--ink);
      border: 1px solid var(--line);
    }
    .status {
      font-size: 0.9rem;
      color: var(--muted);
      min-height: 1.4rem;
    }
    .status.error { color: var(--danger); }
    .pill {
      display: inline-flex;
      padding: 5px 9px;
      border-radius: 999px;
      background: rgba(22,58,44,0.09);
      color: var(--accent);
      font-size: 0.75rem;
      margin-right: 6px;
    }
    .login-shell {
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }
    .login-card {
      width: min(440px, 100%);
      padding: 28px;
      border-radius: 28px;
      background: rgba(255,253,248,0.86);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }
    @media (max-width: 960px) {
      .grid, .editor { grid-template-columns: 1fr; }
      .doc-list { max-height: none; }
      .preview { min-height: 320px; }
    }
  </style>
</head>
<body>
  <div id="app"></div>
  <script>
    const state = {
      docs: [],
      selectedSlug: null,
      isAuthed: false,
    };

    function escapeHtml(value) {
      return String(value || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
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

    function renderLogin() {
      document.getElementById("app").innerHTML = `
        <div class="login-shell">
          <div class="login-card">
            <div class="badge">Protected workspace</div>
            <h1 style="margin:14px 0 8px;font-size:2.2rem;line-height:0.96;letter-spacing:-0.04em;">Knowledge Center</h1>
            <p style="margin:0 0 18px;color:var(--muted);line-height:1.6;">Drop in markdown, ingest page URLs, and keep the agent's reference material current from one small workspace.</p>
            <form id="login-form">
              <label>Password</label>
              <input type="password" name="password" autocomplete="current-password" required>
              <button type="submit">Enter</button>
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
      const docCards = state.docs.map((doc) => `
        <div class="doc-item ${state.selectedSlug === doc.slug ? "active" : ""}" data-slug="${escapeHtml(doc.slug)}">
          <h3>${escapeHtml(doc.title)}</h3>
          <div class="doc-meta">
            <span>${escapeHtml(doc.source_type)}</span>
            <span>${new Date(doc.updated_at).toLocaleString()}</span>
          </div>
          <div style="margin-top:8px;">${(doc.tags || []).map((tag) => `<span class="pill">${escapeHtml(tag)}</span>`).join("")}</div>
        </div>`).join("");

      document.getElementById("app").innerHTML = `
        <div class="shell">
          <div class="hero">
            <div>
              <div class="badge">Private knowledge workspace</div>
              <h1>Knowledge<br>Center</h1>
              <p>Manage team markdown, ingest URLs into readable markdown snapshots, and keep the DM agent grounded in the right material.</p>
            </div>
            <div class="toolbar">
              <button class="secondary" id="refresh-btn">Refresh</button>
              <button class="secondary" id="logout-btn">Log out</button>
            </div>
          </div>
          <div class="grid">
            <div class="panel">
              <div class="panel-inner">
                <p class="section-title">Documents</p>
                <div class="doc-list">${docCards || '<div class="doc-item"><h3>No documents yet</h3><div class="doc-meta">Add markdown or ingest a page URL.</div></div>'}</div>
              </div>
            </div>
            <div class="stack">
              <div class="panel">
                <div class="panel-inner">
                  <p class="section-title">Add Sources</p>
                  <div class="editor">
                    <form id="markdown-form">
                      <label>Title</label>
                      <input name="title" placeholder="Collectives FAQ">
                      <label>Tags</label>
                      <input name="tags" placeholder="collectives, faq, sales">
                      <label>Markdown</label>
                      <textarea name="content" placeholder="# Notes&#10;&#10;Paste markdown here."></textarea>
                      <button type="submit">Save markdown</button>
                    </form>
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
                  <div class="status" data-status></div>
                </div>
              </div>
              <div class="panel">
                <div class="panel-inner">
                  <p class="section-title">Preview</p>
                  <div class="preview" id="preview">Select a document to preview its markdown.</div>
                </div>
              </div>
            </div>
          </div>
        </div>`;

      document.querySelectorAll("[data-slug]").forEach((node) => {
        node.addEventListener("click", () => loadDoc(node.getAttribute("data-slug")));
      });
      document.getElementById("refresh-btn").addEventListener("click", loadDocs);
      document.getElementById("logout-btn").addEventListener("click", logout);
      document.getElementById("markdown-form").addEventListener("submit", submitMarkdown);
      document.getElementById("url-form").addEventListener("submit", submitUrl);
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
        document.getElementById("preview").textContent = doc.content || "No content";
        renderApp();
        document.getElementById("preview").textContent = doc.content || "No content";
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
        render();
        if (state.selectedSlug) {
          await loadDoc(state.selectedSlug);
        }
      } catch (error) {
        setStatus(error.message, true);
      }
    }

    async function logout() {
      await api("/knowledge-center/logout", { method: "POST" });
      state.isAuthed = false;
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
