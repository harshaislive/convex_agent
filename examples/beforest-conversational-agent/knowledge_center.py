import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_HTTP_TIMEOUT_SECONDS = 8.0
SESSION_COOKIE_NAME = "knowledge_center_session"


@dataclass(slots=True)
class KnowledgeEntry:
    slug: str
    title: str
    entry_type: str
    summary: str
    body: str
    tags: list[str]
    intent_tags: list[str]
    audience_tags: list[str]
    priority: float
    status: str
    source_type: str | None
    source_url: str | None
    updated_at: str


class _MarkdownHTMLParser(HTMLParser):
    BLOCK_TAGS = {"article", "section", "div", "p", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6"}

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
        if tag == "li":
            self.parts.append("\n" + "  " * max(self._list_depth - 1, 0) + "- ")
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
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
    return slug or f"entry-{hashlib.sha1(value.encode('utf-8')).hexdigest()[:8]}"


def _normalize_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    seen: set[str] = set()
    values: list[str] = []
    for part in raw.split(","):
        cleaned = part.strip().lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        values.append(cleaned)
    return values


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
    headers = {"User-Agent": "BeforestKnowledgeCenter/2.0 (+https://beforest.co)"}
    if cookie_header:
        headers["Cookie"] = cookie_header
    if auth_header:
        headers["Authorization"] = auth_header
    timeout_seconds = float(os.getenv("BEFOREST_HTTP_TIMEOUT_SECONDS", DEFAULT_HTTP_TIMEOUT_SECONDS))
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


def _fetch_url_content_with_browser(
    url: str,
    *,
    cookie_header: str | None = None,
    auth_header: str | None = None,
) -> tuple[str, str]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Browser import is unavailable because Playwright is not installed.") from exc

    extra_headers: dict[str, str] = {}
    if cookie_header:
        extra_headers["Cookie"] = cookie_header
    if auth_header:
        extra_headers["Authorization"] = auth_header

    timeout_ms = int(float(os.getenv("BEFOREST_BROWSER_IMPORT_TIMEOUT_SECONDS", "18")) * 1000)

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(extra_http_headers=extra_headers or None)
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 8000))
            except Exception:  # noqa: BLE001
                pass
            html = page.content()
            title = page.title()
            context.close()
            browser.close()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Browser import failed for {url}: {exc}") from exc

    extracted_title, markdown = _extract_markdown_from_html(html)
    resolved_title = title.strip() or extracted_title
    if not markdown:
        raise RuntimeError("Browser fetched the page but could not extract readable content.")
    return resolved_title, markdown


def _knowledge_password() -> str:
    password = os.getenv("KNOWLEDGE_CENTER_PASSWORD", "").strip()
    if not password:
        raise RuntimeError("KNOWLEDGE_CENTER_PASSWORD is not configured.")
    return password


def _cookie_secret() -> str:
    return os.getenv("KNOWLEDGE_CENTER_SESSION_SECRET", "").strip() or os.getenv("AGENT_SHARED_SECRET", "").strip() or "beforest-knowledge-center"


def session_cookie_value() -> str:
    return hmac.new(_cookie_secret().encode("utf-8"), _knowledge_password().encode("utf-8"), hashlib.sha256).hexdigest()


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


def _get_convex_base() -> str:
    direct = os.getenv("CONVEX_HTTP_ACTION_URL", "").strip()
    if direct:
        return direct.replace("/instagram/store-dm-event", "")
    site_url = os.getenv("CONVEX_SITE_URL", "").strip()
    if site_url:
        return site_url.rstrip("/")
    raise RuntimeError("CONVEX_HTTP_ACTION_URL or CONVEX_SITE_URL is required.")


def _convex_secret() -> str:
    secret = os.getenv("AGENT_SHARED_SECRET", "").strip()
    if not secret:
        raise RuntimeError("AGENT_SHARED_SECRET is required.")
    return secret


def _convex_request(path: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> Any:
    request = Request(
        _get_convex_base() + path,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type": "application/json", "x-agent-secret": _convex_secret()},
        method=method,
    )
    try:
        with urlopen(request, timeout=15) as response:  # noqa: S310
            raw = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(detail or f"Convex request failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Convex: {exc.reason}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Convex returned invalid JSON.") from exc


def _entry_from_payload(item: dict[str, Any]) -> KnowledgeEntry:
    updated_at = item.get("updatedAt")
    iso = datetime.fromtimestamp(updated_at / 1000, tz=timezone.utc).isoformat() if isinstance(updated_at, (int, float)) else str(updated_at or "")
    return KnowledgeEntry(
        slug=str(item.get("slug", "")),
        title=str(item.get("title", "")),
        entry_type=str(item.get("type", "fact")),
        summary=str(item.get("summary", "") or ""),
        body=str(item.get("body", "") or ""),
        tags=[str(value) for value in item.get("tags", [])],
        intent_tags=[str(value) for value in item.get("intentTags", [])],
        audience_tags=[str(value) for value in item.get("audienceTags", [])],
        priority=float(item.get("priority", 0) or 0),
        status=str(item.get("status", "draft")),
        source_type=str(item.get("sourceType", "") or "") or None,
        source_url=str(item.get("sourceUrl", "") or "") or None,
        updated_at=iso,
    )


def list_entries(*, status: str | None = None, entry_type: str | None = None) -> list[dict[str, Any]]:
    query: list[str] = []
    if status:
        query.append(f"status={quote(status, safe='')}")
    if entry_type:
        query.append(f"type={quote(entry_type, safe='')}")
    suffix = f"?{'&'.join(query)}" if query else ""
    payload = _convex_request(f"/knowledge/entries{suffix}")
    if not isinstance(payload, list):
        return []
    entries = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        entry = _entry_from_payload(item)
        entries.append({
            "slug": entry.slug,
            "title": entry.title,
            "type": entry.entry_type,
            "summary": entry.summary,
            "tags": entry.tags,
            "intent_tags": entry.intent_tags,
            "audience_tags": entry.audience_tags,
            "priority": entry.priority,
            "status": entry.status,
            "source_type": entry.source_type,
            "source_url": entry.source_url,
            "updated_at": entry.updated_at,
        })
    return entries


def search_entries(
    *,
    query: str,
    intent: str | None = None,
    audience: str | None = None,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    params = [f"query={quote(query, safe='')}", f"maxResults={max(1, min(max_results, 10))}"]
    if intent:
        params.append(f"intent={quote(intent, safe='')}")
    if audience:
        params.append(f"audience={quote(audience, safe='')}")
    payload = _convex_request(f"/knowledge/search?{'&'.join(params)}")
    if not isinstance(payload, list):
        return []
    results: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        entry = _entry_from_payload(item)
        results.append(
            {
                "slug": entry.slug,
                "title": entry.title,
                "type": entry.entry_type,
                "summary": entry.summary,
                "tags": entry.tags,
                "intent_tags": entry.intent_tags,
                "audience_tags": entry.audience_tags,
                "priority": entry.priority,
                "status": entry.status,
                "source_type": entry.source_type,
                "source_url": entry.source_url,
                "updated_at": entry.updated_at,
                "score": int(item.get("score", 0) or 0),
                "body": entry.body,
            }
        )
    return results


def read_entry(slug: str) -> KnowledgeEntry:
    payload = _convex_request(f"/knowledge/entry?slug={quote(slug, safe='')}")
    if not isinstance(payload, dict):
        raise FileNotFoundError(slug)
    return _entry_from_payload(payload)


def save_entry(
    *,
    slug: str | None,
    title: str,
    entry_type: str,
    summary: str,
    body: str,
    tags: str | None,
    intent_tags: str | None,
    audience_tags: str | None,
    priority: float,
    status: str,
    source_type: str | None = None,
    source_url: str | None = None,
) -> dict[str, Any]:
    normalized_title = title.strip() or "Untitled entry"
    normalized_body = body.strip()
    if not normalized_body:
        raise ValueError("Body is required.")
    payload: dict[str, Any] = {
        "slug": (slug or "").strip() or _slugify(normalized_title),
        "title": normalized_title,
        "type": entry_type.strip() or "fact",
        "summary": summary.strip(),
        "body": normalized_body,
        "tags": _normalize_tags(tags),
        "intentTags": _normalize_tags(intent_tags),
        "audienceTags": _normalize_tags(audience_tags),
        "priority": float(priority),
        "status": status.strip() or "draft",
    }
    if source_type and source_type.strip():
        payload["sourceType"] = source_type.strip()
    if source_url and source_url.strip():
        payload["sourceUrl"] = source_url.strip()
    result = _convex_request("/knowledge/upsert-entry", method="POST", payload=payload)
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise RuntimeError("Convex did not confirm the save.")
    saved = read_entry(str(result.get("slug", payload["slug"])))
    return {
        "slug": saved.slug,
        "title": saved.title,
        "type": saved.entry_type,
        "summary": saved.summary,
        "body": saved.body,
        "tags": saved.tags,
        "intent_tags": saved.intent_tags,
        "audience_tags": saved.audience_tags,
        "priority": saved.priority,
        "status": saved.status,
        "source_type": saved.source_type,
        "source_url": saved.source_url,
        "updated_at": saved.updated_at,
    }


def import_url_entry(
    *,
    url: str,
    title: str | None = None,
    summary: str | None = None,
    entry_type: str = "fact",
    tags: str | None = None,
    intent_tags: str | None = None,
    audience_tags: str | None = None,
    priority: float = 0.5,
    status: str = "draft",
    cookie_header: str | None = None,
    auth_header: str | None = None,
) -> dict[str, Any]:
    try:
        fetched_title, markdown = _fetch_url_content(
            url,
            cookie_header=cookie_header,
            auth_header=auth_header,
        )
    except RuntimeError:
        fetched_title, markdown = _fetch_url_content_with_browser(
            url,
            cookie_header=cookie_header,
            auth_header=auth_header,
        )
    chosen_title = (title or fetched_title).strip() or fetched_title
    return save_entry(
        slug=None,
        title=chosen_title,
        entry_type=entry_type,
        summary=summary or f"Imported snapshot from {url}",
        body=markdown,
        tags=tags,
        intent_tags=intent_tags,
        audience_tags=audience_tags,
        priority=priority,
        status=status,
        source_type="url",
        source_url=url,
    )


def delete_entry(slug: str) -> dict[str, Any]:
    result = _convex_request("/knowledge/delete-entry", method="POST", payload={"slug": slug})
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise RuntimeError("Convex did not confirm the delete.")
    return {"deleted": bool(result.get("deleted")), "slug": slug}


def render_knowledge_center_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Knowledge Center</title>
  <style>
  :root { --bg:#f4f4f4; --ink:#111111; --muted:#707070; --line:#d8d8d8; --line-strong:#111111; --soft:#ffffff; --soft-2:#fafafa; --soft-3:#f0f0f0; --radius:16px; }
  * { box-sizing:border-box; }
  html { scroll-behavior:smooth; }
  body { margin:0; background:var(--bg); color:var(--ink); font-family:"IBM Plex Sans", Arial, Helvetica, sans-serif; }
  button, input, select, textarea { font:inherit; color:inherit; }
  button { cursor:pointer; border:1px solid var(--line-strong); border-radius:999px; background:var(--ink); color:#fff; padding:10px 16px; transition:transform 140ms ease, background 140ms ease, color 140ms ease, opacity 140ms ease; }
  button:hover { transform:translateY(-1px); }
  button.secondary { background:#fff; color:var(--ink); }
  button.ghost { background:transparent; color:var(--ink); border-color:var(--line); border-radius:999px; }
  input, select, textarea { width:100%; border:1px solid var(--line); border-radius:16px; padding:13px 15px; background:#fff; }
  input:focus, select:focus, textarea:focus { outline:none; border-color:var(--line-strong); box-shadow:0 0 0 3px rgba(0,0,0,0.04); }
  textarea { resize:vertical; min-height:140px; line-height:1.6; }
  .page { width:min(1540px, calc(100vw - 28px)); margin:0 auto; padding:18px 0 28px; }
  .topbar { display:flex; justify-content:space-between; gap:24px; align-items:center; padding:16px 20px; border:1px solid var(--line); border-radius:20px; background:#fff; margin-bottom:16px; }
  .title-block h1 { margin:0; font-size:clamp(1.8rem, 4vw, 3.2rem); line-height:.98; letter-spacing:-.05em; font-weight:700; font-family:"IBM Plex Sans", Arial, Helvetica, sans-serif; }
  .title-block p { margin:8px 0 0; color:var(--muted); max-width:720px; font-size:.95rem; line-height:1.55; }
  .mini { font-size:.7rem; text-transform:uppercase; letter-spacing:.16em; color:var(--muted); }
  .top-actions { display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }
  .workspace { display:grid; grid-template-columns:minmax(0, 1.2fr) minmax(380px, 0.9fr); gap:16px; align-items:start; }
  .panel { border:1px solid var(--line); border-radius:var(--radius); min-height:0; background:#fff; overflow:hidden; box-shadow:0 1px 0 rgba(0,0,0,0.02); }
  .panel-head { display:flex; justify-content:space-between; gap:10px; align-items:center; padding:16px 18px; border-bottom:1px solid var(--line); background:var(--soft-2); }
  .panel-head h2 { margin:0; font:700 .78rem/1 "IBM Plex Sans", Arial, sans-serif; text-transform:uppercase; letter-spacing:.18em; }
  .panel-body { padding:20px; }
  .stack { display:grid; gap:16px; }
  .field { display:grid; gap:8px; }
  .field label { font:.7rem/1 "IBM Plex Sans", Arial, sans-serif; text-transform:uppercase; letter-spacing:.16em; color:var(--muted); }
  .form-grid { display:flex; gap:12px; flex-wrap:wrap; }
  .form-grid > * { flex:1 1 170px; }
  .filters { display:flex; gap:10px; flex-wrap:wrap; padding:16px 18px; border-bottom:1px solid var(--line); background:var(--soft-2); }
  .filters > * { flex:0 0 auto; }
  .filters .grow { flex:1 1 220px; }
  .table-wrap { overflow:auto; }
  table { width:100%; border-collapse:collapse; }
  th, td { text-align:left; padding:14px 16px; border-bottom:1px solid var(--line); vertical-align:middle; }
  th { font:.72rem/1 "IBM Plex Sans", Arial, sans-serif; text-transform:uppercase; letter-spacing:.14em; color:var(--muted); background:#fff; position:sticky; top:0; }
  tbody tr { cursor:pointer; }
  tbody tr.active { background:var(--soft-3); }
  tbody tr:hover { background:var(--soft-2); }
  .cell-title { font-weight:600; }
  .cell-sub { margin-top:4px; color:var(--muted); font-size:.84rem; line-height:1.4; }
  .pill-row { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:8px; }
  .pill { border:1px solid var(--line); border-radius:999px; padding:4px 8px; font:.66rem/1 "IBM Plex Sans", Arial, sans-serif; text-transform:uppercase; letter-spacing:.1em; }
  .textarea-body { min-height:50vh; font-family:"Iowan Old Style", Georgia, "Times New Roman", serif; font-size:1rem; }
  .preview-block { border:1px solid var(--line); border-radius:16px; padding:16px; background:var(--soft-2); }
  .preview-block h3 { margin:0 0 10px; font:700 .76rem/1 "IBM Plex Sans", Arial, sans-serif; text-transform:uppercase; letter-spacing:.16em; }
  .preview-body { white-space:pre-wrap; line-height:1.58; max-height:34vh; overflow:auto; font-size:.94rem; }
  .editor-shell { display:grid; gap:14px; }
  .section-title { display:flex; justify-content:space-between; align-items:end; gap:10px; }
  .section-title h3 { margin:0; font-size:1.3rem; font-weight:700; letter-spacing:-.03em; font-family:"IBM Plex Sans", Arial, Helvetica, sans-serif; }
  .section-title p { margin:4px 0 0; color:var(--muted); font-size:.92rem; }
  .detail-card { border:1px solid var(--line); border-radius:16px; padding:16px; background:#fff; }
  .detail-card h4 { margin:0 0 10px; font:.72rem/1 "IBM Plex Sans", Arial, sans-serif; text-transform:uppercase; letter-spacing:.16em; color:var(--muted); }
  .detail-card p { margin:0; color:var(--muted); line-height:1.5; font-size:.9rem; }
  .toolbar { display:flex; gap:8px; flex-wrap:wrap; }
  .formatbar { display:flex; gap:6px; flex-wrap:wrap; padding:10px; border:1px solid var(--line); border-radius:14px; background:var(--soft-2); }
  .formatbar button { background:#fff; color:var(--ink); border-color:var(--line); padding:8px 12px; }
  .hide { display:none; }
  .login-wrap { min-height:100vh; display:grid; place-items:center; padding:18px; }
  .login-card { width:min(460px, 100%); border:1px solid var(--line-strong); border-radius:22px; padding:30px; background:#fff; }
  .login-card h1 { margin:0 0 12px; font-size:clamp(2.4rem, 9vw, 4.2rem); line-height:.92; letter-spacing:-.06em; font-weight:700; font-family:"IBM Plex Sans", Arial, Helvetica, sans-serif; }
  .login-card p, .status, .empty { color:var(--muted); }
  .status { min-height:1.2rem; font:.84rem/1.4 "IBM Plex Sans", Arial, sans-serif; }
  .status.error { color:#000; font-weight:700; }
  .empty { border:1px dashed var(--line); border-radius:16px; padding:16px; }
  @media (max-width: 1180px) { .workspace { grid-template-columns:1fr; } .textarea-body { min-height:34vh; } }
  @media (max-width: 720px) { .topbar { flex-direction:column; } .page { width:min(100vw - 12px, 100%); } }
  </style>
</head>
<body>
  <div id="app"></div>
  <script>
  const state = { isAuthed:false, entries:[], activeEntry:null, selectedSlug:null, filters:{ query:"", status:"", type:"" }, statusMessage:"", statusError:false, testResults:[], testQuery:"", testIntent:"", testAudience:"", showImport:false, showTester:false };
  function esc(value) { return String(value || "").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;"); }
  function setStatus(message, isError=false) {
    state.statusMessage = message || "";
    state.statusError = isError;
    document.querySelectorAll("[data-status]").forEach((node) => {
      node.textContent = state.statusMessage;
      node.className = "status" + (state.statusError ? " error" : "");
    });
  }
  async function api(path, options={}) {
    const response = await fetch(path, { credentials:"same-origin", headers:{ "Content-Type":"application/json", ...(options.headers || {}) }, ...options });
    const text = await response.text();
    let payload = {};
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch (_error) {
        payload = { detail: text };
      }
    }
    if (!response.ok) throw new Error(payload && payload.detail ? payload.detail : text || "Request failed");
    return payload;
  }
  function entryToForm(entry) {
    return {
      slug: entry?.slug || "",
      title: entry?.title || "",
      type: entry?.type || "fact",
      summary: entry?.summary || "",
      body: entry?.body || "",
      tags: (entry?.tags || []).join(", "),
      intent_tags: (entry?.intent_tags || []).join(", "),
      audience_tags: (entry?.audience_tags || []).join(", "),
      priority: String(entry?.priority ?? 0.5),
      status: entry?.status || "draft",
      source_url: entry?.source_url || "",
    };
  }
  function filteredEntries() {
    const q = state.filters.query.trim().toLowerCase();
    return state.entries.filter((entry) => {
      if (state.filters.status && entry.status !== state.filters.status) return false;
      if (state.filters.type && entry.type !== state.filters.type) return false;
      if (!q) return true;
      const haystack = [entry.title, entry.summary || "", (entry.tags || []).join(" "), (entry.intent_tags || []).join(" "), (entry.audience_tags || []).join(" ")].join(" ").toLowerCase();
      return haystack.includes(q);
    });
  }
  async function loadEntries(preferredSlug=null) {
    const entries = await api("/knowledge-center/api/entries");
    state.entries = entries;
    const slug = preferredSlug || state.selectedSlug || (entries[0] && entries[0].slug) || null;
    if (slug) await loadEntry(slug);
    else { state.selectedSlug = null; state.activeEntry = null; render(); }
  }
  async function loadEntry(slug) {
    const entry = await api(`/knowledge-center/api/entries/${slug}`);
    state.selectedSlug = slug;
    state.activeEntry = entry;
    render();
  }
  function bindLogin() {
    document.getElementById("login-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const form = new FormData(event.currentTarget);
      setStatus("Signing in...");
      try {
        await api("/knowledge-center/login", { method:"POST", body:JSON.stringify({ password: form.get("password") || "" }) });
        state.isAuthed = true;
        setStatus("");
        await loadEntries();
      } catch (error) {
        setStatus(error.message, true);
      }
    });
  }
  function bindShellEvents() {
    const query = document.getElementById("filter-query");
    const status = document.getElementById("filter-status");
    const type = document.getElementById("filter-type");
    if (query) query.addEventListener("input", () => { state.filters.query = query.value; render(); });
    if (status) status.addEventListener("change", () => { state.filters.status = status.value; render(); });
    if (type) type.addEventListener("change", () => { state.filters.type = type.value; render(); });
    document.querySelectorAll("[data-open-slug]").forEach((button) => {
      button.addEventListener("click", async () => {
        try { await loadEntry(button.getAttribute("data-open-slug")); }
        catch (error) { setStatus(error.message, true); }
      });
    });
    const newButton = document.getElementById("new-entry");
    if (newButton) {
      newButton.addEventListener("click", () => {
        state.selectedSlug = null;
        state.activeEntry = { slug:"", title:"", type:"fact", summary:"", body:"", tags:[], intent_tags:[], audience_tags:[], priority:0.5, status:"draft", source_url:"" };
        state.showImport = false;
        state.showTester = false;
        render();
      });
    }
    const importToggle = document.getElementById("toggle-import");
    if (importToggle) importToggle.addEventListener("click", () => {
      state.showImport = !state.showImport;
      if (state.showImport) state.showTester = false;
      render();
    });
    const testerToggle = document.getElementById("toggle-tester");
    if (testerToggle) testerToggle.addEventListener("click", () => {
      state.showTester = !state.showTester;
      if (state.showTester) state.showImport = false;
      render();
    });
    const deleteButton = document.getElementById("delete-entry");
    if (deleteButton) deleteButton.addEventListener("click", async () => {
      if (!state.activeEntry?.slug) return;
      const confirmed = window.confirm(`Delete "${state.activeEntry.title}"?`);
      if (!confirmed) return;
      try {
        setStatus("Deleting entry...");
        await api(`/knowledge-center/api/entries/${state.activeEntry.slug}`, { method:"DELETE" });
        state.activeEntry = null;
        state.selectedSlug = null;
        await loadEntries();
        setStatus("Entry deleted.");
      } catch (error) {
        setStatus(error.message, true);
      }
    });
    document.querySelectorAll("[data-format]").forEach((button) => {
      button.addEventListener("click", () => {
        const textarea = document.querySelector("textarea[name='body']");
        if (!textarea) return;
        const start = textarea.selectionStart || 0;
        const end = textarea.selectionEnd || 0;
        const selected = textarea.value.slice(start, end);
        const mode = button.getAttribute("data-format");
        let insertion = selected;
        if (mode === "bold") insertion = `**${selected || "text"}**`;
        if (mode === "h2") insertion = `## ${selected || "Heading"}`;
        if (mode === "bullet") insertion = `- ${selected || "List item"}`;
        if (mode === "link") insertion = `[${selected || "Link text"}](https://example.com)`;
        textarea.setRangeText(insertion, start, end, "end");
        textarea.focus();
      });
    });
    const refreshButton = document.getElementById("refresh-btn");
    if (refreshButton) refreshButton.addEventListener("click", async () => {
      try { setStatus("Refreshing..."); await loadEntries(state.selectedSlug); setStatus("Library refreshed."); }
      catch (error) { setStatus(error.message, true); }
    });
    const logoutButton = document.getElementById("logout-btn");
    if (logoutButton) logoutButton.addEventListener("click", async () => {
      await api("/knowledge-center/logout", { method:"POST" });
      state.isAuthed = false; state.entries = []; state.activeEntry = null; state.selectedSlug = null; render();
    });
    const entryForm = document.getElementById("entry-form");
    if (entryForm) entryForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const form = new FormData(event.currentTarget);
      const payload = Object.fromEntries(form.entries());
      payload.priority = Number(payload.priority || "0");
      try {
        setStatus("Saving entry...");
        const saved = await api("/knowledge-center/api/entries", { method:"POST", body:JSON.stringify(payload) });
        await loadEntries(saved.slug);
        setStatus(`Saved ${saved.title}.`);
      } catch (error) {
        setStatus(error.message, true);
      }
    });
    const importForm = document.getElementById("import-form");
    if (importForm) importForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const form = new FormData(event.currentTarget);
      const payload = Object.fromEntries(form.entries());
      payload.priority = Number(payload.priority || "0.5");
      try {
        setStatus("Importing URL...");
        const saved = await api("/knowledge-center/api/import-url", { method:"POST", body:JSON.stringify(payload) });
        event.currentTarget.reset();
        await loadEntries(saved.slug);
        state.showImport = false;
        setStatus(`Imported ${saved.title}.`);
      } catch (error) {
        setStatus(error.message, true);
      }
    });
    const testForm = document.getElementById("test-form");
    if (testForm) testForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const form = new FormData(event.currentTarget);
      state.testQuery = String(form.get("query") || "");
      state.testIntent = String(form.get("intent") || "");
      state.testAudience = String(form.get("audience") || "");
      try {
        setStatus("Running retrieval test...");
        state.testResults = await api("/knowledge-center/api/search", {
          method:"POST",
          body:JSON.stringify({
            query: state.testQuery,
            intent: state.testIntent,
            audience: state.testAudience,
          }),
        });
        render();
        setStatus("Retrieval test complete.");
      } catch (error) {
        setStatus(error.message, true);
      }
    });
  }
  function renderLogin() {
    document.getElementById("app").innerHTML = `
      <div class="login-wrap">
        <div class="login-card">
          <div class="mini">private access</div>
          <h1>Knowledge<br>Center</h1>
          <p>Convex-backed editing for the agent knowledge base.</p>
          <form id="login-form" class="stack">
            <div class="field"><label>Password</label><input type="password" name="password" autocomplete="current-password" required></div>
            <button type="submit">Enter workspace</button>
            <div class="status" data-status></div>
          </form>
        </div>
      </div>`;
    bindLogin();
  }
  function renderWorkspace() {
    const entries = filteredEntries();
    const active = entryToForm(state.activeEntry);
    const rows = entries.map((entry) => `
      <tr class="${entry.slug === state.selectedSlug ? "active" : ""}" data-open-slug="${esc(entry.slug)}">
        <td><div class="cell-title">${esc(entry.title)}</div><div class="cell-sub">${esc(entry.summary || "No summary yet.")}</div></td>
        <td>${esc(entry.type)}</td>
        <td><span class="pill">${esc(entry.status)}</span></td>
        <td>${esc((entry.tags || []).slice(0, 2).join(", ") || "-")}</td>
        <td>${esc(entry.updated_at ? entry.updated_at.slice(0, 10) : "-")}</td>
      </tr>`).join("");
    const pills = (values) => values && values.length ? values.map((value) => `<span class="pill">${esc(value)}</span>`).join("") : '<span class="mini">None</span>';
    const testCards = (state.testResults || []).map((item) => `
      <div class="preview-block">
        <h3>${esc(item.title)} <span class="mini">score ${esc(item.score)}</span></h3>
        <div class="pill-row"><span class="pill">${esc(item.type)}</span><span class="pill">${esc(item.status)}</span></div>
        <div class="mini">${esc(item.summary || "No summary")}</div>
        <div class="preview-body">${esc(item.body || "")}</div>
      </div>`).join("");
    document.getElementById("app").innerHTML = `
      <div class="page">
        <section class="topbar">
          <div class="title-block">
            <div class="mini">convex knowledge admin</div>
            <h1>Knowledge Center</h1>
            <p>Manage the agent knowledge base, import source material, and test retrieval from one place.</p>
          </div>
          <div class="top-actions">
            <button type="button" class="secondary" id="new-entry">New entry</button>
            <button type="button" class="ghost" id="toggle-import">${state.showImport ? "Close import" : "Import URL"}</button>
            <button type="button" class="ghost" id="toggle-tester">${state.showTester ? "Close test" : "Retrieval test"}</button>
            <button type="button" class="ghost" id="refresh-btn">Refresh</button>
            <button type="button" class="ghost" id="logout-btn">Log out</button>
          </div>
        </section>
        <section class="workspace">
          <section class="panel">
            <div class="panel-head"><h2>Knowledge base</h2><div class="mini">${state.entries.length} entries</div></div>
            <div class="filters">
              <input class="grow" id="filter-query" placeholder="Search knowledge" value="${esc(state.filters.query)}">
              <div class="form-grid">
                <select id="filter-status"><option value="">All statuses</option><option value="approved"${state.filters.status === "approved" ? " selected" : ""}>Approved</option><option value="draft"${state.filters.status === "draft" ? " selected" : ""}>Draft</option><option value="archived"${state.filters.status === "archived" ? " selected" : ""}>Archived</option></select>
                <select id="filter-type"><option value="">All types</option><option value="fact"${state.filters.type === "fact" ? " selected" : ""}>Fact</option><option value="faq"${state.filters.type === "faq" ? " selected" : ""}>FAQ</option><option value="offer"${state.filters.type === "offer" ? " selected" : ""}>Offer</option><option value="playbook"${state.filters.type === "playbook" ? " selected" : ""}>Playbook</option></select>
              </div>
            </div>
            <div class="table-wrap">
              <table>
                <thead><tr><th>Name</th><th>Type</th><th>Status</th><th>Tags</th><th>Updated</th></tr></thead>
                <tbody>${rows || '<tr><td colspan="5"><div class="empty">No entries match the current filter.</div></td></tr>'}</tbody>
              </table>
            </div>
          </section>
          <section class="workspace-main">
            <section class="panel">
              <div class="panel-head">
                <h2>Editor</h2>
                <div class="toolbar">
                  <button type="button" class="ghost" id="toggle-import">${state.showImport ? "Close import" : "Import URL"}</button>
                  <button type="button" class="ghost" id="toggle-tester">${state.showTester ? "Close test" : "Retrieval test"}</button>
                  <button type="button" class="ghost" id="delete-entry"${active.slug ? "" : " disabled"}>Delete</button>
                </div>
              </div>
              <div class="panel-body">
                <div class="section-title">
                  <div>
                    <h3>${esc(active.title || "New entry")}</h3>
                    <p>${active.slug ? `Editing ${esc(active.slug)}` : "Create a structured knowledge entry"}</p>
                  </div>
                </div>
                <div class="editor-shell">
                  <form id="entry-form" class="stack">
                <input type="hidden" name="slug" value="${esc(active.slug)}">
                <div class="form-grid">
                  <div class="field"><label>Title</label><input name="title" value="${esc(active.title)}" required></div>
                  <div class="field"><label>Type</label><select name="type"><option value="fact"${active.type === "fact" ? " selected" : ""}>Fact</option><option value="faq"${active.type === "faq" ? " selected" : ""}>FAQ</option><option value="offer"${active.type === "offer" ? " selected" : ""}>Offer</option><option value="playbook"${active.type === "playbook" ? " selected" : ""}>Playbook</option></select></div>
                </div>
                <div class="form-grid">
                  <div class="field"><label>Status</label><select name="status"><option value="draft"${active.status === "draft" ? " selected" : ""}>Draft</option><option value="approved"${active.status === "approved" ? " selected" : ""}>Approved</option><option value="archived"${active.status === "archived" ? " selected" : ""}>Archived</option></select></div>
                  <div class="field"><label>Priority</label><input type="number" name="priority" min="0" max="1" step="0.1" value="${esc(active.priority)}"></div>
                  <div class="field"><label>Source URL</label><input name="source_url" value="${esc(active.source_url)}" placeholder="Optional canonical source"></div>
                </div>
                <div class="field"><label>Summary</label><textarea name="summary">${esc(active.summary)}</textarea></div>
                <div class="form-grid">
                  <div class="field"><label>Tags</label><input name="tags" value="${esc(active.tags)}" placeholder="membership, pricing"></div>
                  <div class="field"><label>Intent Tags</label><input name="intent_tags" value="${esc(active.intent_tags)}" placeholder="booking, objections"></div>
                  <div class="field"><label>Audience Tags</label><input name="audience_tags" value="${esc(active.audience_tags)}" placeholder="prospect, member"></div>
                </div>
                <div class="formatbar">
                  <button type="button" data-format="h2">Heading</button>
                  <button type="button" data-format="bold">Bold</button>
                  <button type="button" data-format="bullet">Bullet</button>
                  <button type="button" data-format="link">Link</button>
                </div>
                <div class="field"><label>Body</label><textarea class="textarea-body" name="body" required>${esc(active.body)}</textarea></div>
                <div class="top-actions"><button type="submit">Save entry</button><div class="status" data-status></div></div>
                  </form>
                </div>
              </div>
            </section>
            <section class="${state.showImport || state.showTester ? "" : "hide"} stack">
              <section class="panel ${state.showImport ? "" : "hide"}">
                <div class="panel-head"><h2>Import source page</h2><div class="mini">optional tool</div></div>
                <div class="panel-body">
                  <form id="import-form" class="stack">
                <div class="field"><label>Import URL</label><input name="url" placeholder="https://example.com/page" required></div>
                <div class="field"><label>Entry Title</label><input name="title" placeholder="Optional override"></div>
                <div class="field"><label>Summary</label><textarea name="summary" placeholder="Short editorial note"></textarea></div>
                <div class="form-grid">
                  <div class="field"><label>Type</label><select name="type"><option value="fact">Fact</option><option value="faq">FAQ</option><option value="offer">Offer</option><option value="playbook">Playbook</option></select></div>
                  <div class="field"><label>Status</label><select name="status"><option value="draft">Draft</option><option value="approved">Approved</option></select></div>
                  <div class="field"><label>Priority</label><input type="number" name="priority" min="0" max="1" step="0.1" value="0.5"></div>
                </div>
                <div class="field"><label>Tags</label><input name="tags" placeholder="source, website"></div>
                <div class="field"><label>Intent Tags</label><input name="intent_tags" placeholder="pricing, booking"></div>
                <div class="field"><label>Audience Tags</label><input name="audience_tags" placeholder="prospect, member"></div>
                <div class="field"><label>Cookie Header</label><textarea name="cookie_header" placeholder="Optional"></textarea></div>
                <div class="field"><label>Authorization Header</label><textarea name="auth_header" placeholder="Optional"></textarea></div>
                <button type="submit" class="secondary">Import source</button>
              </form>
                </div>
              </section>
              <section class="panel ${state.showTester ? "" : "hide"}">
                <div class="panel-head"><h2>Retrieval test</h2><div class="mini">optional tool</div></div>
                <div class="panel-body stack">
                  <form id="test-form" class="stack">
                <div class="field"><label>Retrieval Test Query</label><input name="query" value="${esc(state.testQuery)}" placeholder="How does membership work?" required></div>
                <div class="form-grid">
                  <div class="field"><label>Intent</label><input name="intent" value="${esc(state.testIntent)}" placeholder="membership"></div>
                  <div class="field"><label>Audience</label><input name="audience" value="${esc(state.testAudience)}" placeholder="prospect"></div>
                </div>
                <button type="submit" class="secondary">Test retrieval</button>
              </form>
              ${testCards || '<div class="preview-block"><h3>Retrieval Results</h3><div class="empty">Run a retrieval test to see which Convex entries the agent would pull first.</div></div>'}
                </div>
              </section>
            </section>
            <section class="split">
              <div class="detail-card"><h4>Summary</h4><p>${esc(active.summary || "Write a short summary so this entry is easy to scan in the knowledge table.")}</p></div>
              <div class="detail-card"><h4>Metadata</h4><div class="pill-row">${pills(state.activeEntry?.tags || [])}</div><div class="pill-row">${pills(state.activeEntry?.intent_tags || [])}</div><div class="pill-row">${pills(state.activeEntry?.audience_tags || [])}</div></div>
            </section>
            <section class="preview-block">
              <h3>Preview</h3>
              <div class="preview-body">${state.activeEntry ? esc(state.activeEntry.body) : "No active entry."}</div>
            </section>
          </section>
        </section>
      </div>`;
    bindShellEvents();
    setStatus(state.statusMessage, state.statusError);
  }
  function render() { if (!state.isAuthed) { renderLogin(); setStatus(state.statusMessage, state.statusError); return; } renderWorkspace(); }
  (async function init() {
    try { state.isAuthed = true; await loadEntries(); }
    catch (_error) { state.isAuthed = false; render(); }
  })();
  </script>
</body>
</html>"""
