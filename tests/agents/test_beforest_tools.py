from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from agents.beforest_tools import (
    _build_beforest_experiences_outline_markdown,
    _fetch_beforest_page,
    _is_allowed_beforest_url,
    _query_terms,
    _score_text,
    _sync_beforest_experiences_to_outline,
    browse_beforest_page,
    fetch_beforest_markdown,
    search_beforest_experiences,
)


class _FakeResponse:
    def __init__(self, html: str):
        self._html = html.encode("utf-8")
        self.headers = SimpleNamespace(get_content_charset=lambda: "utf-8")

    def read(self) -> bytes:
        return self._html

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _fake_urlopen(request, timeout=0):
    html = """
    <html>
      <head><title>Beforest Experiences</title></head>
      <body>
        <h1>Visit Beforest</h1>
        <p>Plan your stay with <a href="/visit">the visit guide</a>.</p>
        <h2>Highlights</h2>
        <ul>
          <li>Forest stays</li>
          <li>Workshops</li>
        </ul>
        <p><a href="https://outside.example.com/other">External</a></p>
      </body>
    </html>
    """
    return _FakeResponse(html)


def test_is_allowed_beforest_url():
    assert _is_allowed_beforest_url("https://beforest.co")
    assert _is_allowed_beforest_url("https://experiences.beforest.co/retreat")
    assert _is_allowed_beforest_url("https://bewild.life/shop")
    assert not _is_allowed_beforest_url("https://beforest.co.evil.com")
    assert not _is_allowed_beforest_url("https://evilbeforest.co")
    assert not _is_allowed_beforest_url("https://example.com")


@patch("agents.beforest_tools.urlopen", side_effect=_fake_urlopen)
def test_fetch_beforest_page_converts_html_to_markdown(mock_urlopen):
    page = _fetch_beforest_page("https://beforest.co")

    assert page["title"] == "Beforest Experiences"
    assert "# Visit Beforest" in page["markdown"]
    assert "## Highlights" in page["markdown"]
    assert "- Forest stays" in page["markdown"]
    assert "[the visit guide](https://beforest.co/visit)" in page["markdown"]
    assert page["links"] == ["https://beforest.co/visit"]
    assert "Visit Beforest" in page["text"]
    assert "the visit guide" in page["text"]


@patch("agents.beforest_tools.urlopen", side_effect=_fake_urlopen)
def test_fetch_beforest_markdown_tool_returns_markdown(mock_urlopen):
    result = fetch_beforest_markdown.invoke({"url": "https://beforest.co"})

    assert result["title"] == "Beforest Experiences"
    assert "# Visit Beforest" in result["markdown"]
    assert result["links"] == ["https://beforest.co/visit"]


@patch("agents.beforest_tools.urlopen", side_effect=_fake_urlopen)
def test_browse_beforest_page_returns_markdown_backed_snippet(mock_urlopen):
    result = browse_beforest_page.invoke(
        {"url": "https://beforest.co", "query": "visit guide workshops"}
    )

    assert result["title"] == "Beforest Experiences"
    assert "visit guide" in result["snippet"].lower()
    assert "# Visit Beforest" in result["markdown"]
    assert result["links"] == ["https://beforest.co/visit"]


def test_query_terms_expand_semantic_variants():
    terms = _query_terms("Who is the founder of Beforest")

    assert "founder" in terms
    assert "founded" in terms
    assert "founding" in terms


def test_score_text_matches_related_word_forms():
    founder_score = _score_text("founder of beforest", "Beforest was founded by Harsh in 2020.")
    overview_score = _score_text("beforest overview", "About Beforest: a regenerative lifestyle company.")

    assert founder_score > 0
    assert overview_score > 0


@patch("agents.beforest_tools._load_live_search_pages")
def test_search_beforest_experiences_filters_past_dated_results_for_current_queries(mock_pages):
    mock_pages.return_value = [
        {
            "host": "experiences.beforest.co",
            "title": "Past event",
            "text": "This is currently live for booking on Dec 14, 2000.",
            "url": "https://experiences.beforest.co/past-event",
        },
        {
            "host": "experiences.beforest.co",
            "title": "Upcoming event",
            "text": "This is currently live for booking on Jan 26, 2099.",
            "url": "https://experiences.beforest.co/upcoming-event",
        },
    ]

    results = search_beforest_experiences.invoke({"query": "What experiences are currently live?"})
    urls = [str(item.get("url", "")) for item in results if isinstance(item, dict)]

    assert "https://experiences.beforest.co/upcoming-event" in urls
    assert "https://experiences.beforest.co/past-event" not in urls


@patch("agents.beforest_tools._load_live_search_pages")
def test_search_beforest_experiences_returns_fallback_when_no_upcoming_dates(mock_pages):
    mock_pages.return_value = [
        {
            "host": "experiences.beforest.co",
            "title": "Past event only",
            "text": "Several experiences are currently live for booking: Dec 14, 2000.",
            "url": "https://experiences.beforest.co/past-only",
        }
    ]

    results = search_beforest_experiences.invoke({"query": "What experiences are currently live?"})

    assert len(results) == 1
    assert results[0]["source"] == "Live experiences status"
    assert results[0]["url"] == "https://experiences.beforest.co/"


@patch("agents.beforest_tools._load_live_search_pages")
def test_search_beforest_experiences_treats_next_query_as_fresh_date_request(mock_pages):
    mock_pages.return_value = [
        {
            "host": "experiences.beforest.co",
            "title": "Past event",
            "text": "Starry Nights happened on March 1, 2026.",
            "url": "https://experiences.beforest.co/starry-nights",
        }
    ]

    results = search_beforest_experiences.invoke({"query": "What is the next experience?"})

    assert len(results) == 1
    assert results[0]["source"] == "Live experiences status"
    assert results[0]["url"] == "https://experiences.beforest.co/"


@patch("agents.beforest_tools._load_live_search_pages")
def test_build_beforest_experiences_outline_markdown_keeps_only_future_entries(mock_pages):
    mock_pages.return_value = [
        {
            "host": "experiences.beforest.co",
            "title": "Past event",
            "text": "Starry Nights happened on March 1, 2020.",
            "markdown": "# Past event\n\nMarch 1, 2020",
            "url": "https://experiences.beforest.co/past",
        },
        {
            "host": "experiences.beforest.co",
            "title": "Future event",
            "text": "Coffee Safari returns on January 26, 2099.",
            "markdown": "# Future event\n\nJanuary 26, 2099",
            "url": "https://experiences.beforest.co/future",
        },
    ]

    markdown, entries = _build_beforest_experiences_outline_markdown(today=date(2026, 4, 1))

    assert "Future event" in markdown
    assert "https://experiences.beforest.co/future" in markdown
    assert "Past event" not in markdown
    assert len(entries) == 1


@patch("agents.beforest_tools._outline_request")
@patch("agents.beforest_tools._find_outline_document_by_title")
@patch("agents.beforest_tools._build_beforest_experiences_outline_markdown")
def test_sync_beforest_experiences_to_outline_updates_existing_doc(
    mock_build_markdown, mock_find_doc, mock_outline_request
):
    mock_build_markdown.return_value = (
        "# Beforest Experiences Feed\n\nFuture event",
        [{"title": "Future event", "url": "https://experiences.beforest.co/future", "dates": "January 26, 2099", "snippet": "Future event"}],
    )
    mock_find_doc.return_value = {"id": "doc-123", "title": "Beforest Experiences Feed"}
    mock_outline_request.return_value = {"data": {"id": "doc-123"}}

    with (
        patch("agents.beforest_tools.settings.OUTLINE_API_URL", "https://outline.example.com"),
        patch("agents.beforest_tools.settings.OUTLINE_API_TOKEN", SimpleNamespace(get_secret_value=lambda: "token")),
    ):
        result = _sync_beforest_experiences_to_outline()

    assert result["ok"] is True
    assert result["updated"] is True
    assert result["documentId"] == "doc-123"
    assert mock_outline_request.call_args.args[0] == "documents.update"
