from types import SimpleNamespace
from unittest.mock import patch

from agents.beforest_tools import (
    _fetch_beforest_page,
    _is_allowed_beforest_url,
    _query_terms,
    _score_text,
    browse_beforest_page,
    fetch_beforest_markdown,
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