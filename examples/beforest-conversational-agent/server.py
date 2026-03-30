import os
import traceback
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from agent import generate_reply_bundle
from knowledge_center import (
    SESSION_COOKIE_NAME,
    import_url_entry,
    is_authenticated,
    list_entries,
    read_entry,
    render_knowledge_center_html,
    search_entries,
    save_entry,
    session_cookie_value,
    verify_password,
)

app = FastAPI(title="Beforest DM Agent", version="0.1.0")


class ReplyRequest(BaseModel):
    """Incoming request payload for a DM message."""

    message: str = Field(min_length=1)
    user_id: str | None = None
    thread_id: str | None = None
    subscriber_data: dict[str, Any] = Field(default_factory=dict)
    manychat_subscriber_id: str | None = None
    push_to_manychat: bool = True


class ReplyResponse(BaseModel):
    """Response payload returned by the API."""

    ok: bool
    reply: str
    thread_id: str


class KnowledgeCenterLoginRequest(BaseModel):
    password: str = Field(min_length=1)


class KnowledgeEntryRequest(BaseModel):
    slug: str = ""
    title: str = ""
    type: str = "fact"
    summary: str = ""
    body: str = Field(min_length=1)
    tags: str = ""
    intent_tags: str = ""
    audience_tags: str = ""
    priority: float = 0.5
    status: str = "draft"
    source_url: str = ""


class KnowledgeUrlRequest(BaseModel):
    url: str = Field(min_length=1)
    title: str = ""
    summary: str = ""
    type: str = "fact"
    tags: str = ""
    intent_tags: str = ""
    audience_tags: str = ""
    priority: float = 0.5
    status: str = "draft"
    cookie_header: str = ""
    auth_header: str = ""


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    intent: str = ""
    audience: str = ""
    max_results: int = 5


def _require_knowledge_auth(request: Request) -> None:
    if not is_authenticated(request.cookies.get(SESSION_COOKIE_NAME)):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health() -> dict[str, bool]:
    """Basic health check."""
    return {"ok": True}


@app.post("/reply", response_model=ReplyResponse)
def reply(request: ReplyRequest) -> ReplyResponse | JSONResponse:
    """Generate a reply and optionally push it to ManyChat."""
    try:
        result = generate_reply_bundle(
            request.message,
            thread_id=request.thread_id,
            user_id=request.user_id,
            subscriber_data=request.subscriber_data,
            manychat_subscriber_id=request.manychat_subscriber_id,
            push_to_manychat=request.push_to_manychat,
        )
    except Exception as exc:  # noqa: BLE001
        detail = str(exc)
        if os.getenv("DEBUG_REPLY_ERRORS", "").strip().lower() == "true":
            detail = traceback.format_exc()
        print(detail, flush=True)
        return JSONResponse(status_code=500, content={"detail": detail})

    resolved_thread_id = request.thread_id or (f"ig:{request.user_id}" if request.user_id else "")
    return ReplyResponse(
        ok=True,
        reply=str(result["reply"]),
        thread_id=str(result["thread_id"] or resolved_thread_id),
    )


@app.get("/knowledge-center", response_class=HTMLResponse)
def knowledge_center() -> str:
    """Render the private knowledge center UI."""
    return render_knowledge_center_html()


@app.post("/knowledge-center/login")
def knowledge_center_login(
    request: KnowledgeCenterLoginRequest,
    response: Response,
    raw_request: Request,
) -> dict[str, bool]:
    """Create a cookie-backed session for the knowledge center."""
    if not os.getenv("KNOWLEDGE_CENTER_PASSWORD", "").strip():
        raise HTTPException(
            status_code=503,
            detail="KNOWLEDGE_CENTER_PASSWORD is not configured",
        )
    if not verify_password(request.password):
        raise HTTPException(status_code=401, detail="Invalid password")
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_cookie_value(),
        httponly=True,
        samesite="lax",
        secure=raw_request.url.scheme == "https",
        max_age=60 * 60 * 12,
    )
    return {"ok": True}


@app.post("/knowledge-center/logout")
def knowledge_center_logout(response: Response) -> dict[str, bool]:
    """Clear the knowledge center auth cookie."""
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"ok": True}


@app.get("/knowledge-center/api/entries")
def knowledge_center_entries(request: Request) -> list[dict[str, Any]]:
    """List structured knowledge entries."""
    _require_knowledge_auth(request)
    return list_entries()


@app.get("/knowledge-center/api/entries/{slug}")
def knowledge_center_entry(slug: str, request: Request) -> dict[str, Any]:
    """Return one structured knowledge entry."""
    _require_knowledge_auth(request)
    try:
        entry = read_entry(slug)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Entry not found") from exc
    return {
        "slug": entry.slug,
        "title": entry.title,
        "type": entry.entry_type,
        "summary": entry.summary,
        "body": entry.body,
        "tags": entry.tags,
        "intent_tags": entry.intent_tags,
        "audience_tags": entry.audience_tags,
        "priority": entry.priority,
        "status": entry.status,
        "source_type": entry.source_type,
        "source_url": entry.source_url,
        "updated_at": entry.updated_at,
    }


@app.post("/knowledge-center/api/entries")
def knowledge_center_save_entry(
    payload: KnowledgeEntryRequest,
    request: Request,
) -> dict[str, Any]:
    """Create or update a structured knowledge entry."""
    _require_knowledge_auth(request)
    try:
        return save_entry(
            slug=payload.slug or None,
            title=payload.title,
            entry_type=payload.type,
            summary=payload.summary,
            body=payload.body,
            tags=payload.tags,
            intent_tags=payload.intent_tags,
            audience_tags=payload.audience_tags,
            priority=payload.priority,
            status=payload.status,
            source_url=payload.source_url or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/knowledge-center/api/import-url")
def knowledge_center_import_url(
    payload: KnowledgeUrlRequest,
    request: Request,
) -> dict[str, Any]:
    """Fetch a URL, extract markdown, and save it as a structured entry."""
    _require_knowledge_auth(request)
    try:
        return import_url_entry(
            url=payload.url,
            title=payload.title or None,
            summary=payload.summary or None,
            entry_type=payload.type,
            tags=payload.tags or None,
            intent_tags=payload.intent_tags or None,
            audience_tags=payload.audience_tags or None,
            priority=payload.priority,
            status=payload.status,
            cookie_header=payload.cookie_header or None,
            auth_header=payload.auth_header or None,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/knowledge-center/api/search")
def knowledge_center_search(
    payload: KnowledgeSearchRequest,
    request: Request,
) -> list[dict[str, Any]]:
    """Run a Convex-backed retrieval test for a draft query."""
    _require_knowledge_auth(request)
    try:
        return search_entries(
            query=payload.query,
            intent=payload.intent or None,
            audience=payload.audience or None,
            max_results=payload.max_results,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
