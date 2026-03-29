import os
import traceback
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from agent import generate_reply_bundle
from knowledge_center import (
    SESSION_COOKIE_NAME,
    ingest_url_document,
    is_authenticated,
    list_documents,
    read_document,
    render_knowledge_center_html,
    save_markdown_document,
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


class KnowledgeMarkdownRequest(BaseModel):
    title: str = ""
    content: str = Field(min_length=1)
    tags: str = ""


class KnowledgeUrlRequest(BaseModel):
    url: str = Field(min_length=1)
    title: str = ""
    tags: str = ""
    cookie_header: str = ""
    auth_header: str = ""


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
def knowledge_center_login(request: KnowledgeCenterLoginRequest, response: Response) -> dict[str, bool]:
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
        secure=True,
        max_age=60 * 60 * 12,
    )
    return {"ok": True}


@app.post("/knowledge-center/logout")
def knowledge_center_logout(response: Response) -> dict[str, bool]:
    """Clear the knowledge center auth cookie."""
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"ok": True}


@app.get("/knowledge-center/api/documents")
def knowledge_center_documents(request: Request) -> list[dict[str, Any]]:
    """List stored knowledge documents."""
    _require_knowledge_auth(request)
    return list_documents()


@app.get("/knowledge-center/api/documents/{slug}")
def knowledge_center_document(slug: str, request: Request) -> dict[str, Any]:
    """Return one document's metadata and markdown."""
    _require_knowledge_auth(request)
    try:
        doc = read_document(slug)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    return {
        "slug": doc.slug,
        "title": doc.title,
        "content": doc.content,
        "source_type": doc.source_type,
        "tags": doc.tags,
        "source_url": doc.source_url,
        "updated_at": doc.updated_at,
    }


@app.post("/knowledge-center/api/markdown")
def knowledge_center_markdown(
    payload: KnowledgeMarkdownRequest,
    request: Request,
) -> dict[str, Any]:
    """Store a markdown document in the knowledge center."""
    _require_knowledge_auth(request)
    try:
        return save_markdown_document(payload.title, payload.content, payload.tags)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/knowledge-center/api/url")
def knowledge_center_url(
    payload: KnowledgeUrlRequest,
    request: Request,
) -> dict[str, Any]:
    """Fetch a URL, extract markdown, and save it as a knowledge document."""
    _require_knowledge_auth(request)
    try:
        return ingest_url_document(
            payload.url,
            title=payload.title or None,
            tags=payload.tags or None,
            cookie_header=payload.cookie_header or None,
            auth_header=payload.auth_header or None,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
