import os
import traceback
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent import generate_reply_bundle
from tools import get_knowledge_source_status

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


@app.get("/health")
def health() -> dict[str, bool]:
    """Basic health check."""
    return {"ok": True}


@app.get("/health/knowledge")
def knowledge_health() -> dict[str, object]:
    """Knowledge source diagnostics without exposing secrets."""
    return get_knowledge_source_status()


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
