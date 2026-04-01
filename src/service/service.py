import inspect
import json
import logging
import re
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from typing import Annotated, Any
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse  # type: ignore[import-untyped]
from langfuse.langchain import (
    CallbackHandler,  # type: ignore[import-untyped]
)
from langgraph.types import Command, Interrupt
from langsmith import Client as LangsmithClient
from langsmith import uuid7
from pydantic import BaseModel, Field

from agents import DEFAULT_AGENT, AgentGraph, get_agent, get_all_agent_info, load_agent
from agents.beforest_tools import get_knowledge_source_status
from core import settings
from memory import initialize_database, initialize_store
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
)
from service.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)


def custom_generate_unique_id(route: APIRoute) -> str:
    """Generate idiomatic operation IDs for OpenAPI client generation."""
    return route.name


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Configurable lifespan that initializes the appropriate database checkpointer, store,
    and agents with async loading - for example for starting up MCP clients.
    """
    try:
        # Initialize both checkpointer (for short-term memory) and store (for long-term memory)
        async with initialize_database() as saver, initialize_store() as store:
            # Set up both components
            if hasattr(saver, "setup"):  # ignore: union-attr
                await saver.setup()
            # Only setup store for Postgres as InMemoryStore doesn't need setup
            if hasattr(store, "setup"):  # ignore: union-attr
                await store.setup()

            # Configure agents with both memory components and async loading
            agents = get_all_agent_info()
            for a in agents:
                try:
                    await load_agent(a.key)
                    logger.info(f"Agent loaded: {a.key}")
                except Exception as e:
                    logger.error(f"Failed to load agent {a.key}: {e}")
                    # Continue with other agents rather than failing startup

                agent = get_agent(a.key)
                # Set checkpointer for thread-scoped memory (conversation history)
                agent.checkpointer = saver
                # Set store for long-term memory (cross-conversation knowledge)
                agent.store = store
            yield
    except Exception as e:
        logger.error(f"Error during database/store/agents initialization: {e}")
        raise


app = FastAPI(lifespan=lifespan, generate_unique_id_function=custom_generate_unique_id)
router = APIRouter(dependencies=[Depends(verify_bearer)])


class BeforestReplyRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: str | None = None
    thread_id: str | None = None
    subscriber_data: dict[str, Any] = Field(default_factory=dict)
    manychat_subscriber_id: str | None = None
    push_to_manychat: bool = False


class BeforestReplyResponse(BaseModel):
    ok: bool
    reply: str
    thread_id: str


BEFOREST_DM_TARGET_LIMIT = 220
BEFOREST_DM_MAX_SENTENCES = 2
BEFOREST_SESSION_AUTO_CLOSE_SECONDS = 30 * 60
BEFOREST_SESSION_CONTEXT_LOOKBACK_SECONDS = 24 * 60 * 60
_CURRENT_EXPERIENCE_QUERY_HINTS = (
    "current",
    "currently",
    "live",
    "right now",
    "now",
    "today",
    "upcoming",
    "available",
    "latest",
)
_BEFOREST_SESSION_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "creator": ("creator", "influencer", "coverage", "filmmaker", "reels", "content"),
    "partnership": ("partner", "partnership", "collaborate", "collaboration", "brand"),
    "event": ("retreat", "event", "workshop", "facilitator", "host", "offsite", "gathering"),
    "stay": ("stay", "hospitality", "book", "booking", "family", "weekend", "visit"),
    "experience": ("experience", "experiences", "retreat", "workshop"),
    "product": ("bewild", "produce", "product", "products", "shop", "coffee", "spices"),
    "collective": ("collective", "collectives", "hammiyala", "bhopal", "poomaale", "coorg"),
}
_BEFOREST_TRACKED_SESSION_TYPES = {"creator", "partnership", "event", "stay", "experience"}
_BEFOREST_CONFIRMATION_RE = re.compile(
    r"\b(thanks|thank you|got it|understood|works|perfect|great|cool|noted|that helps)\b",
    flags=re.IGNORECASE,
)
_BEFOREST_FOLLOW_UP_RE = re.compile(
    r"\?|"
    r"\b(share|tell me|which dates|what dates|what location|what city|which city|"
    r"what budget|what kind|let me know|send over|send me|feel free to share)\b",
    flags=re.IGNORECASE,
)
_BEFOREST_AMBIGUOUS_FOLLOW_UP_RE = re.compile(
    r"\b("
    r"yes|yeah|yep|ok|okay|sure|please|continue|following up|follow up|"
    r"what dates|which dates|what location|which location|where exactly|how much|"
    r"details|more info|more details|link|send link|timeline|budget"
    r")\b",
    flags=re.IGNORECASE,
)


@dataclass
class BeforestSessionState:
    session_id: str
    status: str
    session_type: str
    summary: str
    last_user_goal: str
    last_activity_at: float
    resolved_at: float | None = None
    closed_reason: str = ""
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
_MONTH_YEAR_RE = re.compile(
    r"\b("
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?"
    r")\s+(20\d{2})\b",
    flags=re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")


def _is_current_experiences_query(message: str) -> bool:
    lowered = message.lower()
    if "experience" not in lowered and "retreat" not in lowered and "workshop" not in lowered:
        return False
    return any(hint in lowered for hint in _CURRENT_EXPERIENCE_QUERY_HINTS)


def _extract_dates_from_text(text: str, *, today: date) -> list[date]:
    parsed_dates: list[date] = []
    seen_dates: set[date] = set()

    def add_date(parsed_date: date) -> None:
        if parsed_date in seen_dates:
            return
        seen_dates.add(parsed_date)
        parsed_dates.append(parsed_date)

    for match in _ISO_DATE_RE.finditer(text):
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            add_date(date(year, month, day))
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
        add_date(parsed_date)

    for match in _MONTH_YEAR_RE.finditer(text):
        month_key = match.group(1).lower()
        month = _MONTH_NAME_TO_NUMBER.get(month_key)
        if month is None:
            continue
        year = int(match.group(2))
        try:
            add_date(date(year, month, 1))
        except ValueError:
            continue
    return parsed_dates


def _enforce_current_experiences_freshness(message: str, reply_text: str) -> str:
    if not _is_current_experiences_query(message):
        return reply_text

    today = datetime.now(UTC).date()
    parsed_dates = _extract_dates_from_text(reply_text, today=today)
    if not parsed_dates:
        return reply_text
    if any(parsed_date < today for parsed_date in parsed_dates):
        return (
            "I can't confirm live experience dates here right now. "
            "Please check https://experiences.beforest.co for the latest upcoming listings."
        )

    return reply_text


def _clamp_beforest_dm_reply(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return text

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]
    if not sentences:
        sentences = [normalized]

    kept: list[str] = []
    for sentence in sentences:
        if len(kept) >= BEFOREST_DM_MAX_SENTENCES:
            break
        candidate = sentence if not kept else f"{' '.join(kept)} {sentence}"
        if len(candidate) <= BEFOREST_DM_TARGET_LIMIT:
            kept.append(sentence)
            continue

        if not kept:
            # If the first sentence is too long, try keeping meaningful clause chunks.
            clauses = [part.strip() for part in re.split(r"(?<=[,;:])\s+", sentence) if part.strip()]
            clause_kept = ""
            for clause in clauses:
                clause_candidate = clause if not clause_kept else f"{clause_kept} {clause}"
                if len(clause_candidate) <= BEFOREST_DM_TARGET_LIMIT:
                    clause_kept = clause_candidate
                else:
                    break
            if clause_kept and len(clause_kept) >= 80:
                kept.append(clause_kept.rstrip(" ,;:"))
        # Continue to allow later shorter sentences to fit instead of hard clipping immediately.
        continue

    if kept:
        return " ".join(kept)

    clipped = normalized[: BEFOREST_DM_TARGET_LIMIT - 1].rstrip()
    last_space = clipped.rfind(" ")
    if last_space > 80:
        clipped = clipped[:last_space]
    return clipped.rstrip(' ,;:') + "…"


def _short_session_text(text: str, *, limit: int = 140) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    clipped = normalized[: limit - 1].rstrip()
    last_space = clipped.rfind(" ")
    if last_space > 40:
        clipped = clipped[:last_space]
    return clipped.rstrip(" ,;:") + "…"


def _infer_beforest_session_type(
    message: str,
    previous_type: str = "",
    *,
    allow_previous_fallback: bool = True,
) -> str:
    lowered = message.lower()
    for session_type, keywords in _BEFOREST_SESSION_TYPE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return session_type
    if allow_previous_fallback and previous_type:
        return previous_type
    return "general"


def _message_is_confirmation(message: str) -> bool:
    return bool(_BEFOREST_CONFIRMATION_RE.search(message))


def _reply_needs_follow_up(reply_text: str) -> bool:
    return bool(_BEFOREST_FOLLOW_UP_RE.search(reply_text))


def _session_matches_message(session: BeforestSessionState, message: str) -> bool:
    inferred_type = _infer_beforest_session_type(message, allow_previous_fallback=False)
    if inferred_type != "general":
        return inferred_type == session.session_type
    if session.session_type == "general":
        return False
    if _message_is_confirmation(message):
        return True
    normalized = re.sub(r"\s+", " ", message).strip().lower()
    if len(normalized.split()) <= 6 and _BEFOREST_AMBIGUOUS_FOLLOW_UP_RE.search(normalized):
        return True
    return False


def _load_session_state_from_event(item: dict[str, Any]) -> BeforestSessionState | None:
    raw_payload = item.get("rawPayload", {})
    if not isinstance(raw_payload, dict):
        return None
    session_payload = raw_payload.get("session", {})
    if not isinstance(session_payload, dict):
        return None

    session_id = str(session_payload.get("session_id", "") or "").strip()
    status = str(session_payload.get("status", "") or "").strip()
    session_type = str(session_payload.get("session_type", "") or "").strip()
    summary = str(session_payload.get("summary", "") or "").strip()
    last_user_goal = str(session_payload.get("last_user_goal", "") or "").strip()
    if not session_id or not status or not session_type:
        return None

    last_activity_at_raw = session_payload.get("last_activity_at", 0.0)
    try:
        last_activity_at = float(last_activity_at_raw)
    except (TypeError, ValueError):
        return None

    resolved_at_raw = session_payload.get("resolved_at")
    try:
        resolved_at = float(resolved_at_raw) if resolved_at_raw is not None else None
    except (TypeError, ValueError):
        resolved_at = None

    return BeforestSessionState(
        session_id=session_id,
        status=status,
        session_type=session_type,
        summary=summary,
        last_user_goal=last_user_goal,
        last_activity_at=last_activity_at,
        resolved_at=resolved_at,
        closed_reason=str(session_payload.get("closed_reason", "") or "").strip(),
    )


def _derive_beforest_session_state(
    events: list[dict[str, Any]],
    *,
    current_message: str,
    now_ts: float,
) -> BeforestSessionState | None:
    latest_session: BeforestSessionState | None = None
    for item in reversed(events):
        if not isinstance(item, dict):
            continue
        latest_session = _load_session_state_from_event(item)
        if latest_session is not None:
            break

    if latest_session is None:
        return None
    if now_ts - latest_session.last_activity_at > BEFOREST_SESSION_CONTEXT_LOOKBACK_SECONDS:
        return None
    if (
        latest_session.status == "awaiting_confirmation"
        and now_ts - latest_session.last_activity_at > BEFOREST_SESSION_AUTO_CLOSE_SECONDS
    ):
        latest_session.status = "auto_closed"
        latest_session.closed_reason = "timeout"
    return latest_session


def _build_beforest_session_context(
    session: BeforestSessionState | None,
    *,
    current_message: str,
) -> SystemMessage | None:
    if session is None or not session.summary:
        return None
    if not _session_matches_message(session, current_message):
        return None

    lines = [
        "Beforest DM session context.",
        f"Previous session type: {session.session_type}.",
        f"Previous session status: {session.status}.",
        f"Previous session summary: {session.summary}",
    ]
    if session.status in {"open", "awaiting_confirmation"}:
        lines.append("If relevant, continue this thread naturally instead of restarting from scratch.")
    else:
        lines.append("If relevant, briefly reference the previous discussion before answering.")
    return SystemMessage(content="\n".join(lines))


def _should_continue_beforest_session(
    session: BeforestSessionState | None,
    *,
    current_message: str,
    now_ts: float,
) -> bool:
    if session is None:
        return False
    if session.status not in {"open", "awaiting_confirmation"}:
        return False
    if now_ts - session.last_activity_at > BEFOREST_SESSION_CONTEXT_LOOKBACK_SECONDS:
        return False
    return _session_matches_message(session, current_message)


def _build_beforest_session_summary(
    *,
    previous_session: BeforestSessionState | None,
    message: str,
    reply_text: str,
) -> str:
    current_summary = _short_session_text(
        f"User asked: {_short_session_text(message, limit=80)} "
        f"Agent replied: {_short_session_text(reply_text, limit=110)}",
        limit=180,
    )
    if previous_session and previous_session.summary:
        return _short_session_text(
            f"{previous_session.summary} Then {current_summary}",
            limit=180,
        )
    return current_summary


def _next_beforest_session_state(
    *,
    previous_session: BeforestSessionState | None,
    message: str,
    reply_text: str,
    now_ts: float,
) -> BeforestSessionState:
    inferred_type = _infer_beforest_session_type(
        message,
        previous_session.session_type if previous_session is not None else "",
    )
    continue_existing = (
        _should_continue_beforest_session(
            previous_session,
            current_message=message,
            now_ts=now_ts,
        )
    )
    session_id = previous_session.session_id if continue_existing else str(uuid4())

    if (
        previous_session is not None
        and previous_session.status == "awaiting_confirmation"
        and _session_matches_message(previous_session, message)
        and _message_is_confirmation(message)
    ):
        status = "solved"
        resolved_at = now_ts
        closed_reason = "user_confirmed"
    elif inferred_type in _BEFOREST_TRACKED_SESSION_TYPES:
        status = "open" if _reply_needs_follow_up(reply_text) else "awaiting_confirmation"
        resolved_at = None
        closed_reason = ""
    else:
        status = "solved"
        resolved_at = now_ts
        closed_reason = "faq_completed"

    return BeforestSessionState(
        session_id=session_id,
        status=status,
        session_type=inferred_type,
        summary=_build_beforest_session_summary(
            previous_session=previous_session if continue_existing else None,
            message=message,
            reply_text=reply_text,
        ),
        last_user_goal=_short_session_text(message, limit=90),
        last_activity_at=now_ts,
        resolved_at=resolved_at,
        closed_reason=closed_reason,
    )

def _convex_history_url() -> str | None:
    if settings.CONVEX_HTTP_ACTION_URL:
        return str(settings.CONVEX_HTTP_ACTION_URL).rstrip("/")
    if settings.CONVEX_SITE_URL:
        return str(settings.CONVEX_SITE_URL).rstrip("/") + "/instagram/store-dm-event"
    return None


def _convex_base_url() -> str | None:
    if settings.CONVEX_HTTP_ACTION_URL:
        return str(settings.CONVEX_HTTP_ACTION_URL).replace("/instagram/store-dm-event", "")
    if settings.CONVEX_SITE_URL:
        return str(settings.CONVEX_SITE_URL).rstrip("/")
    return None


async def _load_beforest_events_from_convex(contact_id: str) -> list[dict[str, Any]]:
    base_url = _convex_base_url()
    if not base_url or not settings.AGENT_SHARED_SECRET:
        return []

    history_url = f"{base_url}/instagram/conversation-history?contactId={contact_id}"
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            history_url,
            headers={"x-agent-secret": settings.AGENT_SHARED_SECRET.get_secret_value()},
        )
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _beforest_messages_from_events(events: list[dict[str, Any]]) -> list[HumanMessage | AIMessage]:
    messages: list[HumanMessage | AIMessage] = []
    for item in events:
        if not isinstance(item, dict):
            continue
        human_text = str(item.get("message", "") or "").strip()
        ai_text = str(item.get("agentReplyText", "") or "").strip()
        if human_text:
            messages.append(HumanMessage(content=human_text))
        if ai_text:
            messages.append(AIMessage(content=ai_text))
    return messages


async def _load_beforest_history_from_convex(contact_id: str) -> list[HumanMessage | AIMessage]:
    return _beforest_messages_from_events(await _load_beforest_events_from_convex(contact_id))


async def _save_beforest_event_to_convex(
    *,
    user_id: str | None,
    thread_id: str,
    subscriber_data: dict[str, Any],
    inbound_message: str,
    reply_text: str,
    manychat_subscriber_id: str | None,
    session_state: BeforestSessionState | None = None,
) -> None:
    convex_url = _convex_history_url()
    if not convex_url or not settings.AGENT_SHARED_SECRET:
        return

    contact_id = manychat_subscriber_id or user_id
    if not contact_id:
        for key in (
            "contact_id",
            "contactId",
            "subscriber_id",
            "subscriberId",
            "manychat_subscriber_id",
        ):
            value = subscriber_data.get(key)
            if value:
                contact_id = str(value)
                break
    if not contact_id:
        return

    now = datetime.now().timestamp()
    payload = {
        "contactId": str(contact_id),
        "message": inbound_message,
        "receivedAt": now,
        "agentReplied": True,
        "agentReplyAt": now,
        "agentReplyText": reply_text,
        "lastReplyType": "agent",
        "rawPayload": {
            "userId": user_id,
            "threadId": thread_id,
            "manychatSubscriberId": manychat_subscriber_id,
            "subscriberData": subscriber_data,
        },
    }
    if session_state is not None:
        payload["rawPayload"]["session"] = asdict(session_state)

    if user_id:
        payload["instagramUserId"] = user_id

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            convex_url,
            json=payload,
            headers={"x-agent-secret": settings.AGENT_SHARED_SECRET.get_secret_value()},
        )
        response.raise_for_status()


def _extract_urls(text: str) -> list[str]:
    return [match.rstrip(".,!?") for match in re.findall(r"https://[^\s)]+", text)]


def _remove_urls_from_text(text: str) -> str:
    cleaned = re.sub(r"\s*https://[^\s)]+", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or text


def _button_caption_for_url(url: str) -> str:
    lower_url = url.lower()
    if "experiences.beforest.co" in lower_url:
        return "Explore Experiences"
    if "hospitality.beforest.co" in lower_url:
        return "View Stays"
    if "10percent.beforest.co" in lower_url:
        return "Explore 10%"
    if "bewild.life" in lower_url:
        return "Explore Products"
    return "Explore Beforest"


MANYCHAT_MAX_TEXT_LENGTH = 640
MANYCHAT_MAX_MESSAGES = 10


def _split_manychat_text(text: str, *, limit: int = MANYCHAT_MAX_TEXT_LENGTH) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return [""]
    if len(normalized) <= limit:
        return [normalized]

    sentence_parts = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for sentence in sentence_parts:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= limit:
            current = candidate
            continue
        flush()
        if len(sentence) <= limit:
            current = sentence
            continue

        words = sentence.split()
        piece = ""
        for word in words:
            candidate = word if not piece else f"{piece} {word}"
            if len(candidate) <= limit:
                piece = candidate
            else:
                if piece:
                    chunks.append(piece)
                piece = word[:limit]
        if piece:
            current = piece
    flush()

    return chunks[:MANYCHAT_MAX_MESSAGES]


def _build_manychat_messages(reply_text: str, *, include_buttons: bool = True) -> list[dict[str, Any]]:
    buttons = [
        {"type": "url", "caption": _button_caption_for_url(url), "url": url}
        for url in _extract_urls(reply_text)[:3]
    ]
    base_text = _remove_urls_from_text(reply_text) if include_buttons and buttons else reply_text
    text_chunks = _split_manychat_text(base_text)
    messages: list[dict[str, Any]] = [{"type": "text", "text": chunk} for chunk in text_chunks]
    if include_buttons and buttons and messages:
        messages[-1]["buttons"] = buttons
    return messages


def _build_manychat_content(reply_text: str, *, include_buttons: bool = True) -> dict[str, Any]:
    return {
        "type": settings.MANYCHAT_CHANNEL,
        "messages": _build_manychat_messages(reply_text, include_buttons=include_buttons),
        "actions": [],
        "quick_replies": [],
    }


async def _post_manychat_content(
    client: httpx.AsyncClient,
    *,
    subscriber_id: str,
    content: dict[str, Any],
) -> httpx.Response:
    payload = {
        "subscriber_id": int(subscriber_id),
        "data": {
            "version": "v2",
            "content": content,
        },
    }

    return await client.post(
        f"{settings.MANYCHAT_API_BASE_URL.rstrip('/')}/fb/sending/sendContent",
        json=payload,
        headers={
            "Authorization": f"Bearer {settings.MANYCHAT_API_TOKEN.get_secret_value()}",
            "Content-Type": "application/json",
        },
    )


async def _push_manychat_reply(subscriber_id: str, reply_text: str) -> None:
    if not settings.MANYCHAT_API_TOKEN:
        return

    rich_content = _build_manychat_content(reply_text, include_buttons=True)
    plain_content = _build_manychat_content(reply_text, include_buttons=False)

    async with httpx.AsyncClient(timeout=15) as client:
        response = await _post_manychat_content(
            client,
            subscriber_id=subscriber_id,
            content=rich_content,
        )
        if response.is_success:
            return

        rich_error_body = response.text
        if response.status_code == 400 and rich_content != plain_content:
            fallback_response = await _post_manychat_content(
                client,
                subscriber_id=subscriber_id,
                content=plain_content,
            )
            if fallback_response.is_success:
                logger.warning(
                    "ManyChat rejected rich Beforest reply; delivered plain text fallback instead. "
                    "status=%s body=%s",
                    response.status_code,
                    rich_error_body,
                )
                return
            fallback_response.raise_for_status()

        response.raise_for_status()


@router.get("/info")
async def info() -> ServiceMetadata:
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )


@router.get("/health/knowledge")
async def knowledge_health() -> dict[str, object]:
    return get_knowledge_source_status()


@router.post("/beforest/reply")
async def beforest_reply(request: BeforestReplyRequest) -> BeforestReplyResponse:
    agent = get_agent("beforest-agent")
    now_ts = datetime.now().timestamp()
    contact_id = request.manychat_subscriber_id or request.user_id
    if not contact_id:
        for key in (
            "contact_id",
            "contactId",
            "subscriber_id",
            "subscriberId",
            "manychat_subscriber_id",
        ):
            value = request.subscriber_data.get(key)
            if value:
                contact_id = str(value)
                break

    user_input = UserInput(
        message=request.message,
        thread_id=request.thread_id,
        user_id=request.user_id,
        agent_config={"subscriber_data": request.subscriber_data},
    )
    kwargs, _ = await _handle_input(user_input, agent)
    history_events = await _load_beforest_events_from_convex(str(contact_id)) if contact_id else []
    history_messages = _beforest_messages_from_events(history_events)
    prior_session = _derive_beforest_session_state(
        history_events,
        current_message=request.message,
        now_ts=now_ts,
    )
    continue_existing_session = _should_continue_beforest_session(
        prior_session,
        current_message=request.message,
        now_ts=now_ts,
    )
    history_messages = (
        _beforest_messages_from_events(history_events) if continue_existing_session else []
    )
    session_context = _build_beforest_session_context(
        prior_session,
        current_message=request.message,
    )
    input_messages: list[HumanMessage | AIMessage | SystemMessage] = []
    if session_context is not None:
        input_messages.append(session_context)
    input_messages.extend(history_messages)
    input_messages.append(HumanMessage(content=request.message))
    kwargs["input"] = {"messages": input_messages}
    response_events: list[tuple[str, Any]] = await agent.ainvoke(
        **kwargs, stream_mode=["updates", "values"]
    )  # type: ignore[arg-type]
    response_type, response = response_events[-1]
    if response_type == "values":
        output = langchain_to_chat_message(response["messages"][-1])
    elif response_type == "updates" and "__interrupt__" in response:
        output = langchain_to_chat_message(AIMessage(content=response["__interrupt__"][0].value))
    else:
        raise HTTPException(status_code=500, detail="Unexpected response type")

    reply_text = _clamp_beforest_dm_reply(
        _enforce_current_experiences_freshness(request.message, output.content)
    )
    next_session = _next_beforest_session_state(
        previous_session=prior_session,
        message=request.message,
        reply_text=reply_text,
        now_ts=now_ts,
    )

    config_payload = kwargs.get("config", {})
    configurable = (
        config_payload.get("configurable", {})
        if isinstance(config_payload, dict)
        else getattr(config_payload, "configurable", {})
    )
    resolved_thread_id = request.thread_id or str(configurable.get("thread_id", ""))
    try:
        await _save_beforest_event_to_convex(
            user_id=request.user_id,
            thread_id=resolved_thread_id,
            subscriber_data=request.subscriber_data,
            inbound_message=request.message,
            reply_text=reply_text,
            manychat_subscriber_id=request.manychat_subscriber_id,
            session_state=next_session,
        )
    except Exception as exc:
        logger.error(f"Failed to write Beforest event to Convex: {exc}")

    if request.push_to_manychat and contact_id:
        try:
            await _push_manychat_reply(str(contact_id), reply_text)
        except Exception as exc:
            logger.error(f"Failed to push Beforest reply to ManyChat: {exc}")

    return BeforestReplyResponse(ok=True, reply=reply_text, thread_id=resolved_thread_id)


async def _handle_input(user_input: UserInput, agent: AgentGraph) -> tuple[dict[str, Any], UUID]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.
    """
    run_id = uuid7()
    thread_id = user_input.thread_id or str(uuid4())
    user_id = user_input.user_id or str(uuid4())

    configurable = {"thread_id": thread_id, "user_id": user_id}
    if user_input.model is not None:
        configurable["model"] = user_input.model

    callbacks: list[Any] = []
    if settings.LANGFUSE_TRACING:
        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        langfuse_handler = CallbackHandler()

        callbacks.append(langfuse_handler)

    if user_input.agent_config:
        # Check for reserved keys (including 'model' even if not in configurable)
        reserved_keys = {"thread_id", "user_id", "model"}
        if overlap := reserved_keys & user_input.agent_config.keys():
            raise HTTPException(
                status_code=422,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(user_input.agent_config)

    config = RunnableConfig(
        configurable=configurable,
        run_id=run_id,
        callbacks=callbacks,
    )

    # Check for interrupts that need to be resumed
    state = await agent.aget_state(config=config)
    interrupted_tasks = [
        task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
    ]

    input: Command | dict[str, Any]
    if interrupted_tasks:
        # assume user input is response to resume agent execution from interrupt
        input = Command(resume=user_input.message)
    else:
        input = {"messages": [HumanMessage(content=user_input.message)]}

    kwargs = {
        "input": input,
        "config": config,
    }

    return kwargs, run_id


@router.post("/{agent_id}/invoke", operation_id="invoke_with_agent_id")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.
    """
    # NOTE: Currently this only returns the last message or interrupt.
    # In the case of an agent outputting multiple AIMessages (such as the background step
    # in interrupt-agent, or a tool step in research-assistant), it's omitted. Arguably,
    # you'd want to include it. You could update the API to return a list of ChatMessages
    # in that case.
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent)

    try:
        response_events: list[tuple[str, Any]] = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])  # type: ignore # fmt: skip
        response_type, response = response_events[-1]
        if response_type == "values":
            # Normal response, the agent completed successfully
            output = langchain_to_chat_message(response["messages"][-1])
        elif response_type == "updates" and "__interrupt__" in response:
            # The last thing to occur was an interrupt
            # Return the value of the first interrupt as an AIMessage
            output = langchain_to_chat_message(
                AIMessage(content=response["__interrupt__"][0].value)
            )
        else:
            raise ValueError(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)
        return output
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent)

    try:
        # Process streamed events from the graph and yield messages over the SSE stream.
        async for stream_event in agent.astream(
            **kwargs, stream_mode=["updates", "messages", "custom"], subgraphs=True
        ):
            if not isinstance(stream_event, tuple):
                continue
            # Handle different stream event structures based on subgraphs
            if len(stream_event) == 3:
                # With subgraphs=True: (node_path, stream_mode, event)
                _, stream_mode, event = stream_event
            else:
                # Without subgraphs: (stream_mode, event)
                stream_mode, event = stream_event
            new_messages = []
            if stream_mode == "updates":
                for node, updates in event.items():
                    # A simple approach to handle agent interrupts.
                    # In a more sophisticated implementation, we could add
                    # some structured ChatMessage type to return the interrupt value.
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates:
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue
                    updates = updates or {}
                    update_messages = updates.get("messages", [])
                    # special cases for using langgraph-supervisor library
                    if "supervisor" in node or "sub-agent" in node:
                        # the only tools that come from the actual agent are the handoff and handback tools
                        if isinstance(update_messages[-1], ToolMessage):
                            if "sub-agent" in node and len(update_messages) > 1:
                                # If this is a sub-agent, we want to keep the last 2 messages - the handback tool, and it's result
                                update_messages = update_messages[-2:]
                            else:
                                # If this is a supervisor, we want to keep the last message only - the handoff result. The tool comes from the 'agent' node.
                                update_messages = [update_messages[-1]]
                        else:
                            update_messages = []
                    new_messages.extend(update_messages)

            if stream_mode == "custom":
                new_messages = [event]

            # LangGraph streaming may emit tuples: (field_name, field_value)
            # e.g. ('content', <str>), ('tool_calls', [ToolCall,...]), ('additional_kwargs', {...}), etc.
            # We accumulate only supported fields into `parts` and skip unsupported metadata.
            # More info at: https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/
            processed_messages = []
            current_message: dict[str, Any] = {}
            for message in new_messages:
                if isinstance(message, tuple):
                    key, value = message
                    # Store parts in temporary dict
                    current_message[key] = value
                else:
                    # Add complete message if we have one in progress
                    if current_message:
                        processed_messages.append(_create_ai_message(current_message))
                        current_message = {}
                    processed_messages.append(message)

            # Add any remaining message parts
            if current_message:
                processed_messages.append(_create_ai_message(current_message))

            for message in processed_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                    continue
                # LangGraph re-sends the input message, which feels weird, so drop it
                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            if stream_mode == "messages":
                if not user_input.stream_tokens:
                    continue
                msg, metadata = event
                if "skip_stream" in metadata.get("tags", []):
                    continue
                # For some reason, astream("messages") causes non-LLM nodes to send extra messages.
                # Drop them.
                if not isinstance(msg, AIMessageChunk):
                    continue
                content = remove_tool_calls(msg.content)
                if content:
                    # Empty content in the context of OpenAI usually means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content.
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
    except Exception as e:
        logger.error(f"Error in message generator: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'})}\n\n"
    finally:
        yield "data: [DONE]\n\n"


def _create_ai_message(parts: dict) -> AIMessage:
    sig = inspect.signature(AIMessage)
    valid_keys = set(sig.parameters)
    filtered = {k: v for k, v in parts.items() if k in valid_keys}
    return AIMessage(**filtered)


def _sse_response_example() -> dict[int | str, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
    operation_id="stream_with_agent_id",
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
    )


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    Record feedback for a run to LangSmith.

    This is a simple wrapper for the LangSmith create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return FeedbackResponse()


@router.post("/history")
async def history(input: ChatHistoryInput) -> ChatHistory:
    """
    Get chat history.
    """
    # TODO: Hard-coding DEFAULT_AGENT here is wonky
    agent: AgentGraph = get_agent(DEFAULT_AGENT)
    try:
        state_snapshot = await agent.aget_state(
            config=RunnableConfig(configurable={"thread_id": input.thread_id})
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


@app.get("/health")
async def health_check():
    """Health check endpoint."""

    health_status = {"status": "ok"}

    if settings.LANGFUSE_TRACING:
        try:
            langfuse = Langfuse()
            health_status["langfuse"] = "connected" if langfuse.auth_check() else "disconnected"
        except Exception as e:
            logger.error(f"Langfuse connection error: {e}")
            health_status["langfuse"] = "disconnected"

    return health_status


app.include_router(router)
