import hashlib
import inspect
import json
import logging
import re
import secrets
import urllib.parse
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

import httpx
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    Form,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
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
BEFOREST_OPS_COOKIE_NAME = "beforest_ops_session"
REPO_ROOT = Path(__file__).resolve().parents[2]
MEDIA_DIR = REPO_ROOT / "media"
BEFOREST_FAVICON_ICO_PATH = MEDIA_DIR / "favicon.ico"
BEFOREST_FAVICON_PNG_PATH = MEDIA_DIR / "beforest-favicon.png"
BEFOREST_OG_IMAGE_PATH = MEDIA_DIR / "beforest-og.jpg"


def custom_generate_unique_id(route: APIRoute) -> str:
    """Generate idiomatic operation IDs for OpenAPI client generation."""
    return route.name


def _beforest_ops_password_value() -> str | None:
    if not settings.BEFOREST_OPS_PASSWORD:
        return None
    return settings.BEFOREST_OPS_PASSWORD.get_secret_value().strip() or None


def _beforest_ops_cookie_value() -> str | None:
    password = _beforest_ops_password_value()
    if not password:
        return None
    secret_source = password
    if settings.AUTH_SECRET:
        secret_source += "|" + settings.AUTH_SECRET.get_secret_value()
    elif settings.AGENT_SHARED_SECRET:
        secret_source += "|" + settings.AGENT_SHARED_SECRET.get_secret_value()
    digest = hashlib.sha256(secret_source.encode("utf-8")).hexdigest()
    return digest


def _beforest_ops_authenticated(request: Request) -> bool:
    expected_cookie = _beforest_ops_cookie_value()
    if not expected_cookie:
        return False
    actual_cookie = str(request.cookies.get(BEFOREST_OPS_COOKIE_NAME, "") or "")
    return bool(actual_cookie) and secrets.compare_digest(actual_cookie, expected_cookie)


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _format_beforest_ops_timestamp(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "Unknown"
    return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M")


def _truncate_beforest_ops_text(value: Any, *, limit: int = 140) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _beforest_admin_status_payload_from_recent_conversations(
    recent_conversations: list[dict[str, Any]],
    *,
    contact_id: str,
) -> dict[str, Any] | None:
    if not contact_id:
        return None
    for item in recent_conversations:
        if str(item.get("contactId", "") or "").strip() != contact_id:
            continue
        return {
            "handover_status": item.get("handoverStatus", "bot"),
            "updated_at": item.get("receivedAt"),
            "updated_by": item.get("updatedBy", ""),
            "note": item.get("note", ""),
        }
    return None


def _render_beforest_admin_page(
    *,
    authenticated: bool,
    contact_id: str = "",
    search_query: str = "",
    status_payload: dict[str, Any] | None = None,
    recent_conversations: list[dict[str, Any]] | None = None,
    message: str = "",
    error: str = "",
    page_url: str = "",
    favicon_url: str = "/favicon.ico",
    og_image_url: str = "/og/beforest-og.jpg",
) -> str:
    safe_search_query = _escape_html(search_query)
    safe_message = _escape_html(message)
    safe_error = _escape_html(error)
    safe_page_url = _escape_html(page_url)
    safe_favicon_url = _escape_html(favicon_url)
    safe_og_image_url = _escape_html(og_image_url)
    conversation_rows = ""
    for item in recent_conversations or []:
        item_contact_id = str(item.get("contactId", "") or "").strip()
        if not item_contact_id:
            continue
        display_name = (
            str(item.get("name", "") or "").strip()
            or str(item.get("instagramAccountName", "") or "").strip()
            or "Unknown contact"
        )
        username = str(item.get("instagramAccountName", "") or "").strip()
        handover_status = str(item.get("handoverStatus", "bot") or "bot").strip().lower()
        status_class = handover_status if handover_status in {"human", "paused"} else "bot"
        preview = str(item.get("message", "") or "").strip() or str(item.get("agentReplyText", "") or "").strip()
        preview = _truncate_beforest_ops_text(preview or "No message preview yet.")
        timestamp_label = _format_beforest_ops_timestamp(item.get("receivedAt"))
        selected_class = " selected" if item_contact_id == contact_id else ""
        display_name_html = _escape_html(display_name)
        username_html = _escape_html(f"@{username}" if username else "")
        preview_html = _escape_html(preview)
        contact_id_html = _escape_html(item_contact_id)
        timestamp_html = _escape_html(timestamp_label)
        conversation_rows += f"""
        <form method="post" action="/admin/beforest/handover" class="conversation-row{selected_class}">
          <input type="hidden" name="contact_id" value="{contact_id_html}" />
          <input type="hidden" name="updated_by" value="ops" />
          <input type="hidden" name="note" value="" />
          <input type="hidden" name="q" value="{safe_search_query}" />
          <div class="row-main">
            <div class="row-name">{display_name_html}</div>
            <div class="row-meta">{username_html} <span class="dot">•</span> {contact_id_html}</div>
          </div>
          <div class="row-preview">{preview_html}</div>
          <div class="row-time">{timestamp_html}</div>
          <div class="row-toggles">
            <button class="toggle {'active' if status_class == 'bot' else ''}" data-status="bot" type="submit" name="status" value="bot">Bot</button>
            <button class="toggle {'active' if status_class == 'human' else ''}" data-status="human" type="submit" name="status" value="human">Human</button>
            <button class="toggle {'active' if status_class == 'paused' else ''}" data-status="paused" type="submit" name="status" value="paused">Pause</button>
          </div>
        </form>
        """
    if authenticated and not conversation_rows:
        conversation_rows = """
        <div class="empty-state">No conversations matched this search.</div>
        """
    login_markup = """
    <form method="post" action="/admin/beforest/login" class="stack">
      <label>Password
        <input type="password" name="password" placeholder="Enter admin password" />
      </label>
      <button type="submit">Unlock Ops</button>
    </form>
    """
    controls_markup = f"""
    <section class="panel stack">
      <div class="stack compact">
        <div>
          <h2>Inbox</h2>
          <p>Search contacts and switch bot ownership inline.</p>
        </div>
      </div>
      <form method="get" action="/admin/beforest" class="search-row" data-live-search="true">
        <input type="text" name="q" value="{safe_search_query}" placeholder="Search name, username, contact ID, or message" autocomplete="off" />
        <button class="secondary" type="submit">Search</button>
      </form>
      <div class="legend">
        <span><strong>Bot</strong>: replies automatically.</span>
        <span><strong>Human</strong>: teammate/founder owns the conversation, bot stays silent.</span>
        <span><strong>Pause</strong>: bot is muted temporarily without marking active human takeover.</span>
      </div>
      <div class="table-head">
        <span>Contact</span>
        <span>Last message</span>
        <span>Updated</span>
        <span>Status</span>
      </div>
      <div id="conversation-list" class="conversation-list">{conversation_rows}</div>
      <div id="ops-toast" class="toast" aria-live="polite"></div>
    </section>
    """

    body = f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Beforest Ops</title>
        <link rel="icon" href="{safe_favicon_url}" sizes="any" />
        <link rel="apple-touch-icon" href="{safe_favicon_url}" />
        <meta property="og:title" content="Beforest Ops" />
        <meta property="og:description" content="Beforest DM inbox and handover controls." />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="{safe_page_url}" />
        <meta property="og:image" content="{safe_og_image_url}" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="Beforest Ops" />
        <meta name="twitter:description" content="Beforest DM inbox and handover controls." />
        <meta name="twitter:image" content="{safe_og_image_url}" />
        <style>
          :root {{
            --bg: #f7f7f5;
            --panel: #ffffff;
            --ink: #191919;
            --muted: #6f6f6b;
            --line: #e6e6e2;
            --accent: #191919;
            --soft: #f1f1ef;
            --danger: #b42318;
          }}
          * {{ box-sizing: border-box; }}
          body {{
            margin: 0;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: var(--bg);
            color: var(--ink);
          }}
          .shell {{
            max-width: 1120px;
            margin: 28px auto;
            padding: 0 18px;
          }}
          .panel {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 18px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
          }}
          h1 {{ margin: 0 0 4px; font-size: 28px; font-weight: 650; }}
          h2 {{ margin: 0 0 4px; font-size: 16px; font-weight: 650; }}
          p, .meta, .note {{ color: var(--muted); line-height: 1.45; font-size: 14px; }}
          .stack {{ display: grid; gap: 12px; }}
          .stack.compact {{ gap: 8px; }}
          label {{ display: grid; gap: 6px; font-weight: 500; font-size: 13px; color: var(--muted); }}
          input, textarea {{
            width: 100%;
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 10px 12px;
            background: #fff;
            color: var(--ink);
            font: inherit;
          }}
          input:focus, textarea:focus {{ outline: none; border-color: #b8b8b1; box-shadow: 0 0 0 3px rgba(0,0,0,0.04); }}
          textarea {{ min-height: 84px; resize: vertical; }}
          .search-row {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; }}
          .legend {{
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            color: var(--muted);
            font-size: 12px;
            line-height: 1.45;
            padding: 2px 2px 6px;
          }}
          .legend strong {{ color: var(--ink); font-weight: 600; }}
          .table-head,
          .conversation-row {{
            display: grid;
            grid-template-columns: minmax(220px, 1.1fr) minmax(260px, 1.5fr) 140px 220px;
            gap: 12px;
            align-items: center;
          }}
          .table-head {{
            padding: 0 12px;
            color: var(--muted);
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
          }}
          .conversation-list {{ display: grid; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }}
          .conversation-row {{
            padding: 12px;
            border-top: 1px solid var(--line);
            background: #fff;
          }}
          .conversation-row:first-child {{ border-top: 0; }}
          .conversation-row:hover {{ background: #fafaf9; }}
          .conversation-row.selected {{ background: #f7f7f5; }}
          .row-main {{ min-width: 0; grid-area: main; }}
          .row-name {{ font-size: 14px; font-weight: 600; }}
          .row-meta,
          .row-time {{
            color: var(--muted);
            font-size: 12px;
          }}
          .row-time {{ grid-area: time; }}
          .dot {{ color: #c1c1bc; }}
          .row-preview {{
            font-size: 13px;
            line-height: 1.5;
            color: #2b2b28;
            min-width: 0;
            grid-area: preview;
          }}
          .row-toggles {{ display: flex; gap: 6px; justify-content: flex-end; grid-area: toggles; }}
          .toggle {{
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 6px 10px;
            background: #fff;
            color: var(--muted);
            font-size: 12px;
            font-weight: 600;
            transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
          }}
          .toggle.active {{
            color: var(--ink);
            border-color: #d2d2ce;
          }}
          .toggle.active[data-status="bot"] {{
            background: #f1f1ef;
            border-color: #d2d2ce;
          }}
          .toggle.active[data-status="human"] {{
            background: #fef3f2;
            border-color: #f0c5c0;
            color: #9f1f18;
          }}
          .toggle.active[data-status="paused"] {{
            background: #fffaeb;
            border-color: #efd9a7;
            color: #9a6700;
          }}
          button {{
            width: 100%;
            border: 0;
            border-radius: 10px;
            padding: 10px 12px;
            background: var(--accent);
            color: #fff;
            font: inherit;
            font-weight: 600;
            cursor: pointer;
          }}
          button.secondary {{ background: #3b3b38; }}
          .toggle {{
            width: auto;
            min-width: 0;
            flex: 0 0 auto;
          }}
          .banner, .error {{
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 14px;
            font-size: 13px;
          }}
          .banner {{ background: #f4f4f2; color: #3b3b38; }}
          .error {{ background: #fef3f2; color: var(--danger); }}
          .topbar {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }}
          form.inline {{ margin: 0; }}
          .empty-state {{ padding: 18px; color: var(--muted); font-size: 13px; background: #fff; }}
          .toast {{
            position: fixed;
            right: 18px;
            bottom: 18px;
            background: #191919;
            color: #fff;
            border-radius: 10px;
            padding: 10px 12px;
            font-size: 13px;
            opacity: 0;
            transform: translateY(6px);
            pointer-events: none;
            transition: opacity 120ms ease, transform 120ms ease;
          }}
          .toast.show {{
            opacity: 1;
            transform: translateY(0);
          }}
          @media (max-width: 900px) {{
            .table-head {{ display: none; }}
            .conversation-row {{
              grid-template-columns: minmax(0, 1fr) auto;
              grid-template-areas:
                "main toggles"
                "preview preview"
                "time time";
              gap: 8px 10px;
              align-items: start;
            }}
            .row-preview {{
              display: -webkit-box;
              -webkit-line-clamp: 2;
              -webkit-box-orient: vertical;
              overflow: hidden;
            }}
            .row-toggles {{ justify-content: flex-end; }}
            .toggle {{
              padding: 5px 9px;
              font-size: 11px;
            }}
          }}
          @media (max-width: 640px) {{
            .shell {{
              margin: 12px auto;
              padding: 0 10px;
            }}
            .panel {{
              padding: 12px;
              border-radius: 10px;
            }}
            h1 {{ font-size: 22px; }}
            h2 {{ font-size: 15px; }}
            .search-row {{ grid-template-columns: minmax(0, 1fr) auto; gap: 6px; }}
            .search-row button {{
              width: auto;
              padding: 10px;
              min-width: 72px;
            }}
            .legend {{
              gap: 6px;
              font-size: 11px;
              padding-bottom: 2px;
            }}
            .conversation-row {{
              padding: 10px;
              gap: 6px 8px;
            }}
            .row-name {{ font-size: 13px; }}
            .row-meta, .row-time, .row-preview {{ font-size: 11px; }}
            .row-toggles {{ gap: 4px; }}
            .toggle {{
              padding: 4px 8px;
              font-size: 10px;
            }}
            .toast {{
              left: 10px;
              right: 10px;
              bottom: 10px;
              font-size: 12px;
            }}
          }}
        </style>
        <script>
          document.addEventListener("DOMContentLoaded", function () {{
            const form = document.querySelector("form[data-live-search='true']");
            const toast = document.getElementById("ops-toast");
            let toastTimer = null;
            function showToast(message, isError) {{
              if (!toast) return;
              toast.textContent = message;
              toast.style.background = isError ? "#b42318" : "#191919";
              toast.classList.add("show");
              window.clearTimeout(toastTimer);
              toastTimer = window.setTimeout(function () {{
                toast.classList.remove("show");
              }}, 1800);
            }}

            async function bindRowForms(scope) {{
              scope.querySelectorAll("form.conversation-row").forEach(function (rowForm) {{
                if (rowForm.dataset.bound === "true") return;
                rowForm.dataset.bound = "true";
                rowForm.addEventListener("submit", async function (event) {{
                  event.preventDefault();
                  const submitter = event.submitter;
                  if (!submitter) return;
                  const formData = new FormData(rowForm);
                  formData.set("status", submitter.value);
                  try {{
                    const response = await fetch(rowForm.action, {{
                      method: "POST",
                      body: formData,
                      headers: {{
                        "x-requested-with": "fetch"
                      }}
                    }});
                    const payload = await response.json();
                    if (!response.ok || !payload.ok) {{
                      showToast(payload.error || "Could not update handover status.", true);
                      return;
                    }}
                    rowForm.querySelectorAll(".toggle").forEach(function (button) {{
                      button.classList.toggle("active", button.value === payload.handover_status);
                    }});
                    showToast("Status updated", false);
                  }} catch (_error) {{
                    showToast("Could not update handover status.", true);
                  }}
                }});
              }});
            }}

            async function refreshConversationList(query) {{
              const target = document.getElementById("conversation-list");
              if (!target) return;
              const url = new URL(window.location.href);
              url.searchParams.set("q", query);
              url.searchParams.delete("contact_id");
              url.searchParams.delete("message");
              url.searchParams.delete("error");
              try {{
                const response = await fetch(url.toString(), {{
                  headers: {{
                    "x-requested-with": "fetch"
                  }}
                }});
                if (!response.ok) {{
                  showToast("Could not refresh inbox.", true);
                  return;
                }}
                const html = await response.text();
                const doc = new DOMParser().parseFromString(html, "text/html");
                const nextList = doc.getElementById("conversation-list");
                if (!nextList) {{
                  showToast("Could not refresh inbox.", true);
                  return;
                }}
                target.innerHTML = nextList.innerHTML;
                window.history.replaceState({{}}, "", url.toString());
                bindRowForms(target);
              }} catch (_error) {{
                showToast("Could not refresh inbox.", true);
              }}
            }}

            if (form) {{
              const input = form.querySelector("input[name='q']");
              let timer = null;
              form.addEventListener("submit", async function (event) {{
                event.preventDefault();
                const currentInput = form.querySelector("input[name='q']");
                await refreshConversationList(currentInput ? currentInput.value : "");
              }});
              if (input) {{
                input.addEventListener("input", function () {{
                  window.clearTimeout(timer);
                  timer = window.setTimeout(function () {{
                    refreshConversationList(input.value);
                  }}, 220);
                }});
              }}
            }}

            bindRowForms(document);
          }});
        </script>
      </head>
      <body>
        <div class="shell">
          <div class="panel stack">
            <div class="topbar">
              <div>
                <h1>Beforest Ops</h1>
                <p>Minimal admin page for Instagram DM handover control.</p>
              </div>
              {"<form class='inline' method='post' action='/admin/beforest/logout'><button class='secondary' type='submit'>Logout</button></form>" if authenticated else ""}
            </div>
            {f"<div class='banner'>{safe_message}</div>" if safe_message else ""}
            {f"<div class='error'>{safe_error}</div>" if safe_error else ""}
            {controls_markup if authenticated else login_markup}
          </div>
        </div>
      </body>
    </html>
    """
    return body


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
    queued: bool = False
    suppressed: bool = False
    handover_status: str = "bot"


class BeforestHandoverRequest(BaseModel):
    status: Literal["bot", "human", "paused"]
    contact_id: str | None = None
    user_id: str | None = None
    manychat_subscriber_id: str | None = None
    thread_id: str | None = None
    updated_by: str | None = None
    note: str | None = None
    subscriber_data: dict[str, Any] = Field(default_factory=dict)


class BeforestHandoverResponse(BaseModel):
    ok: bool
    contact_id: str
    thread_id: str
    handover_status: str


class BeforestHandoverStatusResponse(BaseModel):
    ok: bool
    contact_id: str
    handover_status: str
    updated_at: float | None = None
    updated_by: str = ""
    note: str = ""


BEFOREST_DM_TARGET_LIMIT = 220
BEFOREST_DM_MAX_SENTENCES = 2
BEFOREST_SESSION_AUTO_CLOSE_SECONDS = 30 * 60
BEFOREST_SESSION_CONTEXT_LOOKBACK_SECONDS = 24 * 60 * 60
_CURRENT_EXPERIENCE_QUERY_HINTS = (
    "current",
    "currently",
    "live",
    "next",
    "next one",
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


@dataclass
class BeforestAutomationState:
    status: str
    updated_at: float
    updated_by: str = ""
    note: str = ""
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


def _load_automation_state_from_event(item: dict[str, Any]) -> BeforestAutomationState | None:
    raw_payload = item.get("rawPayload", {})
    if not isinstance(raw_payload, dict):
        return None
    automation_payload = raw_payload.get("automation", {})
    if not isinstance(automation_payload, dict):
        return None

    status = str(automation_payload.get("status", "") or "").strip()
    if status not in {"bot", "human", "paused"}:
        return None

    updated_at_raw = automation_payload.get("updated_at")
    try:
        updated_at = float(updated_at_raw)
    except (TypeError, ValueError):
        return None

    return BeforestAutomationState(
        status=status,
        updated_at=updated_at,
        updated_by=str(automation_payload.get("updated_by", "") or "").strip(),
        note=str(automation_payload.get("note", "") or "").strip(),
    )


def _derive_beforest_automation_state(events: list[dict[str, Any]]) -> BeforestAutomationState | None:
    for item in reversed(events):
        if not isinstance(item, dict):
            continue
        automation_state = _load_automation_state_from_event(item)
        if automation_state is not None:
            return automation_state
    return None


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


def _resolve_beforest_contact_id(request: BeforestReplyRequest) -> str | None:
    contact_id = request.manychat_subscriber_id or request.user_id
    if contact_id:
        return str(contact_id)

    for key in (
        "contact_id",
        "contactId",
        "subscriber_id",
        "subscriberId",
        "manychat_subscriber_id",
    ):
        value = request.subscriber_data.get(key)
        if value:
            return str(value)
    return None


def _resolve_beforest_manychat_subscriber_id(request: BeforestReplyRequest) -> str | None:
    if request.manychat_subscriber_id:
        return str(request.manychat_subscriber_id)

    for key in (
        "contact_id",
        "contactId",
        "subscriber_id",
        "subscriberId",
        "manychat_subscriber_id",
    ):
        value = request.subscriber_data.get(key)
        if value:
            return str(value)
    return None


def _resolve_beforest_handover_contact_id(request: BeforestHandoverRequest) -> str | None:
    if request.contact_id:
        return str(request.contact_id)
    if request.manychat_subscriber_id:
        return str(request.manychat_subscriber_id)
    if request.user_id:
        return str(request.user_id)
    for key in (
        "contact_id",
        "contactId",
        "subscriber_id",
        "subscriberId",
        "manychat_subscriber_id",
    ):
        value = request.subscriber_data.get(key)
        if value:
            return str(value)
    return None


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


async def _load_beforest_recent_conversations_from_convex(
    *, search_query: str = "", limit: int = 25
) -> list[dict[str, Any]]:
    base_url = _convex_base_url()
    if not base_url or not settings.AGENT_SHARED_SECRET:
        return []

    query = urllib.parse.urlencode(
        {"q": search_query, "limit": max(1, min(limit, 100))},
        doseq=False,
    )
    history_url = f"{base_url}/instagram/recent-conversations?{query}"
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
    automation_state: BeforestAutomationState | None = None,
    agent_replied: bool = True,
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
        "agentReplied": agent_replied,
        "lastReplyType": "agent" if agent_replied else "user",
        "rawPayload": {
            "userId": user_id,
            "threadId": thread_id,
            "manychatSubscriberId": manychat_subscriber_id,
            "subscriberData": subscriber_data,
        },
    }
    if agent_replied:
        payload["agentReplyAt"] = now
    if reply_text:
        payload["agentReplyText"] = reply_text
    if session_state is not None:
        payload["rawPayload"]["session"] = asdict(session_state)
    if automation_state is not None:
        payload["rawPayload"]["automation"] = asdict(automation_state)

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


def _tracking_value(value: Any) -> str:
    normalized = re.sub(r"\s+", "_", str(value or "").strip())
    normalized = re.sub(r"[^A-Za-z0-9_.-]", "", normalized)
    return normalized[:80]


def _enrich_typeform_url(
    url: str,
    *,
    subscriber_id: str | None = None,
    subscriber_data: dict[str, Any] | None = None,
) -> str:
    lower_url = url.lower()
    if "form.typeform.com" not in lower_url:
        return url

    subscriber_data = subscriber_data or {}
    username = _tracking_value(
        subscriber_data.get("username")
        or subscriber_data.get("instagram_username")
        or subscriber_data.get("contact_username")
    )
    ig_user_id = _tracking_value(
        subscriber_data.get("instagram_user_id")
        or subscriber_data.get("instagramUserId")
        or subscriber_data.get("user_id")
        or subscriber_data.get("userId")
    )
    contact_tracking_id = _tracking_value(
        subscriber_id
        or subscriber_data.get("contact_id")
        or subscriber_data.get("contactId")
        or subscriber_data.get("subscriber_id")
        or subscriber_data.get("subscriberId")
    )
    utm_content = username or contact_tracking_id or ig_user_id

    parsed = urllib.parse.urlsplit(url)
    query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    query["utm_source"] = "instagram"
    query["utm_medium"] = "dm_bot"
    query["utm_campaign"] = query.get("utm_campaign") or "beforest_collective_interest"
    if utm_content:
        query["utm_content"] = utm_content

    fragment = dict(urllib.parse.parse_qsl(parsed.fragment, keep_blank_values=True))
    if username:
        fragment["ig_username"] = username
    if contact_tracking_id:
        fragment["manychat_contact_id"] = contact_tracking_id
    if ig_user_id:
        fragment["ig_user_id"] = ig_user_id

    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urllib.parse.urlencode(query, doseq=True),
            urllib.parse.urlencode(fragment, doseq=True),
        )
    )


def _remove_urls_from_text(text: str) -> str:
    cleaned = re.sub(r"\s*https://[^\s)]+", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or text


def _button_caption_for_url(url: str) -> str:
    lower_url = url.lower()
    if "form.typeform.com" in lower_url:
        return "Show Interest"
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


def _build_manychat_messages(
    reply_text: str,
    *,
    include_buttons: bool = True,
    subscriber_id: str | None = None,
    subscriber_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    buttons = [
        {
            "type": "url",
            "caption": _button_caption_for_url(url),
            "url": _enrich_typeform_url(
                url,
                subscriber_id=subscriber_id,
                subscriber_data=subscriber_data,
            ),
        }
        for url in _extract_urls(reply_text)[:3]
    ]
    base_text = _remove_urls_from_text(reply_text) if include_buttons and buttons else reply_text
    text_chunks = _split_manychat_text(base_text)
    messages: list[dict[str, Any]] = [{"type": "text", "text": chunk} for chunk in text_chunks]
    if include_buttons and buttons and messages:
        messages[-1]["buttons"] = buttons
    return messages


def _build_manychat_content(
    reply_text: str,
    *,
    include_buttons: bool = True,
    subscriber_id: str | None = None,
    subscriber_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": settings.MANYCHAT_CHANNEL,
        "messages": _build_manychat_messages(
            reply_text,
            include_buttons=include_buttons,
            subscriber_id=subscriber_id,
            subscriber_data=subscriber_data,
        ),
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


async def _push_manychat_reply(
    subscriber_id: str,
    reply_text: str,
    *,
    subscriber_data: dict[str, Any] | None = None,
) -> None:
    if not settings.MANYCHAT_API_TOKEN:
        return

    rich_content = _build_manychat_content(
        reply_text,
        include_buttons=True,
        subscriber_id=subscriber_id,
        subscriber_data=subscriber_data,
    )
    plain_content = _build_manychat_content(
        reply_text,
        include_buttons=False,
        subscriber_id=subscriber_id,
        subscriber_data=subscriber_data,
    )

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


async def _generate_beforest_reply_text(
    request: BeforestReplyRequest,
    *,
    contact_id: str | None,
    resolved_thread_id: str,
    automation_state: BeforestAutomationState | None = None,
) -> str:
    agent = get_agent("beforest-agent")
    now_ts = datetime.now().timestamp()

    user_input = UserInput(
        message=request.message,
        thread_id=resolved_thread_id,
        user_id=request.user_id,
        agent_config={"subscriber_data": request.subscriber_data},
    )
    kwargs, _ = await _handle_input(user_input, agent)
    history_events = await _load_beforest_events_from_convex(str(contact_id)) if contact_id else []
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
    try:
        await _save_beforest_event_to_convex(
            user_id=request.user_id,
            thread_id=resolved_thread_id,
            subscriber_data=request.subscriber_data,
            inbound_message=request.message,
            reply_text=reply_text,
            manychat_subscriber_id=request.manychat_subscriber_id,
            session_state=next_session,
            automation_state=automation_state,
        )
    except Exception as exc:
        logger.error(f"Failed to write Beforest event to Convex: {exc}")

    return reply_text


async def _deliver_beforest_reply(
    request: BeforestReplyRequest,
    *,
    contact_id: str | None,
    resolved_thread_id: str,
    manychat_subscriber_id: str | None = None,
    automation_state: BeforestAutomationState | None = None,
) -> str:
    reply_text = await _generate_beforest_reply_text(
        request,
        contact_id=contact_id,
        resolved_thread_id=resolved_thread_id,
        automation_state=automation_state,
    )
    if manychat_subscriber_id:
        await _push_manychat_reply(
            manychat_subscriber_id,
            reply_text,
            subscriber_data=request.subscriber_data,
        )
    return reply_text


async def _deliver_beforest_reply_background(
    request: BeforestReplyRequest,
    *,
    contact_id: str | None,
    resolved_thread_id: str,
    manychat_subscriber_id: str,
    automation_state: BeforestAutomationState | None = None,
) -> None:
    try:
        await _deliver_beforest_reply(
            request,
            contact_id=contact_id,
            resolved_thread_id=resolved_thread_id,
            manychat_subscriber_id=manychat_subscriber_id,
            automation_state=automation_state,
        )
    except Exception as exc:
        logger.error(
            "Failed background Beforest reply delivery for thread_id=%s contact_id=%s: %s",
            resolved_thread_id,
            manychat_subscriber_id,
            exc,
        )


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


@app.get("/favicon.ico", include_in_schema=False)
async def beforest_favicon() -> FileResponse:
    return FileResponse(BEFOREST_FAVICON_ICO_PATH, media_type="image/x-icon")


@app.get("/og/beforest-og.jpg", include_in_schema=False)
async def beforest_og_image() -> FileResponse:
    return FileResponse(BEFOREST_OG_IMAGE_PATH, media_type="image/jpeg")


@app.get("/admin/beforest", response_class=HTMLResponse)
async def beforest_admin_page(
    request: Request,
    contact_id: str = "",
    q: str = "",
    message: str = "",
    error: str = "",
) -> HTMLResponse:
    base_url = str(request.base_url).rstrip("/")
    page_url = str(request.url)
    favicon_url = f"{base_url}/favicon.ico"
    og_image_url = f"{base_url}/og/beforest-og.jpg"
    if not _beforest_ops_authenticated(request):
        return HTMLResponse(
            _render_beforest_admin_page(
                authenticated=False,
                contact_id=contact_id,
                search_query=q,
                error=error,
                page_url=page_url,
                favicon_url=favicon_url,
                og_image_url=og_image_url,
            )
        )

    status_payload: dict[str, Any] | None = None
    recent_conversations: list[dict[str, Any]] = []
    try:
        recent_conversations = await _load_beforest_recent_conversations_from_convex(search_query=q)
    except Exception as exc:
        logger.warning("Failed loading Beforest recent conversations: %s", exc)
        error = error or "Inbox data could not be loaded from Convex."
    status_payload = _beforest_admin_status_payload_from_recent_conversations(
        recent_conversations,
        contact_id=contact_id,
    )
    if contact_id:
        try:
            history_events = await _load_beforest_events_from_convex(contact_id)
        except Exception as exc:
            logger.warning("Failed loading Beforest contact history for %s: %s", contact_id, exc)
            history_events = []
            error = error or "Selected contact details could not be loaded from Convex."
        if status_payload is None:
            automation_state = _derive_beforest_automation_state(history_events)
            status_payload = {
                "handover_status": automation_state.status if automation_state is not None else "bot",
                "updated_at": automation_state.updated_at if automation_state is not None else None,
                "updated_by": automation_state.updated_by if automation_state is not None else "",
                "note": automation_state.note if automation_state is not None else "",
            }
    return HTMLResponse(
        _render_beforest_admin_page(
            authenticated=True,
            contact_id=contact_id,
            search_query=q,
            status_payload=status_payload,
            recent_conversations=recent_conversations,
            message=message,
            error=error,
            page_url=page_url,
            favicon_url=favicon_url,
            og_image_url=og_image_url,
        )
    )


@app.post("/admin/beforest/login")
async def beforest_admin_login(password: str = Form(...)) -> RedirectResponse:
    expected_password = _beforest_ops_password_value()
    if not expected_password or not secrets.compare_digest(password, expected_password):
        return RedirectResponse("/admin/beforest?error=Incorrect+password", status_code=303)

    response = RedirectResponse("/admin/beforest", status_code=303)
    cookie_value = _beforest_ops_cookie_value()
    if cookie_value:
        response.set_cookie(
            BEFOREST_OPS_COOKIE_NAME,
            cookie_value,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=60 * 60 * 12,
        )
    return response


@app.post("/admin/beforest/logout")
async def beforest_admin_logout() -> RedirectResponse:
    response = RedirectResponse("/admin/beforest", status_code=303)
    response.delete_cookie(BEFOREST_OPS_COOKIE_NAME)
    return response


@app.post("/admin/beforest/handover", response_model=None)
async def beforest_admin_handover(
    request: Request,
    contact_id: str = Form(...),
    status_value: Literal["bot", "human", "paused"] = Form(..., alias="status"),
    updated_by: str = Form(""),
    note: str = Form(""),
    q: str = Form(""),
) -> RedirectResponse | JSONResponse:
    is_fetch_request = request.headers.get("x-requested-with", "").lower() == "fetch"
    if not _beforest_ops_authenticated(request):
        if is_fetch_request:
            return JSONResponse({"ok": False, "error": "Please login again."}, status_code=401)
        return RedirectResponse("/admin/beforest?error=Please+login+again", status_code=303)

    try:
        await beforest_handover(
            BeforestHandoverRequest(
                status=status_value,
                contact_id=contact_id,
                updated_by=updated_by or "ops",
                note=note or "",
            )
        )
    except Exception:
        logger.exception("Beforest admin handover failed for %s", contact_id)
        if is_fetch_request:
            return JSONResponse({"ok": False, "error": "Could not update handover status."}, status_code=500)
        query_params = {"contact_id": contact_id, "error": "Could not update handover status."}
        if q:
            query_params["q"] = q
        return RedirectResponse(
            f"/admin/beforest?{urllib.parse.urlencode(query_params)}",
            status_code=303,
        )
    if is_fetch_request:
        return JSONResponse(
            {
                "ok": True,
                "contact_id": contact_id,
                "handover_status": status_value,
                "message": f"Updated {contact_id} to {status_value}",
            }
        )
    success_message = f"Updated {contact_id} to {status_value}"
    query_params = {"contact_id": contact_id, "message": success_message}
    if q:
        query_params["q"] = q
    return RedirectResponse(
        f"/admin/beforest?{urllib.parse.urlencode(query_params)}",
        status_code=303,
    )


@router.post("/beforest/handover")
async def beforest_handover(request: BeforestHandoverRequest) -> BeforestHandoverResponse:
    contact_id = _resolve_beforest_handover_contact_id(request)
    if not contact_id:
        raise HTTPException(status_code=422, detail="contact_id or manychat_subscriber_id is required")

    resolved_thread_id = request.thread_id or str(uuid4())
    automation_state = BeforestAutomationState(
        status=request.status,
        updated_at=datetime.now().timestamp(),
        updated_by=request.updated_by or "",
        note=request.note or "",
    )
    await _save_beforest_event_to_convex(
        user_id=request.user_id,
        thread_id=resolved_thread_id,
        subscriber_data=request.subscriber_data,
        inbound_message="",
        reply_text="",
        manychat_subscriber_id=request.manychat_subscriber_id or contact_id,
        automation_state=automation_state,
        agent_replied=False,
    )
    return BeforestHandoverResponse(
        ok=True,
        contact_id=contact_id,
        thread_id=resolved_thread_id,
        handover_status=request.status,
    )


@router.get("/beforest/handover/{contact_id}")
async def beforest_handover_status(contact_id: str) -> BeforestHandoverStatusResponse:
    history_events = await _load_beforest_events_from_convex(contact_id)
    automation_state = _derive_beforest_automation_state(history_events)
    if automation_state is None:
        return BeforestHandoverStatusResponse(ok=True, contact_id=contact_id, handover_status="bot")
    return BeforestHandoverStatusResponse(
        ok=True,
        contact_id=contact_id,
        handover_status=automation_state.status,
        updated_at=automation_state.updated_at,
        updated_by=automation_state.updated_by,
        note=automation_state.note,
    )


@router.post("/beforest/reply")
async def beforest_reply(
    request: BeforestReplyRequest,
    background_tasks: BackgroundTasks,
) -> BeforestReplyResponse:
    contact_id = _resolve_beforest_contact_id(request)
    resolved_thread_id = request.thread_id or str(uuid4())
    history_events = await _load_beforest_events_from_convex(str(contact_id)) if contact_id else []
    automation_state = _derive_beforest_automation_state(history_events)
    handover_status = automation_state.status if automation_state is not None else "bot"

    if handover_status in {"human", "paused"}:
        try:
            await _save_beforest_event_to_convex(
                user_id=request.user_id,
                thread_id=resolved_thread_id,
                subscriber_data=request.subscriber_data,
                inbound_message=request.message,
                reply_text="",
                manychat_subscriber_id=request.manychat_subscriber_id,
                automation_state=automation_state,
                agent_replied=False,
            )
        except Exception as exc:
            logger.error(f"Failed to write suppressed Beforest event to Convex: {exc}")
        return BeforestReplyResponse(
            ok=True,
            reply="",
            thread_id=resolved_thread_id,
            queued=False,
            suppressed=True,
            handover_status=handover_status,
        )

    if request.push_to_manychat:
        manychat_subscriber_id = _resolve_beforest_manychat_subscriber_id(request)
        if not manychat_subscriber_id:
            raise HTTPException(
                status_code=422,
                detail="manychat_subscriber_id or subscriber_data.contact_id is required when push_to_manychat is true",
            )
        background_tasks.add_task(
            _deliver_beforest_reply_background,
            request.model_copy(deep=True),
            contact_id=contact_id,
            resolved_thread_id=resolved_thread_id,
            manychat_subscriber_id=manychat_subscriber_id,
            automation_state=automation_state,
        )
        return BeforestReplyResponse(
            ok=True,
            reply="",
            thread_id=resolved_thread_id,
            queued=True,
            handover_status=handover_status,
        )

    reply_text = await _deliver_beforest_reply(
        request,
        contact_id=contact_id,
        resolved_thread_id=resolved_thread_id,
        automation_state=automation_state,
    )
    return BeforestReplyResponse(
        ok=True,
        reply=reply_text,
        thread_id=resolved_thread_id,
        handover_status=handover_status,
    )


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
