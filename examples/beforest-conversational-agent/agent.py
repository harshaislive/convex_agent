import argparse
import json
import os
import re
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel

from tools import (
    browse_beforest_page,
    search_beforest_experiences,
    search_beforest_knowledge,
)

load_dotenv()

EXAMPLE_DIR = Path(__file__).parent.resolve()
BOT_LABEL = "Beforest"
console = Console()
URL_PATTERN = re.compile(r"https://[^\s)]+")
BUTTON_LIMIT = 3


def _sanitize_message_content(message: BaseMessage) -> BaseMessage:
    """Ensure every outbound message has non-empty content for Azure chat completions."""
    content = message.content
    if isinstance(content, str):
        if content != "":
            return message
        return message.model_copy(update={"content": " "})

    if len(content) > 0:
        return message

    return message.model_copy(update={"content": [{"type": "text", "text": " "}]})


class AzureMessageSanitizerMiddleware(AgentMiddleware):  # type: ignore[type-arg]
    """Normalize empty message content before requests hit Azure chat completions."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        sanitized_messages = [
            _sanitize_message_content(message) for message in request.messages
        ]
        return handler(request.override(messages=sanitized_messages))


def _resolve_thread_id(thread_id: str | None, user_id: str | None) -> str:
    """Resolve a stable thread id for DM-style conversations."""
    if thread_id:
        return thread_id
    if user_id:
        return user_id
    return f"beforest-{uuid.uuid4()}"


def _parse_subscriber_data(raw: str | None) -> dict[str, Any]:
    """Parse subscriber data JSON passed from an external DM system."""
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        msg = "--subscriber-data must be a JSON object."
        raise ValueError(msg)
    return parsed


def _build_context_messages(
    user_id: str | None, subscriber_data: dict[str, Any]
) -> list[dict[str, str]]:
    """Build a compact context message for platform metadata."""
    context: dict[str, Any] = {}
    if user_id:
        context["user_id"] = user_id
    if subscriber_data:
        context["subscriber_data"] = subscriber_data

    if not context:
        return []

    return [
        {
            "role": "system",
            "content": "Instagram DM context:\n"
            + json.dumps(context, ensure_ascii=True, separators=(",", ":")),
        }
    ]


def _build_knowledge_context_messages(question: str) -> list[dict[str, str]]:
    """Preload grounded Beforest knowledge so replies do not depend on tool choice alone."""
    try:
        results = search_beforest_knowledge.invoke(
            {
                "query": question,
                "max_results": 3,
            }
        )
    except Exception as exc:  # noqa: BLE001
        if os.getenv("DEBUG_KNOWLEDGE_ERRORS", "").strip().lower() == "true":
            print(f"[knowledge] preload failed: {exc}", flush=True)
        return []

    if not isinstance(results, list) or not results:
        return []

    lines = [
        "Beforest knowledge context from the source of truth.",
        "Use this context for factual answers. If it does not cover the answer, say that plainly.",
    ]
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "Unknown source"))
        snippet = str(item.get("snippet", "")).strip()
        if not snippet:
            continue
        lines.append(f"{index}. {source}: {snippet}")

    if len(lines) <= 2:
        return []

    return [{"role": "system", "content": "\n".join(lines)}]


def _resolve_manychat_subscriber_id(
    explicit_subscriber_id: str | None,
    subscriber_data: dict[str, Any],
) -> str | None:
    """Resolve a ManyChat subscriber/contact id from explicit input or subscriber data."""
    if explicit_subscriber_id:
        return explicit_subscriber_id

    for key in (
        "manychat_subscriber_id",
        "subscriber_id",
        "contact_id",
        "manychat_contact_id",
    ):
        value = subscriber_data.get(key)
        if value is None:
            continue
        return str(value)

    return None


def _normalize_manychat_text(reply_text: str) -> str:
    """Convert model output into cleaner DM-friendly text."""
    text = reply_text.replace("\r\n", "\n").strip()
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^\s*[-*]\s+", "• ", text)
    return text


def _extract_urls(text: str) -> list[str]:
    """Return unique URLs in first-seen order."""
    seen: set[str] = set()
    urls: list[str] = []
    for url in URL_PATTERN.findall(text):
        cleaned_url = url.rstrip(".,!?")
        if cleaned_url in seen:
            continue
        seen.add(cleaned_url)
        urls.append(cleaned_url)
    return urls


def _button_caption_for_url(url: str) -> str:
    """Build a short button caption from a Beforest URL."""
    lower_url = url.lower()
    if "experiences.beforest.co" in lower_url:
        return "Explore Experiences"
    if "/collect" in lower_url:
        return "View Collectives"
    if "/about" in lower_url:
        return "About Beforest"
    if "/contact" in lower_url or "/connect" in lower_url:
        return "Contact Team"
    if "beforest.co" in lower_url:
        return "Explore Beforest"
    return "Open Link"


def _build_manychat_buttons(urls: list[str]) -> list[dict[str, str]]:
    """Create a small set of URL buttons for Instagram DM replies."""
    buttons: list[dict[str, str]] = []
    for url in urls[:BUTTON_LIMIT]:
        buttons.append(
            {
                "type": "url",
                "caption": _button_caption_for_url(url),
                "url": url,
            }
        )
    return buttons


def _build_manychat_messages(question: str, reply_text: str) -> list[dict[str, Any]]:
    """Compose a DM-friendly message list with concise text and useful buttons."""
    normalized_text = _normalize_manychat_text(reply_text)
    discovered_urls = _extract_urls(normalized_text)

    message: dict[str, Any] = {
        "type": "text",
        "text": normalized_text,
    }
    buttons = _build_manychat_buttons(discovered_urls)
    if buttons:
        message["buttons"] = buttons

    return [message]


def _build_manychat_payload(reply_text: str, channel: str, question: str) -> dict[str, Any]:
    """Build a ManyChat `sendContent` payload for a text reply."""

    return {
        "version": "v2",
        "content": {
            "type": channel,
            "messages": _build_manychat_messages(question, reply_text),
            "actions": [],
            "quick_replies": [],
        },
    }


def _push_manychat_reply(subscriber_id: str, reply_text: str, question: str) -> None:
    """Send the final reply to ManyChat using `sendContent`.

    This expects the conversation to be within ManyChat's allowed messaging window.
    """
    token = os.getenv("MANYCHAT_API_TOKEN", "").strip()
    if not token:
        msg = "MANYCHAT_API_TOKEN is required to push replies to ManyChat."
        raise ValueError(msg)

    api_base_url = os.getenv(
        "MANYCHAT_API_BASE_URL", "https://api.manychat.com"
    ).rstrip("/")
    channel = os.getenv("MANYCHAT_CHANNEL", "instagram").strip().lower() or "instagram"
    payload = {
        "subscriber_id": int(subscriber_id),
        "data": _build_manychat_payload(reply_text, channel, question),
    }

    request = Request(
        f"{api_base_url}/fb/sending/sendContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlopen(request, timeout=15) as response:  # noqa: S310
        raw_response = response.read().decode("utf-8", errors="replace")

    parsed_response = json.loads(raw_response)
    if parsed_response.get("status") != "success":
        msg = f"ManyChat sendContent failed: {raw_response}"
        raise RuntimeError(msg)


def _get_convex_base() -> str | None:
    """Get Convex base URL from env."""
    url = os.getenv("CONVEX_HTTP_ACTION_URL", "").strip()
    if not url:
        return None
    return url.replace("/instagram/store-dm-event", "")


def _load_thread_history_from_convex(contact_id: str) -> list[dict[str, str]]:
    """Load conversation history from Convex for a ManyChat contact id."""
    base = _get_convex_base()
    if not base:
        return []
    secret = os.getenv("AGENT_SHARED_SECRET", "").strip()
    if not secret:
        return []

    url = f"{base}/instagram/conversation-history?contactId={contact_id}"
    request = Request(
        url,
        headers={"x-agent-secret": secret},
        method="GET",
    )
    try:
        with urlopen(request, timeout=10) as response:  # noqa: S310
            raw = response.read().decode("utf-8", errors="replace")
    except HTTPError:
        return []
    except OSError:
        return []

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def _save_event_to_convex(
    *,
    user_id: str | None,
    thread_id: str,
    subscriber_data: dict[str, Any],
    inbound_message: str,
    reply_text: str,
    manychat_subscriber_id: str | None,
) -> None:
    """Persist a DM event to Convex through a protected HTTP action."""
    convex_http_action_url = os.getenv("CONVEX_HTTP_ACTION_URL", "").strip()
    if not convex_http_action_url:
        return

    shared_secret = os.getenv("AGENT_SHARED_SECRET", "").strip()
    if not shared_secret:
        msg = "AGENT_SHARED_SECRET is required for Convex HTTP action writes."
        raise ValueError(msg)

    contact_id = manychat_subscriber_id
    if not contact_id:
        for key in (
            "contact_id",
            "contactId",
            "subscriber_id",
            "subscriberId",
            "manychat_subscriber_id",
        ):
            value = subscriber_data.get(key)
            if value is not None:
                contact_id = str(value)
                break
    if not contact_id:
        return

    now = datetime.now(timezone.utc).timestamp()

    def _optional_string(*keys: str) -> str | None:
        for key in keys:
            value = subscriber_data.get(key)
            if value is None or value == "":
                continue
            return str(value)
        return None

    def _optional_float(*keys: str) -> float | None:
        for key in keys:
            value = subscriber_data.get(key)
            if value is None or value == "":
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _optional_bool(*keys: str) -> bool | None:
        for key in keys:
            value = subscriber_data.get(key)
            if isinstance(value, bool):
                return value
        return None

    payload: dict[str, Any] = {
        "contactId": str(contact_id),
        "message": inbound_message,
        "receivedAt": now,
        "agentReplied": True,
        "agentReplyAt": now,
        "agentReplyText": reply_text,
        "lastReplyType": "agent",
        "rawPayload": {
            "userId": user_id,
            "manychatSubscriberId": manychat_subscriber_id,
            "subscriberData": subscriber_data,
        },
    }

    if (val := _optional_string("name", "first_name", "full_name")) is not None:
        payload["name"] = val
    if (val := _optional_string("instagram_user_id", "instagramUserId")) is not None:
        payload["instagramUserId"] = val
    elif user_id:
        payload["instagramUserId"] = user_id
    if (
        val := _optional_string(
            "instagram_account_name", "instagramAccountName", "account_name"
        )
    ) is not None:
        payload["instagramAccountName"] = val
    if (
        val := _optional_float(
            "ig_followers_count",
            "igFollowersCount",
            "followers_count",
            "follower_count",
            "ig_followers",
            "followers",
        )
    ) is not None:
        payload["igFollowersCount"] = val
    if (
        val := _optional_string("ig_messaging_window", "igMessagingWindow")
    ) is not None:
        payload["igMessagingWindow"] = val
    if (
        val := _optional_bool(
            "is_ig_account_follow_user", "isIgAccountFollowUser", "account_follows_user"
        )
    ) is not None:
        payload["isIgAccountFollowUser"] = val
    if (
        val := _optional_bool(
            "is_ig_account_follower",
            "isIgAccountFollower",
            "is_follower",
            "follows_account",
        )
    ) is not None:
        payload["isIgAccountFollower"] = val
    if (
        val := _optional_bool(
            "is_ig_verified_user", "isIgVerifiedUser", "is_verified", "verified"
        )
    ) is not None:
        payload["isIgVerifiedUser"] = val
    if (
        val := _optional_string("last_ig_interaction", "lastIgInteraction")
    ) is not None:
        payload["lastIgInteraction"] = val
    if (val := _optional_string("last_ig_seen", "lastIgSeen")) is not None:
        payload["lastIgSeen"] = val
    if (val := _optional_bool("optin_instagram", "optinInstagram")) is not None:
        payload["optinInstagram"] = val

    request = Request(
        convex_http_action_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-agent-secret": shared_secret,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=15) as response:  # noqa: S310
            raw_response = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        msg = f"Convex HTTP action failed with {exc.code}: {error_body}"
        raise RuntimeError(msg) from exc

    parsed_response = json.loads(raw_response)
    if parsed_response.get("ok") is not True:
        msg = f"Convex HTTP action returned ok=false: {raw_response}"
        raise RuntimeError(msg)


def _normalize_azure_ai_base_url(url: str) -> str:
    """Accept either Azure AI Foundry's preview endpoint or OpenAI-compatible base URL."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("AZURE_AI_BASE_URL must be a valid absolute URL.")

    if parsed.path.startswith("/models/chat/completions"):
        return f"{parsed.scheme}://{parsed.netloc}/openai/v1/"

    normalized_path = parsed.path.rstrip("/")
    if normalized_path.endswith("/openai/v1"):
        return f"{parsed.scheme}://{parsed.netloc}{normalized_path}/"

    return url


def _build_model():
    """Build the configured chat model."""
    provider = os.getenv("BEFOREST_MODEL_PROVIDER", "openai").strip().lower()

    if provider == "azure_ai":
        api_key = os.getenv("AZURE_AI_API_KEY")
        model_name = os.getenv("AZURE_AI_MODEL")
        base_url = _normalize_azure_ai_base_url(
            os.getenv(
                "AZURE_AI_BASE_URL",
                "https://azureuserbeforest.services.ai.azure.com/openai/v1/",
            )
        )

        missing = [
            name
            for name, value in {
                "AZURE_AI_API_KEY": api_key,
                "AZURE_AI_MODEL": model_name,
            }.items()
            if not value
        ]
        if missing:
            missing_vars = ", ".join(missing)
            raise ValueError(
                f"Azure AI provider selected but these environment variables are missing: {missing_vars}"
            )

        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=0.2,
        )

    model_name = os.getenv("BEFOREST_MODEL", "openai:gpt-4.1-mini")
    return init_chat_model(model_name, temperature=0.2)


def create_beforest_agent():
    """Create the Beforest concierge agent."""
    model = _build_model()

    return create_deep_agent(
        model=model,
        memory=[str(EXAMPLE_DIR / "AGENTS.md")],
        skills=[str(EXAMPLE_DIR / "skills")],
        middleware=[AzureMessageSanitizerMiddleware()],
        tools=[
            search_beforest_knowledge,
            search_beforest_experiences,
            browse_beforest_page,
        ],
        backend=FilesystemBackend(root_dir=EXAMPLE_DIR, virtual_mode=False),
        name="beforest-concierge",
    )


def generate_reply_bundle(
    question: str,
    *,
    thread_id: str | None = None,
    user_id: str | None = None,
    subscriber_data: dict[str, Any] | None = None,
    manychat_subscriber_id: str | None = None,
    push_to_manychat: bool = True,
) -> dict[str, Any]:
    """Generate a reply and structured metadata for DM clients."""
    agent = create_beforest_agent()
    resolved_thread_id = _resolve_thread_id(thread_id, user_id)
    resolved_manychat_subscriber_id = _resolve_manychat_subscriber_id(
        manychat_subscriber_id,
        subscriber_data or {},
    )
    context_messages = _build_context_messages(user_id, subscriber_data or {})
    knowledge_messages = _build_knowledge_context_messages(question)

    if resolved_manychat_subscriber_id:
        history_messages = _load_thread_history_from_convex(
            resolved_manychat_subscriber_id
        )
    else:
        history_messages = []

    result = agent.invoke(
        {
            "messages": [
                *history_messages,
                *context_messages,
                *knowledge_messages,
                {"role": "user", "content": question},
            ]
        },
        config={"configurable": {"thread_id": resolved_thread_id}},
    )
    final_message = result["messages"][-1]
    answer = str(getattr(final_message, "content", str(final_message)))
    if push_to_manychat and resolved_manychat_subscriber_id:
        _push_manychat_reply(resolved_manychat_subscriber_id, answer, question)
    _save_event_to_convex(
        user_id=user_id,
        thread_id=resolved_thread_id,
        subscriber_data=subscriber_data or {},
        inbound_message=question,
        reply_text=answer,
        manychat_subscriber_id=resolved_manychat_subscriber_id,
    )
    return {
        "reply": answer,
        "thread_id": resolved_thread_id,
    }


def generate_reply(
    question: str,
    *,
    thread_id: str | None = None,
    user_id: str | None = None,
    subscriber_data: dict[str, Any] | None = None,
    manychat_subscriber_id: str | None = None,
    push_to_manychat: bool = True,
) -> str:
    """Generate a reply, optionally pushing it back to ManyChat."""
    result = generate_reply_bundle(
        question,
        thread_id=thread_id,
        user_id=user_id,
        subscriber_data=subscriber_data,
        manychat_subscriber_id=manychat_subscriber_id,
        push_to_manychat=push_to_manychat,
    )
    return str(result["reply"])


def run_one_shot(
    question: str,
    *,
    thread_id: str | None = None,
    user_id: str | None = None,
    subscriber_data: dict[str, Any] | None = None,
    manychat_subscriber_id: str | None = None,
) -> None:
    """Run a single-turn interaction."""
    answer = generate_reply(
        question,
        thread_id=thread_id,
        user_id=user_id,
        subscriber_data=subscriber_data,
        manychat_subscriber_id=manychat_subscriber_id,
    )
    console.print(Panel(answer, title=BOT_LABEL, border_style="green"))


def run_interactive(
    *,
    thread_id: str | None = None,
    user_id: str | None = None,
    subscriber_data: dict[str, Any] | None = None,
    manychat_subscriber_id: str | None = None,
) -> None:
    """Run a small local chat loop for conversational testing."""
    agent = create_beforest_agent()
    resolved_thread_id = _resolve_thread_id(thread_id, user_id)
    messages: list[dict[str, str]] = _build_context_messages(
        user_id, subscriber_data or {}
    )

    console.print(Panel("Beforest is ready. Type `exit` to stop.", border_style="cyan"))

    while True:
        user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        knowledge_messages = _build_knowledge_context_messages(user_input)
        result = agent.invoke(
            {"messages": [*messages[:-1], *knowledge_messages, messages[-1]]},
            config={"configurable": {"thread_id": resolved_thread_id}},
        )
        final_message = result["messages"][-1]
        answer = getattr(final_message, "content", str(final_message))
        messages.append({"role": "assistant", "content": str(answer)})
        resolved_manychat_subscriber_id = _resolve_manychat_subscriber_id(
            manychat_subscriber_id,
            subscriber_data or {},
        )
        if resolved_manychat_subscriber_id:
            _push_manychat_reply(resolved_manychat_subscriber_id, str(answer))
        console.print(Panel(str(answer), title=BOT_LABEL, border_style="green"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Beforest conversational agent powered by deepagents."
    )
    parser.add_argument("question", nargs="*", help="Optional one-shot question")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run a local conversational chat loop",
    )
    parser.add_argument(
        "--thread-id",
        help="Stable conversation id. If omitted and --user-id is set, uses ig:<user_id>.",
    )
    parser.add_argument(
        "--user-id",
        help="External platform user id, recommended for Instagram DM session continuity.",
    )
    parser.add_argument(
        "--subscriber-data",
        help="Optional JSON object with external subscriber metadata.",
    )
    parser.add_argument(
        "--manychat-subscriber-id",
        help="ManyChat subscriber/contact id for pushing the final reply back via ManyChat API.",
    )
    args = parser.parse_args()
    subscriber_data = _parse_subscriber_data(args.subscriber_data)

    if args.interactive or not args.question:
        run_interactive(
            thread_id=args.thread_id,
            user_id=args.user_id,
            subscriber_data=subscriber_data,
            manychat_subscriber_id=args.manychat_subscriber_id,
        )
        return

    run_one_shot(
        " ".join(args.question),
        thread_id=args.thread_id,
        user_id=args.user_id,
        subscriber_data=subscriber_data,
        manychat_subscriber_id=args.manychat_subscriber_id,
    )


if __name__ == "__main__":
    main()
