import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import langsmith
import pytest
from fastapi import BackgroundTasks
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.pregel.types import StateSnapshot
from langgraph.types import Interrupt
from starlette.requests import Request

from agents.agents import Agent
from schema import ChatHistory, ChatMessage, ServiceMetadata
from schema.models import OpenAIModelName


def test_invoke(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.ainvoke.assert_awaited_once()
    input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
    assert input_message.content == QUESTION

    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER


def test_invoke_custom_agent(test_client, mock_agent) -> None:
    """Test that /invoke works with a custom agent_id path parameter."""
    CUSTOM_AGENT = "custom_agent"
    QUESTION = "What is the weather in Tokyo?"
    CUSTOM_ANSWER = "The weather in Tokyo is sunny."
    DEFAULT_ANSWER = "This is from the default agent."

    # Create a separate mock for the default agent
    default_mock = AsyncMock()
    default_mock.ainvoke.return_value = [
        ("values", {"messages": [AIMessage(content=DEFAULT_ANSWER)]})
    ]

    # Configure our custom mock agent
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=CUSTOM_ANSWER)]})]

    # Patch get_agent to return the correct agent based on the provided agent_id
    def agent_lookup(agent_id):
        if agent_id == CUSTOM_AGENT:
            return mock_agent
        return default_mock

    with patch("service.service.get_agent", side_effect=agent_lookup):
        response = test_client.post(f"/{CUSTOM_AGENT}/invoke", json={"message": QUESTION})
        assert response.status_code == 200

        # Verify custom agent was called and default wasn't
        mock_agent.ainvoke.assert_awaited_once()
        default_mock.ainvoke.assert_not_awaited()

        input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
        assert input_message.content == QUESTION

        output = ChatMessage.model_validate(response.json())
        assert output.type == "ai"
        assert output.content == CUSTOM_ANSWER  # Verify we got the custom agent's response


def test_invoke_model_param(test_client, mock_agent) -> None:
    """Test that the model parameter is correctly passed to the agent if specified."""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."
    CUSTOM_MODEL = "claude-sonnet-4-5"
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    response = test_client.post("/invoke", json={"message": QUESTION, "model": CUSTOM_MODEL})
    assert response.status_code == 200

    # Verify the model was passed correctly in the config
    mock_agent.ainvoke.assert_awaited_once()
    config = mock_agent.ainvoke.await_args.kwargs["config"]
    assert config["configurable"]["model"] == CUSTOM_MODEL

    # Verify the response is still correct
    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER

    # Verify an invalid model throws a validation error
    INVALID_MODEL = "gpt-7-notreal"
    response = test_client.post("/invoke", json={"message": QUESTION, "model": INVALID_MODEL})
    assert response.status_code == 422


def test_invoke_no_model_param_uses_none_default(test_client, mock_agent) -> None:
    """Test that when no model is specified, UserInput defaults to None and isn't passed to the runnable config (not hardcoded gpt-5-nano)."""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    # Don't specify model in the request
    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.ainvoke.assert_awaited_once()
    config = mock_agent.ainvoke.await_args.kwargs["config"]
    assert "model" not in config["configurable"]  # Should not be present when None

    # Verify the response is still correct
    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER


def test_invoke_custom_agent_config(test_client, mock_agent) -> None:
    """Test that the agent_config parameter is correctly passed to the agent."""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."
    CUSTOM_CONFIG = {"spicy_level": 0.1, "additional_param": "value_foo"}

    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    response = test_client.post(
        "/invoke", json={"message": QUESTION, "agent_config": CUSTOM_CONFIG}
    )
    assert response.status_code == 200

    # Verify the agent_config was passed correctly in the config
    mock_agent.ainvoke.assert_awaited_once()
    config = mock_agent.ainvoke.await_args.kwargs["config"]
    assert config["configurable"]["spicy_level"] == 0.1
    assert config["configurable"]["additional_param"] == "value_foo"

    # Verify the response is still correct
    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER

    # Verify a reserved key in agent_config throws a validation error
    INVALID_CONFIG = {"model": "gpt-5-nano"}
    response = test_client.post(
        "/invoke", json={"message": QUESTION, "agent_config": INVALID_CONFIG}
    )
    assert response.status_code == 422


def test_invoke_interrupt(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    INTERRUPT = "Confirm weather check"
    mock_agent.ainvoke.return_value = [
        ("values", {"messages": [AIMessage(content=ANSWER)]}),
        ("updates", {"__interrupt__": [Interrupt(value=INTERRUPT)]}),
    ]

    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.ainvoke.assert_awaited_once()
    input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
    assert input_message.content == QUESTION

    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == INTERRUPT


@patch("service.service.LangsmithClient")
def test_feedback(mock_client: langsmith.Client, test_client) -> None:
    ls_instance = mock_client.return_value
    ls_instance.create_feedback.return_value = None
    body = {
        "run_id": "847c6285-8fc9-4560-a83f-4e6285809254",
        "key": "human-feedback-stars",
        "score": 0.8,
    }
    response = test_client.post("/feedback", json=body)
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    ls_instance.create_feedback.assert_called_once_with(
        run_id="847c6285-8fc9-4560-a83f-4e6285809254",
        key="human-feedback-stars",
        score=0.8,
    )


def test_history(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    user_question = HumanMessage(content=QUESTION)
    agent_response = AIMessage(content=ANSWER)
    mock_agent.aget_state.return_value = StateSnapshot(
        values={"messages": [user_question, agent_response]},
        next=(),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )

    response = test_client.post(
        "/history", json={"thread_id": "7bcc7cc1-99d7-4b1d-bdb5-e6f90ed44de6"}
    )
    assert response.status_code == 200

    output = ChatHistory.model_validate(response.json())
    assert output.messages[0].type == "human"
    assert output.messages[0].content == QUESTION
    assert output.messages[1].type == "ai"
    assert output.messages[1].content == ANSWER


@pytest.mark.asyncio
async def test_stream(test_client, mock_agent) -> None:
    """Test streaming tokens and messages."""
    QUESTION = "What is the weather in Tokyo?"
    TOKENS = ["The", " weather", " in", " Tokyo", " is", " sunny", "."]
    FINAL_ANSWER = "The weather in Tokyo is sunny."

    # Configure mock to use our async iterator function
    events = [
        (
            "messages",
            (
                AIMessageChunk(content=token),
                {"tags": []},
            ),
        )
        for token in TOKENS
    ] + [
        (
            "updates",
            {"chat_model": {"messages": [AIMessage(content=FINAL_ANSWER)]}},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    # Make request with streaming
    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_tokens": True}
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and line.strip() != "data: [DONE]":  # Skip [DONE] message
                messages.append(json.loads(line.lstrip("data: ")))

        # Verify streamed tokens
        token_messages = [msg for msg in messages if msg["type"] == "token"]
        assert len(token_messages) == len(TOKENS)
        for i, msg in enumerate(token_messages):
            assert msg["content"] == TOKENS[i]

        # Verify final message
        final_messages = [msg for msg in messages if msg["type"] == "message"]
        assert len(final_messages) == 1
        assert final_messages[0]["content"]["content"] == FINAL_ANSWER
        assert final_messages[0]["content"]["type"] == "ai"


@pytest.mark.asyncio
async def test_stream_no_tokens(test_client, mock_agent) -> None:
    """Test streaming without tokens."""
    QUESTION = "What is the weather in Tokyo?"
    TOKENS = ["The", " weather", " in", " Tokyo", " is", " sunny", "."]
    FINAL_ANSWER = "The weather in Tokyo is sunny."

    # Configure mock to use our async iterator function
    events = [
        (
            "messages",
            (
                AIMessageChunk(content=token),
                {"tags": []},
            ),
        )
        for token in TOKENS
    ] + [
        (
            "updates",
            {"chat_model": {"messages": [AIMessage(content=FINAL_ANSWER)]}},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    # Make request with streaming disabled
    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_tokens": False}
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and line.strip() != "data: [DONE]":  # Skip [DONE] message
                messages.append(json.loads(line.lstrip("data: ")))

        # Verify no token messages
        token_messages = [msg for msg in messages if msg["type"] == "token"]
        assert len(token_messages) == 0

        # Verify final message
        assert len(messages) == 1
        assert messages[0]["type"] == "message"
        assert messages[0]["content"]["content"] == FINAL_ANSWER
        assert messages[0]["content"]["type"] == "ai"


def test_stream_interrupt(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    INTERRUPT = "Confirm weather check"
    # Configure mock to use our async iterator function
    events = [
        (
            "updates",
            {"__interrupt__": [Interrupt(value=INTERRUPT)]},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    # Make request with streaming disabled
    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_tokens": False}
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and line.strip() != "data: [DONE]":  # Skip [DONE] message
                messages.append(json.loads(line.lstrip("data: ")))

        # Verify interrupt message
        assert len(messages) == 1
        assert messages[0]["content"]["content"] == INTERRUPT
        assert messages[0]["content"]["type"] == "ai"


def test_info(test_client, mock_settings) -> None:
    """Test that /info returns the correct service metadata."""

    base_agent = Agent(description="A base agent.", graph_like=None)
    mock_settings.AUTH_SECRET = None
    mock_settings.DEFAULT_MODEL = OpenAIModelName.GPT_5_NANO
    mock_settings.AVAILABLE_MODELS = {OpenAIModelName.GPT_5_NANO, OpenAIModelName.GPT_5_MINI}
    with patch.dict("agents.agents.agents", {"base-agent": base_agent}, clear=True):
        response = test_client.get("/info")
        assert response.status_code == 200
        output = ServiceMetadata.model_validate(response.json())

    assert output.default_agent == "beforest-agent"
    assert len(output.agents) == 1
    assert output.agents[0].key == "base-agent"
    assert output.agents[0].description == "A base agent."

    assert output.default_model == OpenAIModelName.GPT_5_NANO
    assert output.models == [OpenAIModelName.GPT_5_MINI, OpenAIModelName.GPT_5_NANO]


def test_build_manychat_content_includes_buttons_for_urls() -> None:
    from service.service import _build_manychat_content

    content = _build_manychat_content(
        "Visit https://experiences.beforest.co/retreat and https://bewild.life/shop"
    )

    assert content["type"] == "instagram"
    assert content["messages"][0]["type"] == "text"
    assert content["messages"][0]["text"] == "Visit and"
    assert len(content["messages"][0]["buttons"]) == 2
    assert content["messages"][0]["buttons"][0]["caption"] == "Explore Experiences"
    assert content["messages"][0]["buttons"][1]["caption"] == "Explore Products"


def test_build_manychat_content_uses_show_interest_caption_for_typeform() -> None:
    from service.service import _build_manychat_content

    content = _build_manychat_content("Apply here https://form.typeform.com/to/hbDB2ybS")

    assert content["messages"][0]["buttons"][0]["caption"] == "Show Interest"


def test_build_manychat_content_enriches_typeform_tracking() -> None:
    from service.service import _build_manychat_content

    content = _build_manychat_content(
        "Apply here https://form.typeform.com/to/hbDB2ybS?utm_source=xxxxx&utm_medium=xxxxx&utm_content=xxxxx#current_page=xxxxx",
        subscriber_id="771052958",
        subscriber_data={"username": "harsha.live", "instagram_user_id": "ig-7710"},
    )

    url = content["messages"][0]["buttons"][0]["url"]
    assert "utm_source=instagram" in url
    assert "utm_medium=dm_bot" in url
    assert "utm_content=harsha.live" in url
    assert "ig_username=harsha.live" in url
    assert "manychat_contact_id=771052958" in url
    assert "ig_user_id=ig-7710" in url


@pytest.mark.asyncio
async def test_push_manychat_reply_retries_without_buttons_on_400() -> None:
    from service.service import _push_manychat_reply

    responses = [
        httpx.Response(400, content=b'{"error":"buttons rejected"}'),
        httpx.Response(200, content=b'{"status":"ok"}'),
    ]
    calls: list[dict] = []

    async def fake_post(url: str, **kwargs):
        calls.append({"url": url, **kwargs})
        return responses.pop(0)

    fake_client = SimpleNamespace(post=AsyncMock(side_effect=fake_post))

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return fake_client

        async def __aexit__(self, exc_type, exc, tb):
            return False

    fake_token = SimpleNamespace(get_secret_value=lambda: "token")

    with patch("service.service.settings") as mock_settings:
        mock_settings.MANYCHAT_API_TOKEN = fake_token
        mock_settings.MANYCHAT_API_BASE_URL = "https://api.manychat.com"
        mock_settings.MANYCHAT_CHANNEL = "instagram"
        with patch("service.service.httpx.AsyncClient", FakeAsyncClient):
            await _push_manychat_reply(
                "12345", "Visit https://experiences.beforest.co/retreat for details"
            )

    assert len(calls) == 2
    assert calls[0]["json"]["data"]["content"]["messages"][0]["text"] == "Visit for details"
    assert calls[0]["json"]["data"]["content"]["messages"][0]["buttons"]
    assert calls[1]["json"]["data"]["content"]["messages"][0]["text"] == "Visit https://experiences.beforest.co/retreat for details"
    assert "buttons" not in calls[1]["json"]["data"]["content"]["messages"][0]


def test_build_manychat_messages_splits_long_instagram_text() -> None:
    from service.service import _build_manychat_messages

    long_text = " ".join(["Beforest builds regenerative communities."] * 40)

    with patch("service.service.settings") as mock_settings:
        mock_settings.MANYCHAT_CHANNEL = "instagram"
        messages = _build_manychat_messages(long_text, include_buttons=False)

    assert len(messages) > 1
    assert len(messages) <= 10
    assert all(len(message["text"]) <= 640 for message in messages)
    assert all("buttons" not in message for message in messages)

def test_clamp_beforest_dm_reply_prefers_short_sentences() -> None:
    from service.service import _clamp_beforest_dm_reply

    text = (
        "Beforest is a regenerative lifestyle company. "
        "You can show interest in Hammiyala via beforest.co/contact or hello@beforest.co. "
        "If you want, I can share the direct link. "
        "Here is extra detail that should not be kept in an Instagram DM reply."
    )

    result = _clamp_beforest_dm_reply(text)

    assert "regenerative lifestyle company" in result
    assert "hello@beforest.co" in result or "beforest.co/contact" in result
    assert "extra detail" not in result
    assert len(result) <= 220


def test_clamp_beforest_dm_reply_skips_overlong_first_sentence_when_shorter_followup_exists() -> None:
    from service.service import _clamp_beforest_dm_reply

    long_sentence = " ".join(["Beforest builds regenerative communities"] * 30) + "."
    text = f"{long_sentence} Check https://experiences.beforest.co for latest experiences."

    result = _clamp_beforest_dm_reply(text)

    assert "https://experiences.beforest.co" in result
    assert len(result) <= 220
    assert result.endswith("...") is False
    assert result.endswith("…") is False


def test_enforce_current_experiences_freshness_rewrites_past_live_claim() -> None:
    from service.service import _enforce_current_experiences_freshness

    message = "What experiences are currently live?"
    stale_reply = (
        "Several immersive experiences are currently live for booking: "
        "Starry Nights (Dec 14, 2025), Family Roots (Jan 4, 2026), "
        "Coffee Safari (Jan 26, 2026)."
    )

    result = _enforce_current_experiences_freshness(message, stale_reply)

    assert "can't confirm live experience dates" in result
    assert "https://experiences.beforest.co" in result
    assert "Dec 14, 2025" not in result


def test_enforce_current_experiences_freshness_keeps_future_live_claim() -> None:
    from service.service import _enforce_current_experiences_freshness

    message = "What experiences are currently live?"
    valid_reply = "An upcoming experience is currently live for booking on Jan 26, 2099."

    result = _enforce_current_experiences_freshness(message, valid_reply)

    assert result == valid_reply


def test_enforce_current_experiences_freshness_rewrites_stale_upcoming_dates() -> None:
    from service.service import _enforce_current_experiences_freshness

    message = "Show me current Beforest experiences."
    stale_upcoming_reply = (
        "Our upcoming experiences include Starry Nights in Hyderabad (Dec 14, 2025), "
        "Family Roots in Coorg (Jan 4, 2026), and Hands & Soil in Hyderabad (Feb 14, 2026)."
    )

    result = _enforce_current_experiences_freshness(message, stale_upcoming_reply)

    assert "can't confirm live experience dates" in result
    assert "https://experiences.beforest.co" in result
    assert "Dec 14, 2025" not in result


def test_enforce_current_experiences_freshness_rewrites_mixed_stale_and_future_dates() -> None:
    from service.service import _enforce_current_experiences_freshness

    message = "Are there any upcoming retreats or workshops?"
    mixed_reply = (
        "Upcoming options include Hands & Soil (Feb 14, 2026) and "
        "SloMo retreat (May 1, 2099)."
    )

    result = _enforce_current_experiences_freshness(message, mixed_reply)

    assert "can't confirm live experience dates" in result
    assert "https://experiences.beforest.co" in result


def test_enforce_current_experiences_freshness_rewrites_stale_month_year_date() -> None:
    from service.service import _enforce_current_experiences_freshness

    message = "Are there any Coorg experiences live right now?"
    stale_month_year_reply = (
        "No Coorg experiences are live right now. "
        "Check upcoming ones like Coffee Safari in January 2026 at "
        "https://experiences.beforest.co."
    )

    result = _enforce_current_experiences_freshness(message, stale_month_year_reply)

    assert "can't confirm live experience dates" in result
    assert "https://experiences.beforest.co" in result
    assert "January 2026" not in result


def test_enforce_current_experiences_freshness_rewrites_stale_next_experience_date() -> None:
    from service.service import _enforce_current_experiences_freshness

    message = "What is the next experience?"
    stale_next_reply = "The next experience is Starry Nights on March 1, 2026."

    result = _enforce_current_experiences_freshness(message, stale_next_reply)

    assert "can't confirm live experience dates" in result
    assert "https://experiences.beforest.co" in result
    assert "March 1, 2026" not in result


def test_derive_beforest_session_state_auto_closes_stale_confirmation() -> None:
    from service.service import _derive_beforest_session_state

    events = [
        {
            "rawPayload": {
                "session": {
                    "session_id": "sess-1",
                    "status": "awaiting_confirmation",
                    "session_type": "creator",
                    "summary": "Creator collaboration ask for Coorg stay.",
                    "last_user_goal": "creator collab",
                    "last_activity_at": 0,
                    "resolved_at": None,
                    "closed_reason": "",
                }
            }
        }
    ]

    result = _derive_beforest_session_state(
        events,
        current_message="Following up on the creator collab",
        now_ts=31 * 60,
    )

    assert result is not None
    assert result.status == "auto_closed"
    assert result.closed_reason == "timeout"


def test_derive_beforest_automation_state_reads_latest_status() -> None:
    from service.service import _derive_beforest_automation_state

    events = [
        {
            "rawPayload": {
                "automation": {
                    "status": "bot",
                    "updated_at": 10,
                    "updated_by": "system",
                    "note": "",
                }
            }
        },
        {
            "rawPayload": {
                "automation": {
                    "status": "human",
                    "updated_at": 20,
                    "updated_by": "ops",
                    "note": "Team taking over creator lead",
                }
            }
        },
    ]

    result = _derive_beforest_automation_state(events)

    assert result is not None
    assert result.status == "human"
    assert result.updated_by == "ops"
    assert "creator lead" in result.note


def test_beforest_ops_cookie_auth_roundtrip() -> None:
    import service.service as svc

    fake_secret = SimpleNamespace(get_secret_value=lambda: "secret")
    with (
        patch.object(svc.settings, "BEFOREST_OPS_PASSWORD", fake_secret),
        patch.object(svc.settings, "AUTH_SECRET", None),
        patch.object(svc.settings, "AGENT_SHARED_SECRET", None),
    ):
        cookie_value = svc._beforest_ops_cookie_value()
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/admin/beforest",
                "headers": [
                    (b"cookie", f"{svc.BEFOREST_OPS_COOKIE_NAME}={cookie_value}".encode())
                ],
            }
        )

        assert cookie_value is not None
        assert svc._beforest_ops_authenticated(request) is True


@pytest.mark.asyncio
async def test_beforest_admin_page_renders_login_form() -> None:
    import service.service as svc

    fake_secret = SimpleNamespace(get_secret_value=lambda: "secret")
    request = Request({"type": "http", "method": "GET", "path": "/admin/beforest", "headers": []})
    with patch.object(svc.settings, "BEFOREST_OPS_PASSWORD", fake_secret):
        response = await svc.beforest_admin_page(request)

    body = response.body.decode("utf-8")
    assert response.status_code == 200
    assert "Beforest Ops" in body
    assert "Enter admin password" in body
    assert 'property="og:image"' in body
    assert 'rel="icon"' in body


@pytest.mark.asyncio
@patch("service.service._load_beforest_events_from_convex", new_callable=AsyncMock)
@patch("service.service._load_beforest_recent_conversations_from_convex", new_callable=AsyncMock)
async def test_beforest_admin_page_renders_recent_conversations(
    mock_recent_conversations, mock_load_events
) -> None:
    import service.service as svc

    fake_secret = SimpleNamespace(get_secret_value=lambda: "secret")
    with (
        patch.object(svc.settings, "BEFOREST_OPS_PASSWORD", fake_secret),
        patch.object(svc.settings, "AUTH_SECRET", None),
        patch.object(svc.settings, "AGENT_SHARED_SECRET", None),
    ):
        cookie_value = svc._beforest_ops_cookie_value()
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/admin/beforest",
                "query_string": b"q=poomaale&contact_id=12345",
                "headers": [
                    (b"cookie", f"{svc.BEFOREST_OPS_COOKIE_NAME}={cookie_value}".encode())
                ],
            }
        )
        mock_recent_conversations.return_value = [
            {
                "contactId": "12345",
                "name": "Aditi",
                "instagramAccountName": "aditi.travels",
                "message": "Can we collaborate on a Beforest stay?",
                "receivedAt": 1_711_234_567.0,
                "handoverStatus": "human",
                "note": "Founder took over",
            }
        ]
        mock_load_events.return_value = []
        response = await svc.beforest_admin_page(request, contact_id="12345", q="poomaale")

    body = response.body.decode("utf-8")
    assert response.status_code == 200
    assert "Search name, username, contact ID, or message" in body
    assert "Aditi" in body
    assert "aditi.travels" in body
    assert ">Bot<" in body
    assert ">Human<" in body
    assert ">Pause<" in body


@pytest.mark.asyncio
async def test_beforest_admin_login_sets_cookie() -> None:
    import service.service as svc

    fake_secret = SimpleNamespace(get_secret_value=lambda: "secret")
    with (
        patch.object(svc.settings, "BEFOREST_OPS_PASSWORD", fake_secret),
        patch.object(svc.settings, "AUTH_SECRET", None),
        patch.object(svc.settings, "AGENT_SHARED_SECRET", None),
    ):
        response = await svc.beforest_admin_login(password="secret")

    assert response.status_code == 303
    assert response.headers["location"] == "/admin/beforest"
    assert svc.BEFOREST_OPS_COOKIE_NAME in response.headers.get("set-cookie", "")


@pytest.mark.asyncio
@patch("service.service.beforest_handover", new_callable=AsyncMock)
async def test_beforest_admin_handover_uses_existing_handover_flow(mock_beforest_handover) -> None:
    import service.service as svc

    fake_secret = SimpleNamespace(get_secret_value=lambda: "secret")
    with (
        patch.object(svc.settings, "BEFOREST_OPS_PASSWORD", fake_secret),
        patch.object(svc.settings, "AUTH_SECRET", None),
        patch.object(svc.settings, "AGENT_SHARED_SECRET", None),
    ):
        cookie_value = svc._beforest_ops_cookie_value()
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/admin/beforest/handover",
                "headers": [
                    (b"cookie", f"{svc.BEFOREST_OPS_COOKIE_NAME}={cookie_value}".encode())
                ],
            }
        )
        response = await svc.beforest_admin_handover(
            request,
            contact_id="12345",
            status_value="human",
            updated_by="founder",
            note="Taking over founder thread",
        )

    assert response.status_code == 303
    assert "contact_id=12345" in response.headers["location"]
    handover_request = mock_beforest_handover.await_args.args[0]
    assert handover_request.contact_id == "12345"
    assert handover_request.status == "human"


@pytest.mark.asyncio
@patch("service.service.beforest_handover", new_callable=AsyncMock)
async def test_beforest_admin_handover_returns_json_for_fetch_requests(mock_beforest_handover) -> None:
    import service.service as svc

    fake_secret = SimpleNamespace(get_secret_value=lambda: "secret")
    with (
        patch.object(svc.settings, "BEFOREST_OPS_PASSWORD", fake_secret),
        patch.object(svc.settings, "AUTH_SECRET", None),
        patch.object(svc.settings, "AGENT_SHARED_SECRET", None),
    ):
        cookie_value = svc._beforest_ops_cookie_value()
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/admin/beforest/handover",
                "headers": [
                    (b"cookie", f"{svc.BEFOREST_OPS_COOKIE_NAME}={cookie_value}".encode()),
                    (b"x-requested-with", b"fetch"),
                ],
            }
        )
        response = await svc.beforest_admin_handover(
            request,
            contact_id="12345",
            status_value="paused",
            updated_by="founder",
            note="",
        )

    assert response.status_code == 200
    assert response.body
    assert b'"ok":true' in response.body
    assert b'"handover_status":"paused"' in response.body


def test_beforest_brand_asset_paths_exist() -> None:
    import service.service as svc

    assert svc.BEFOREST_FAVICON_ICO_PATH.exists()
    assert svc.BEFOREST_FAVICON_PNG_PATH.exists()
    assert svc.BEFOREST_OG_IMAGE_PATH.exists()


def test_beforest_operating_context_message_includes_collective_rules() -> None:
    from agents.beforest_agent import _beforest_operating_context_message

    result = _beforest_operating_context_message("How do I join a Beforest collective?")

    assert result is not None
    assert "https://beforest.co/call-mail/" in result.content
    assert "https://beforest.co/the-bhopal-collective/" in result.content
    assert "https://www.instagram.com/beforestfarming/" in result.content
    assert "Mumbai, Poomaale 2.0, Bhopal, Hammiyala" in result.content
    assert "hello@beforest.co" in result.content
    assert "show interest" in result.content
    assert "Default collective routing: send the relevant collective page first." in result.content
    assert "form.typeform.com/to/CYae8hmZ" not in result.content


def test_beforest_operating_context_message_includes_typeform_for_high_intent() -> None:
    from agents.beforest_agent import _beforest_operating_context_message

    result = _beforest_operating_context_message("I already explored Bhopal. Can you send the form?")

    assert result is not None
    assert "form.typeform.com/to/CYae8hmZ" in result.content
    assert "form.typeform.com/to/i8eBLQkz" in result.content


@pytest.mark.asyncio
@patch("service.service.httpx.AsyncClient")
async def test_save_beforest_event_omits_null_agent_reply_at(mock_async_client_class) -> None:
    import service.service as svc

    post_mock = AsyncMock()
    response_mock = MagicMock()
    response_mock.raise_for_status.return_value = None
    post_mock.return_value = response_mock
    client_mock = MagicMock()
    client_mock.post = post_mock
    async_context = AsyncMock()
    async_context.__aenter__.return_value = client_mock
    async_context.__aexit__.return_value = False
    mock_async_client_class.return_value = async_context

    fake_secret = SimpleNamespace(get_secret_value=lambda: "shared-secret")
    with (
        patch.object(svc.settings, "CONVEX_HTTP_ACTION_URL", "https://example.convex.site/instagram/store-dm-event"),
        patch.object(svc.settings, "AGENT_SHARED_SECRET", fake_secret),
    ):
        await svc._save_beforest_event_to_convex(
            user_id=None,
            thread_id="thread-1",
            subscriber_data={},
            inbound_message="",
            reply_text="",
            manychat_subscriber_id="12345",
            automation_state=svc.BeforestAutomationState(
                status="human",
                updated_at=123.0,
                updated_by="ops",
                note="",
            ),
            agent_replied=False,
        )

    payload = post_mock.await_args.kwargs["json"]
    assert "agentReplyAt" not in payload


def test_next_beforest_session_state_reopens_solved_topic_with_new_session_id() -> None:
    from service.service import BeforestSessionState, _next_beforest_session_state

    previous_session = BeforestSessionState(
        session_id="sess-1",
        status="solved",
        session_type="partnership",
        summary="Partnership details were shared.",
        last_user_goal="partnership details",
        last_activity_at=datetime.now().timestamp(),
        resolved_at=datetime.now().timestamp(),
        closed_reason="user_confirmed",
    )

    result = _next_beforest_session_state(
        previous_session=previous_session,
        message="We want to discuss a partnership for our brand.",
        reply_text="Please share your brand, city, and timeline.",
        now_ts=datetime.now().timestamp(),
    )

    assert result.session_id != "sess-1"
    assert result.session_type == "partnership"
    assert result.status == "open"


@pytest.mark.asyncio
@patch("service.service._save_beforest_event_to_convex", new_callable=AsyncMock)
async def test_beforest_handover_saves_human_ownership(mock_save_beforest_event) -> None:
    from service.service import BeforestHandoverRequest, beforest_handover

    response = await beforest_handover(
        BeforestHandoverRequest(
            status="human",
            contact_id="12345",
            updated_by="harsha",
            note="Human agent picked up the partnership thread",
        )
    )

    assert response.ok is True
    assert response.contact_id == "12345"
    assert response.handover_status == "human"
    assert mock_save_beforest_event.await_args.kwargs["agent_replied"] is False
    assert mock_save_beforest_event.await_args.kwargs["automation_state"].status == "human"


@pytest.mark.asyncio
@patch("service.service._load_beforest_events_from_convex", new_callable=AsyncMock)
async def test_beforest_handover_status_defaults_to_bot(mock_load_beforest_events) -> None:
    from service.service import beforest_handover_status

    mock_load_beforest_events.return_value = []
    response = await beforest_handover_status("12345")

    assert response.ok is True
    assert response.contact_id == "12345"
    assert response.handover_status == "bot"


@pytest.mark.asyncio
@patch("service.service._load_beforest_events_from_convex", new_callable=AsyncMock)
async def test_beforest_handover_status_returns_latest_saved_state(mock_load_beforest_events) -> None:
    from service.service import beforest_handover_status

    mock_load_beforest_events.return_value = [
        {
            "rawPayload": {
                "automation": {
                    "status": "paused",
                    "updated_at": 20,
                    "updated_by": "founder",
                    "note": "reviewing this lead personally",
                }
            }
        }
    ]
    response = await beforest_handover_status("12345")

    assert response.ok is True
    assert response.handover_status == "paused"
    assert response.updated_by == "founder"
    assert "reviewing" in response.note


@pytest.mark.asyncio
@patch("service.service._save_beforest_event_to_convex", new_callable=AsyncMock)
@patch("service.service._load_beforest_events_from_convex", new_callable=AsyncMock)
@patch("service.service._deliver_beforest_reply_background", new_callable=AsyncMock)
async def test_beforest_reply_suppresses_when_handover_is_human(
    mock_deliver_background,
    mock_load_beforest_events,
    mock_save_beforest_event,
) -> None:
    from service.service import BeforestReplyRequest, beforest_reply

    mock_load_beforest_events.return_value = [
        {
            "rawPayload": {
                "automation": {
                    "status": "human",
                    "updated_at": datetime.now().timestamp(),
                    "updated_by": "ops",
                    "note": "Human is handling this lead",
                }
            }
        }
    ]

    response = await beforest_reply(
        BeforestReplyRequest(
            message="Any update on the collab?",
            manychat_subscriber_id="12345",
            push_to_manychat=True,
        ),
        BackgroundTasks(),
    )

    assert response.ok is True
    assert response.suppressed is True
    assert response.queued is False
    assert response.reply == ""
    assert response.handover_status == "human"
    assert mock_deliver_background.await_count == 0
    assert mock_save_beforest_event.await_args.kwargs["agent_replied"] is False
    assert mock_save_beforest_event.await_args.kwargs["automation_state"].status == "human"


@patch("service.service._load_beforest_events_from_convex", new_callable=AsyncMock)
@patch("service.service._save_beforest_event_to_convex", new_callable=AsyncMock)
def test_beforest_reply_clamps_long_reply(
    mock_save_beforest_event,
    mock_load_beforest_events,
    test_client,
):
    long_reply = " ".join(["Beforest builds regenerative communities."] * 30)
    mock_load_beforest_events.return_value = []

    beforest_agent = AsyncMock()
    beforest_agent.ainvoke.return_value = [
        ("values", {"messages": [AIMessage(content=long_reply)]})
    ]

    with patch("service.service.get_agent", return_value=beforest_agent):
        response = test_client.post(
            "/beforest/reply",
            json={"message": "Tell me about Beforest", "push_to_manychat": False},
        )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["reply"]) <= 220
    assert mock_save_beforest_event.await_args.kwargs["reply_text"] == payload["reply"]
    assert mock_save_beforest_event.await_args.kwargs["session_state"].status == "solved"
    assert payload["queued"] is False


@patch("service.service._load_beforest_events_from_convex", new_callable=AsyncMock)
@patch("service.service._save_beforest_event_to_convex", new_callable=AsyncMock)
def test_beforest_reply_rewrites_stale_current_experiences_reply(
    mock_save_beforest_event,
    mock_load_beforest_events,
    test_client,
):
    stale_reply = (
        "Several immersive experiences are currently live for booking: "
        "Starry Nights in Hyderabad (Dec 14, 2025), "
        "Family Roots in Coorg (Jan 4, 2026), and "
        "Coffee Safari in Coorg (Jan 26, 2026)."
    )
    mock_load_beforest_events.return_value = []

    beforest_agent = AsyncMock()
    beforest_agent.ainvoke.return_value = [
        ("values", {"messages": [AIMessage(content=stale_reply)]})
    ]

    with patch("service.service.get_agent", return_value=beforest_agent):
        response = test_client.post(
            "/beforest/reply",
            json={"message": "What experiences are currently live?", "push_to_manychat": False},
        )

    assert response.status_code == 200
    payload = response.json()
    assert "https://experiences.beforest.co" in payload["reply"]
    assert "Dec 14, 2025" not in payload["reply"]
    assert len(payload["reply"]) <= 220
    assert mock_save_beforest_event.await_args.kwargs["reply_text"] == payload["reply"]
    assert payload["queued"] is False


@patch("service.service._load_beforest_events_from_convex", new_callable=AsyncMock)
@patch("service.service._save_beforest_event_to_convex", new_callable=AsyncMock)
def test_beforest_reply_continues_matching_prior_session_and_saves_context(
    mock_save_beforest_event,
    mock_load_beforest_events,
    test_client,
):
    prior_events = [
        {
            "message": "I want to collaborate with Beforest.",
            "agentReplyText": "Please share your details and goals.",
            "rawPayload": {
                "session": {
                    "session_id": "sess-creator-1",
                    "status": "open",
                    "session_type": "partnership",
                    "summary": "Partnership conversation started.",
                    "last_user_goal": "collaborate with Beforest",
                    "last_activity_at": datetime.now().timestamp(),
                    "resolved_at": None,
                    "closed_reason": "",
                }
            },
        }
    ]
    mock_load_beforest_events.return_value = prior_events

    beforest_agent = AsyncMock()
    beforest_agent.ainvoke.return_value = [
        ("values", {"messages": [AIMessage(content="Share your brand, scope, and timeline.")]}),
    ]

    with patch("service.service.get_agent", return_value=beforest_agent):
        response = test_client.post(
            "/beforest/reply",
            json={
                "message": "We want to discuss a partnership.",
                "user_id": "test-contact-1",
                "push_to_manychat": False,
            },
        )

    assert response.status_code == 200
    input_messages = beforest_agent.ainvoke.await_args.kwargs["input"]["messages"]
    assert input_messages[0].content.startswith("Beforest DM session context.")
    assert "Partnership conversation started." in input_messages[0].content
    assert len(input_messages) == 4
    assert mock_save_beforest_event.await_args.kwargs["session_state"].session_id == "sess-creator-1"
    assert mock_save_beforest_event.await_args.kwargs["session_state"].session_type == "partnership"


@patch("service.service._load_beforest_events_from_convex", new_callable=AsyncMock)
@patch("service.service._save_beforest_event_to_convex", new_callable=AsyncMock)
def test_beforest_reply_starts_fresh_for_unrelated_topic(
    mock_save_beforest_event,
    mock_load_beforest_events,
    test_client,
):
    prior_events = [
        {
            "message": "We want to explore a partnership.",
            "agentReplyText": "Please share your brand brief.",
            "rawPayload": {
                "session": {
                    "session_id": "sess-partner-1",
                    "status": "open",
                    "session_type": "partnership",
                    "summary": "Partnership conversation started.",
                    "last_user_goal": "partnership ask",
                    "last_activity_at": datetime.now().timestamp(),
                    "resolved_at": None,
                    "closed_reason": "",
                }
            },
        }
    ]
    mock_load_beforest_events.return_value = prior_events

    beforest_agent = AsyncMock()
    beforest_agent.ainvoke.return_value = [
        ("values", {"messages": [AIMessage(content="You can browse Bewild products at https://bewild.life.")]}),
    ]

    with patch("service.service.get_agent", return_value=beforest_agent):
        response = test_client.post(
            "/beforest/reply",
            json={
                "message": "Do you sell coffee or spices?",
                "user_id": "test-contact-2",
                "push_to_manychat": False,
            },
        )

    assert response.status_code == 200
    input_messages = beforest_agent.ainvoke.await_args.kwargs["input"]["messages"]
    assert len(input_messages) == 1
    assert input_messages[0].content == "Do you sell coffee or spices?"
    assert mock_save_beforest_event.await_args.kwargs["session_state"].session_id != "sess-partner-1"
    assert mock_save_beforest_event.await_args.kwargs["session_state"].session_type == "product"


@patch("service.service._load_beforest_events_from_convex", new_callable=AsyncMock)
@patch("service.service._save_beforest_event_to_convex", new_callable=AsyncMock)
def test_beforest_reply_reopens_solved_session_with_context_but_without_old_history(
    mock_save_beforest_event,
    mock_load_beforest_events,
    test_client,
):
    prior_events = [
        {
            "message": "We want to discuss a partnership.",
            "agentReplyText": "Please share your brand, scope, and timeline.",
            "rawPayload": {
                "session": {
                    "session_id": "sess-partner-closed",
                    "status": "solved",
                    "session_type": "partnership",
                    "summary": "Partnership details were shared.",
                    "last_user_goal": "partnership details",
                    "last_activity_at": datetime.now().timestamp(),
                    "resolved_at": datetime.now().timestamp(),
                    "closed_reason": "user_confirmed",
                }
            },
        }
    ]
    mock_load_beforest_events.return_value = prior_events

    beforest_agent = AsyncMock()
    beforest_agent.ainvoke.return_value = [
        ("values", {"messages": [AIMessage(content="Please share your brand, city, and dates.")]}),
    ]

    with patch("service.service.get_agent", return_value=beforest_agent):
        response = test_client.post(
            "/beforest/reply",
            json={
                "message": "We want to revisit the partnership idea.",
                "user_id": "test-contact-3",
                "push_to_manychat": False,
            },
        )

    assert response.status_code == 200
    input_messages = beforest_agent.ainvoke.await_args.kwargs["input"]["messages"]
    assert len(input_messages) == 2
    assert input_messages[0].content.startswith("Beforest DM session context.")
    assert input_messages[1].content == "We want to revisit the partnership idea."
    assert mock_save_beforest_event.await_args.kwargs["session_state"].session_id != "sess-partner-closed"


@pytest.mark.asyncio
@patch("service.service._load_beforest_events_from_convex", new_callable=AsyncMock)
@patch("service.service._deliver_beforest_reply_background", new_callable=AsyncMock)
async def test_beforest_reply_queues_background_delivery_for_manychat_push(
    mock_deliver_background,
    mock_load_beforest_events,
):
    from service.service import BeforestReplyRequest, beforest_reply

    mock_load_beforest_events.return_value = []
    background_tasks = BackgroundTasks()
    response = await beforest_reply(
        BeforestReplyRequest(
            message="Tell me about stays in Coorg",
            manychat_subscriber_id="12345",
            push_to_manychat=True,
        ),
        background_tasks,
    )

    assert response.ok is True
    assert response.queued is True
    assert response.reply == ""
    assert response.handover_status == "bot"
    assert response.thread_id
    assert len(background_tasks.tasks) == 1
    task = background_tasks.tasks[0]
    assert task.func
    assert task.kwargs["manychat_subscriber_id"] == "12345"
    assert task.kwargs["resolved_thread_id"] == response.thread_id
    assert task.kwargs["automation_state"] is None


@pytest.mark.asyncio
@patch("service.service._load_beforest_events_from_convex", new_callable=AsyncMock)
async def test_beforest_reply_rejects_background_push_without_manychat_subscriber(
    mock_load_beforest_events,
):
    from service.service import BeforestReplyRequest, beforest_reply

    mock_load_beforest_events.return_value = []
    with pytest.raises(Exception) as exc_info:
        await beforest_reply(
            BeforestReplyRequest(
                message="Tell me about stays in Coorg",
                user_id="ig-user-1",
                push_to_manychat=True,
            ),
            BackgroundTasks(),
        )

    assert getattr(exc_info.value, "status_code", None) == 422
    assert "manychat_subscriber_id" in str(getattr(exc_info.value, "detail", ""))
