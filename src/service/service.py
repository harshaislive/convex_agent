import inspect
import json
import logging
import re
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse  # type: ignore[import-untyped]
from langfuse.langchain import (
    CallbackHandler,  # type: ignore[import-untyped]
)
from langgraph.types import Command, Interrupt
from langsmith import Client as LangsmithClient
from langsmith import uuid7

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


async def _load_beforest_history_from_convex(contact_id: str) -> list[HumanMessage | AIMessage]:
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

    messages: list[HumanMessage | AIMessage] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        human_text = str(item.get("message", "") or "").strip()
        ai_text = str(item.get("agentReplyText", "") or "").strip()
        if human_text:
            messages.append(HumanMessage(content=human_text))
        if ai_text:
            messages.append(AIMessage(content=ai_text))
    return messages


async def _save_beforest_event_to_convex(
    *,
    user_id: str | None,
    thread_id: str,
    subscriber_data: dict[str, Any],
    inbound_message: str,
    reply_text: str,
    manychat_subscriber_id: str | None,
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


async def _push_manychat_reply(subscriber_id: str, reply_text: str) -> None:
    if not settings.MANYCHAT_API_TOKEN:
        return

    buttons = [
        {"type": "url", "caption": _button_caption_for_url(url), "url": url}
        for url in _extract_urls(reply_text)[:3]
    ]
    message: dict[str, Any] = {"type": "text", "text": reply_text}
    if buttons:
        message["buttons"] = buttons

    payload = {
        "subscriber_id": int(subscriber_id),
        "data": {
            "version": "v2",
            "content": {
                "type": settings.MANYCHAT_CHANNEL,
                "messages": [message],
                "actions": [],
                "quick_replies": [],
            },
        },
    }

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            f"{settings.MANYCHAT_API_BASE_URL.rstrip('/')}/fb/sending/sendContent",
            json=payload,
            headers={
                "Authorization": f"Bearer {settings.MANYCHAT_API_TOKEN.get_secret_value()}",
                "Content-Type": "application/json",
            },
        )
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
    history_messages = (
        await _load_beforest_history_from_convex(str(contact_id)) if contact_id else []
    )
    kwargs["input"] = {"messages": [*history_messages, HumanMessage(content=request.message)]}
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

    thread_id = request.thread_id or kwargs["config"].configurable["thread_id"]
    try:
        await _save_beforest_event_to_convex(
            user_id=request.user_id,
            thread_id=str(thread_id),
            subscriber_data=request.subscriber_data,
            inbound_message=request.message,
            reply_text=output.content,
            manychat_subscriber_id=request.manychat_subscriber_id,
        )
    except Exception as exc:
        logger.error(f"Failed to write Beforest event to Convex: {exc}")

    if request.push_to_manychat and contact_id:
        try:
            await _push_manychat_reply(str(contact_id), output.content)
        except Exception as exc:
            logger.error(f"Failed to push Beforest reply to ManyChat: {exc}")

    return BeforestReplyResponse(ok=True, reply=output.content, thread_id=str(thread_id))


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
