from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.beforest_tools import (
    browse_beforest_page,
    search_beforest_experiences,
    search_beforest_knowledge,
)
from core import get_model, settings

PROMPT_PATH = Path(__file__).parent / "beforest" / "AGENTS.md"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()
tools = [
    search_beforest_knowledge,
    search_beforest_experiences,
    browse_beforest_page,
]


class AgentState(MessagesState, total=False):
    remaining_steps: RemainingSteps


def _sanitize_message_content(message: BaseMessage) -> BaseMessage:
    """Azure-compatible normalization for empty message content."""
    content = message.content
    if isinstance(content, str):
        if content != "":
            return message
        return message.model_copy(update={"content": " "})

    if len(content) > 0:
        return message

    return message.model_copy(update={"content": [{"type": "text", "text": " "}]})


def _latest_human_message(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _knowledge_context_message(question: str) -> SystemMessage | None:
    if not question.strip():
        return None
    results = search_beforest_knowledge.invoke({"query": question, "max_results": 3})
    if not isinstance(results, list) or not results:
        return None

    lines = [
        "Beforest approved knowledge context from Outline.",
        "Use this for factual answers. If it does not cover the answer, say that plainly.",
    ]
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "Unknown source"))
        snippet = str(item.get("snippet", "")).strip()
        if snippet:
            lines.append(f"{index}. {source}: {snippet}")
    if len(lines) <= 2:
        return None
    return SystemMessage(content="\n".join(lines))


def wrap_model(model) -> RunnableSerializable[AgentState, AIMessage]:
    bound_model = model.bind_tools(tools)

    def _state_modifier(state: AgentState):
        messages = list(state["messages"])
        question = _latest_human_message(messages)
        prompt_messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        knowledge_message = _knowledge_context_message(question)
        if knowledge_message is not None:
            prompt_messages.append(knowledge_message)
        prompt_messages.extend(_sanitize_message_content(message) for message in messages)
        return prompt_messages

    return RunnableLambda(_state_modifier, name="BeforestStateModifier") | bound_model  # type: ignore[return-value]


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(model)
    response = await model_runnable.ainvoke(state, config)
    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I need more steps to process this request.",
                )
            ]
        }
    return {"messages": [response]}


agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.set_entry_point("model")


def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_edge("tools", "model")
agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

beforest_agent = agent.compile()
