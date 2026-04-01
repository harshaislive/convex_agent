from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.beforest_tools import (
    browse_beforest_page,
    fetch_beforest_markdown,
    search_beforest_experiences,
    search_beforest_knowledge,
    search_beforest_live,
)
from core import get_model, settings

PROMPT_PATH = Path(__file__).parent / "beforest" / "AGENTS.md"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()
tools = [
    search_beforest_knowledge,
    search_beforest_live,
    search_beforest_experiences,
    fetch_beforest_markdown,
    browse_beforest_page,
]
_COLLECTIVE_QUERY_TERMS = (
    "collective",
    "collectives",
    "hammiyala",
    "bhopal",
    "poomaale",
    "mumbai",
    "join",
    "visit",
    "resident",
    "stay",
    "show interest",
    "invite",
)
_CONTACT_QUERY_TERMS = (
    "contact",
    "email",
    "reach out",
    "write to",
    "partnership",
    "collab",
    "collaborate",
    "creator",
    "influencer",
)
_ROUTING_QUERY_TERMS = (
    "link",
    "website",
    "page",
    "homepage",
    "instagram",
    "facebook",
    "social",
)
_COLLECTIVE_INTEREST_LINKS = {
    "bhopal": "https://form.typeform.com/to/CYae8hmZ?utm_source=xxxxx&utm_medium=xxxxx&utm_content=xxxxx&utm_time_spent=xxxxx#device=xxxxx&intent_copy=xxxxx&distraction_score=xxxxx&first_visit=xxxxx&total_visits=xxxxx&behavioral_journey=xxxxx&current_page=xxxxx&rage_clicks=xxxxx&confusion_score=xxxxx",
    "poomaale 2.0": "https://form.typeform.com/to/i8eBLQkz?utm_source=xxxxx&utm_medium=xxxxx&utm_content=xxxxx&utm_time_spent=xxxxx#current_page=xxxxx&behavioral_journey=xxxxx&total_visits=xxxxx&first_visit=xxxxx&distraction_score=xxxxx&intent_copy=xxxxx&device=xxxxx&rage_clicks=xxxxx&confusion_score=xxxxx",
    "hammiyala": "https://form.typeform.com/to/hbDB2ybS?utm_source=xxxxx&utm_medium=xxxxx&utm_content=xxxxx&utm_time_spent=xxxxx#current_page=xxxxx&behavioral_journey=xxxxx&total_visits=xxxxx&first_visit=xxxxx&distraction_score=xxxxx&intent_copy=xxxxx&device=xxxxx&rage_clicks=xxxxx&confusion_score=xxxxx",
    "mumbai": "https://form.typeform.com/to/kfcjiXxR?utm_source=xxxxx&utm_medium=xxxxx&utm_content=xxxxx&utm_campaign=xxxxx&utm_term=xxxxx&utm_time_spent=xxxxx#current_page=xxxxx&behavioral_journey=xxxxx&total_visits=xxxxx&first_visit=xxxxx&distraction_score=xxxxx&intent_copy=xxxxx&device=xxxxx&rage_clicks=xxxxx&confusion_score=xxxxx",
}
_FAST_LINKS = {
    "beforest_home": "https://beforest.co/",
    "bewild_home": "https://bewild.life/",
    "experiences": "https://experiences.beforest.co/",
    "hospitality": "https://hospitality.beforest.co/",
    "ten_percent": "https://10percent.beforest.co/",
    "contact": "https://beforest.co/call-mail/",
    "poomaale_1": "https://beforest.co/the-poomaale-estate/",
    "poomaale_2": "https://beforest.co/poomaale-2-0-collective/",
    "hyderabad_collective": "https://beforest.co/hyderabad-collective/",
    "mumbai_collective": "https://beforest.co/the-mumbai-collective/",
    "hammiyala_collective": "https://beforest.co/co-forest/",
    "bhopal_collective": "https://beforest.co/the-bhopal-collective/",
    "beforest_instagram": "https://www.instagram.com/beforestfarming/",
    "beforest_facebook": "https://www.facebook.com/beforestfarming/",
    "bewild_instagram": "https://www.instagram.com/bewild.life/",
}


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


def _beforest_operating_context_message(question: str) -> SystemMessage | None:
    lowered = question.lower()
    if not any(
        term in lowered
        for term in (*_COLLECTIVE_QUERY_TERMS, *_CONTACT_QUERY_TERMS, *_ROUTING_QUERY_TERMS)
    ):
        return None

    lines = [
        "Beforest operating guidance.",
        "Use these canonical fast links when routing users:",
        f"- Beforest home: {_FAST_LINKS['beforest_home']}",
        f"- Bewild: {_FAST_LINKS['bewild_home']}",
        f"- Experiences: {_FAST_LINKS['experiences']}",
        f"- Hospitality: {_FAST_LINKS['hospitality']}",
        f"- 10Percent: {_FAST_LINKS['ten_percent']}",
        f"- Contact: {_FAST_LINKS['contact']}",
        "Operating collectives to mention: Mumbai, Poomaale 2.0, Bhopal, Hammiyala.",
        "Do not imply that other collectives are currently open; say they are full if needed.",
        "Collective page links:",
        f"- Poomaale 1.0: {_FAST_LINKS['poomaale_1']}",
        f"- Poomaale 2.0: {_FAST_LINKS['poomaale_2']}",
        f"- Hyderabad Collective: {_FAST_LINKS['hyderabad_collective']}",
        f"- Mumbai Collective: {_FAST_LINKS['mumbai_collective']}",
        f"- Hammiyala Collective: {_FAST_LINKS['hammiyala_collective']}",
        f"- Bhopal Collective: {_FAST_LINKS['bhopal_collective']}",
        "For collective interest, use 'show interest' language. Do not assume pages say 'get invite'.",
        "For specific collective sign-up interest, use these exact links when relevant:",
        f"- Bhopal: {_COLLECTIVE_INTEREST_LINKS['bhopal']}",
        f"- Poomaale 2.0: {_COLLECTIVE_INTEREST_LINKS['poomaale 2.0']}",
        f"- Hammiyala: {_COLLECTIVE_INTEREST_LINKS['hammiyala']}",
        f"- Mumbai: {_COLLECTIVE_INTEREST_LINKS['mumbai']}",
        "Social links:",
        f"- Beforest Instagram: {_FAST_LINKS['beforest_instagram']}",
        f"- Beforest Facebook: {_FAST_LINKS['beforest_facebook']}",
        f"- Bewild Instagram: {_FAST_LINKS['bewild_instagram']}",
        "Default non-signup contact route is hello@beforest.co or https://beforest.co/call-mail/.",
    ]
    return SystemMessage(content="\n".join(lines))


def wrap_model(model) -> RunnableSerializable[AgentState, AIMessage]:
    bound_model = model.bind_tools(tools)

    def _state_modifier(state: AgentState):
        messages = list(state["messages"])
        question = _latest_human_message(messages)
        prompt_messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        operating_message = _beforest_operating_context_message(question)
        if operating_message is not None:
            prompt_messages.append(operating_message)
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
