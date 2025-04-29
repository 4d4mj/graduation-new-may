from typing import Dict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.agents.guardrails.local_guardrails import LocalGuardrails
from .state import AgentState, AgentDecision
from app.config.agent import settings
from app.config.provider import PROVIDER  # singleton provider instance

provider = PROVIDER  # reuse singleton provider

_output_guard = None

def _get_output_guard():
    global _output_guard
    if _output_guard is None:
        _output_guard = LocalGuardrails(provider.chat())
    return _output_guard

def analyze_input(state: AgentState) -> AgentState:
    """Light pre-processing: guard-rails."""
    current_input = state.get("current_input", "")
    input_text = current_input if isinstance(current_input, str) else current_input.get("text", "")

    guard = _get_output_guard()
    allowed, message = guard.check_input(input_text)
    if not allowed:
        state["messages"] = message
        state["agent_name"] = "INPUT_GUARDRAILS"
        state["bypass_routing"] = True
    return state


def route_to_agent(state: AgentState) -> Dict:
    """Compose prompt + last N messages, call LLM, parse decision."""
    current_input = state.get("current_input", "")
    input_text = current_input if isinstance(current_input, str) else current_input.get("text", "")

    # assemble recent conversation
    recent_context = ""
    for msg in state.get("messages", [])[-6:]:
        if isinstance(msg, HumanMessage):
            recent_context += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            recent_context += f"Assistant: {msg.content}\n"

    decision_input = f"User query: {input_text}\n\nRecent conversation context:\n{recent_context}"

    # build chain
    prompt = ChatPromptTemplate.from_messages(
        [("system", settings.DECISION_SYSTEM_PROMPT),
         ("human",  "{input}")]
    )
    parser = JsonOutputParser(pydantic_object=AgentDecision)
    decision_chain = prompt | provider.chat() | parser
    decision = decision_chain.invoke({"input": decision_input})

    updated = {**state, "agent_name": decision["agent"]}
    if decision.get("confidence", 0.0) < settings.CONFIDENCE_THRESHOLD:
     return {"agent_state": updated, "next": "needs_validation"}
    return {"agent_state": updated, "next": decision["agent"]}


def confidence_based_routing(state: AgentState) -> str:
    """If RAG was low-confidence or insufficient info, reroute to web search."""
    low_conf = state.get("retrieval_confidence", 0.0) < settings.rag.min_retrieval_confidence
    if low_conf or state.get("insufficient_info", False):
        return "WEB_SEARCH_PROCESSOR_AGENT"
    return "check_validation"
