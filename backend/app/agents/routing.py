from typing import Dict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.agents.states import BaseAgentState
from app.config.prompts import DECISION_SYSTEM_PROMPT
from app.schemas.llm import AgentDecision
from app.config.settings import settings
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.constants import AgentName
from app.config.agent import settings as agentSettings


def route_to_agent(state: BaseAgentState) -> BaseAgentState:
    # 1) assemble user+last 6 turns
    inp = state.get("current_input", "") or ""
    session = ""
    for m in state.get("messages", [])[-6:]:
        prefix = "User: " if isinstance(m, HumanMessage) else "Assistant: "
        session += prefix + m.content + "\n"

    payload = f"User query: {inp}\n\nRecent conversation:\n{session}"
    prompt = ChatPromptTemplate.from_messages([
      ("system", DECISION_SYSTEM_PROMPT),
      ("human",  "{input}")
    ])
    parser = JsonOutputParser(pydantic_object=AgentDecision)
    llm = ChatGoogleGenerativeAI(
      model="gemini-2.5-flash-preview-04-17",
      api_key=settings.google_api_key
    )
    chain = prompt | llm | parser
    decision: AgentDecision = chain.invoke({"input": payload})

    state["agent_name"] = decision["agent"]
    # pick next based on confidence
    if (decision.get("confidence") or 0.0) < agentSettings.CONFIDENCE_THRESHOLD:
        state["next_agent"] = "needs_validation"
    else:
        state["next_agent"] = decision["agent"]
    return state

def confidence_based_routing(state: BaseAgentState) -> str:
    """If RAG was low-confidence or insufficient info, reroute to web search."""
    low_conf = state.get("retrieval_confidence", 0.0) < settings.rag.min_retrieval_confidence
    if low_conf or state.get("insufficient_info", False):
        return AgentName.WEB_SEARCH.value
    return "check_validation"
