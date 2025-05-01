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
import logging

logger = logging.getLogger(__name__)

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

    # Map the agent name to the correct node in the graph
    if decision["agent"] == "CONVERSATION_AGENT":
        state["next_agent"] = AgentName.CONVERSATION.value
    elif decision["agent"] == "RAG_AGENT":
        state["next_agent"] = AgentName.RAG.value
    elif decision["agent"] == "WEB_SEARCH_PROCESSOR_AGENT":
        state["next_agent"] = AgentName.WEB_SEARCH.value
    else:
        # Default to conversation if we don't recognize the agent
        logger.warning(f"Unrecognized agent: {decision['agent']}, defaulting to conversation")
        state["next_agent"] = AgentName.CONVERSATION.value

    logger.info(f"Routing to agent: {state['next_agent']}")

    # Preserve patient_response_text in the final_output if it exists
    # This ensures that patient responses from analyze_patient_query make it to the final state
    if state.get("patient_response_text") and not state.get("final_output"):
        state["final_output"] = state["patient_response_text"]

    return state

def confidence_based_routing(state: BaseAgentState) -> str:
    """If RAG was low-confidence or insufficient info, reroute to web search."""
    low_conf = state.get("retrieval_confidence", 0.0) < settings.rag.min_retrieval_confidence
    if low_conf or state.get("insufficient_info", False):
        return AgentName.WEB_SEARCH.value
    return "perform_human_validation" # Changed from "check_validation" to match node name
