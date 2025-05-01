import logging
from app.agents.states import PatientState
from app.config.settings import settings
from app.config.constants import AgentName, Task
from app.config.prompts import PATIENT_ANALYSIS_SYSTEM_PROMPT
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.schemas.llm import PatientAnalysisOutput
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

def patient_supervisor(state: PatientState) -> PatientState:
    """
    Unified function to analyze patient queries, generate responses, determine intent,
    and decide if scheduling is needed. This replaces both the keyword classifier and
    the analyze_patient_query function.

    Args:
        state: The current PatientState

    Returns:
        Updated PatientState with patient_response_text, final_output, intent, and request_scheduling
    """
    logger.info("Node: patient_supervisor")

    # 1) Extract the raw user query
    messages = state.get("messages", [])
    history = messages[:-1]
    current_query = ""
    if messages and hasattr(messages[-1], "content"):
        current_query = messages[-1].content

    # 2) Handle empty query
    if not current_query.strip():
        logger.warning("No user query found; asking for clarification")
        state["patient_response_text"] = (
            "I'm sorry, I didn't quite catch that. "
            "Could you please repeat your question?"
        )
        state["final_output"] = state["patient_response_text"]
        state["request_scheduling"] = False
        state["intent"] = "conversation"
        return state

    # 3) Build LLM prompt
    history_str = "\n".join(
        f"{m.type.upper()}: {m.content}" for m in history
        if hasattr(m, "type") and hasattr(m, "content")
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            api_key=settings.google_api_key
        )
        prompt = ChatPromptTemplate.from_template(
            PATIENT_ANALYSIS_SYSTEM_PROMPT
        )
        prompt.input_variables = ["history", "query"]
        parser = JsonOutputParser(pydantic_object=PatientAnalysisOutput)
        chain = prompt | llm | parser

    except Exception as e:
        logger.error("Failed to initialize LLM or chain", exc_info=True)
        state["patient_response_text"] = (
            "Sorry, I'm having trouble understanding right now."
        )
        state["final_output"] = state["patient_response_text"]
        state["request_scheduling"] = False
        state["intent"] = "conversation"
        return state

    # 4) Invoke LLM and parse
    try:
        llm_response: PatientAnalysisOutput = chain.invoke({
            "history": history_str,
            "query": current_query
        })
        # Extract fields with safe defaults
        response_text = llm_response.get("response_text", "")
        request_sched_flag = llm_response.get("request_scheduling", False)
        intent = llm_response.get("intent", "conversation")

    except Exception as e:
        logger.error("Error during LLM call", exc_info=True)
        state["patient_response_text"] = (
            "Sorry, I encountered an error while processing your request."
        )
        state["final_output"] = state["patient_response_text"]
        state["request_scheduling"] = False
        state["intent"] = "conversation"
        return state

    # 5) Write back into state
    state["patient_response_text"] = response_text or "I'm not sure how to respond to that."
    state["final_output"] = state["patient_response_text"]
    state["request_scheduling"] = bool(request_sched_flag)
    state["intent"] = intent

    # Override intent if we need scheduling
    if state["request_scheduling"]:
        state["intent"] = "scheduling"

    logger.info(
        f"Patient supervisor: intent={state['intent']}, "
        f"scheduling={state['request_scheduling']}, "
        f"response={state['patient_response_text']!r}"
    )
    return state

def build_patient_supervisor_graph() -> StateGraph:
    """
    Build a simple graph that just uses the patient supervisor.

    Returns:
        A StateGraph with a single patient supervisor node
    """
    g = StateGraph(PatientState)
    g.add_node("supervisor", patient_supervisor)
    g.set_entry_point("supervisor")
    g.add_edge("supervisor", END)
    return g
