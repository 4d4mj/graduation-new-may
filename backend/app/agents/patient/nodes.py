import logging

from app.agents.states import PatientState
from app.config.settings import settings
from app.config.prompts import PATIENT_ANALYSIS_SYSTEM_PROMPT
from app.config.constants import AgentName  # add constant import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.schemas.llm import PatientAnalysisOutput

logger = logging.getLogger(__name__)


def analyze_patient_query(state: PatientState) -> PatientState:
    """
    Uses an LLM to analyze the patient query, generate a response,
    and decide if scheduling is needed.  Mutates and returns the same state.
    """
    logger.info("Node: analyze_patient_query")

    # 1) extract the raw user query
    messages = state.get("messages", [])
    history = messages[:-1]
    current_query = ""
    if messages and hasattr(messages[-1], "content"):
        current_query = messages[-1].content

    # 2) handle empty query
    if not current_query.strip():
        logger.warning("No user query found; asking for clarification")
        state["patient_response_text"] = (
            "I'm sorry, I didn't quite catch that. "
            "Could you please repeat your question?"
        )
        state["request_scheduling"] = False
        return state

    # 3) build LLM prompt
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
        state["request_scheduling"] = False
        return state

    # 4) invoke LLM and parse
    try:
        llm_response: PatientAnalysisOutput = chain.invoke({
            "history": history_str,
            "query": current_query
        })
        # extract fields with safe defaults
        response_text      = llm_response.get("response_text", "")
        request_sched_flag = llm_response.get("request_scheduling", False)

    except Exception as e:
        logger.error("Error during LLM call", exc_info=True)
        state["patient_response_text"] = (
            "Sorry, I encountered an error while processing your request."
        )
        state["request_scheduling"] = False
        return state

    # 5) write back into state
    state["patient_response_text"] = (
        response_text or "I'm not sure how to respond to that."
    )
    state["request_scheduling"] = bool(request_sched_flag)
    logger.info(
        f"Patient analysis: scheduling={state['request_scheduling']}, "
        f"response={state['patient_response_text']!r}"
    )
    return state

def prepare_for_scheduling(state: PatientState) -> PatientState:
    """Sets the state to signal handoff to the scheduling supervisor"""
    logger.info("Node: prepare_for_scheduling")
    response_text = state.get("patient_response_text", "") or "Okay, let's look into scheduling."
    state["final_output"] = response_text
    state["next_agent"] = AgentName.SCHEDULER.value  # hand off to scheduler
    return state
