from langchain_core.messages import AIMessage

from app.agents.states import BaseAgentState
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.settings import settings
from .core import Guardrails

_guard = None

def _get_guard():
    global _guard
    if (_guard is None):
        llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        api_key=settings.google_api_key,
        temperature=0.7
    )
        _guard = Guardrails(llm)
    return _guard

def apply_input_guardrails(state: BaseAgentState) -> BaseAgentState:
    guard = _get_guard()
    allowed, msg = guard.check_input(state.get("current_input", "") or "")
    if not allowed:
        # stash the single AIMessage in a one-element list
        state["messages"] = [msg]
        state["agent_name"] = "INPUT_GUARDRAILS"
        state["bypass_routing"] = True
    return state


def apply_output_guardrails(state: BaseAgentState) -> BaseAgentState:
    guard = _get_guard()
    output = state.get("output")
    txt = output.content if isinstance(output, AIMessage) else str(output)
    clean = guard.check_output(txt, state.get("current_input", "") or "")
    state["output"] = AIMessage(content=clean)
    # Also set final_output for consistency across all nodes
    state["final_output"] = clean
    state["messages"] = (state.get("messages", []) or []) + [state["output"]]
    return state


def perform_human_validation(state: BaseAgentState) -> BaseAgentState:
    # build and append a "please click yes/no" prompt
    output = state.get("output")
    txt = output.content if isinstance(output, AIMessage) else str(output)
    validation_message = f"{txt}\n\nHuman Validation Required:**\n- If you're a healthcare professional: Please validate the output. Select **Yes** or **No**. If No, provide comments.\n- If you're a patient: Simply click Yes to confirm."
    state["output"] = AIMessage(content=validation_message)
    # Also set final_output for consistency across all nodes
    state["final_output"] = validation_message
    state["agent_name"] = state.get("agent_name", "") + ",HUMAN_VALIDATION"
    return state
