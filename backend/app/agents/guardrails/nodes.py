from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
import logging

from app.agents.states import BaseAgentState
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.settings import settings
from .core import Guardrails

logger = logging.getLogger(__name__)

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
    # Ensure current_input is a string and handle None
    current_input_str = state.get("current_input", "")
    if not isinstance(current_input_str, str):
        current_input_str = str(current_input_str)

    allowed, msg = guard.check_input(current_input_str)
    if not allowed:
        # Ensure msg is a BaseMessage before adding to list
        if isinstance(msg, BaseMessage):
             state["messages"] = [msg]
        else:
             # If check_input returns a string, wrap it
             state["messages"] = [AIMessage(content=str(msg))]
        state["agent_name"] = "INPUT_GUARDRAILS"
        state["bypass_routing"] = True # Assuming this key is used elsewhere to stop graph execution
    else:
        # Ensure state doesn't accidentally get bypassed
        state["bypass_routing"] = False
    return state


def apply_output_guardrails(state: BaseAgentState) -> BaseAgentState:
    guard = _get_guard()
    txt_to_check = "" # Default to empty

    # --- Refined Logic to Find Agent's Final Output ---
    messages = state.get("messages", [])
    if messages:
        # Iterate backwards to find the latest AIMessage
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                txt_to_check = msg.content
                logger.debug(f"Output Guardrails: Found last AIMessage content: '{txt_to_check[:100]}...'")
                break
        # If no AIMessage found, maybe log or handle differently?
        # For now, txt_to_check might remain "" if no AIMessage is last
        if not txt_to_check:
             logger.warning("Output Guardrails: Could not find AIMessage in the end of messages list.")
             # Optional: Fallback to other state keys if necessary, but prioritize last AIMessage
             output = state.get("output")
             if output is not None:
                 txt_to_check = output.content if isinstance(output, AIMessage) else str(output)
             elif state.get("final_output") is not None:
                 txt_to_check = str(state.get("final_output"))

    else:
        logger.warning("Output Guardrails: Messages list is empty.")
        # Fallback if messages list is empty
        output = state.get("output")
        if output is not None:
            txt_to_check = output.content if isinstance(output, AIMessage) else str(output)
        elif state.get("final_output") is not None:
            txt_to_check = str(state.get("final_output"))


    # Ensure user_input for context is a string
    current_input_str = state.get("current_input", "")
    if not isinstance(current_input_str, str):
        current_input_str = str(current_input_str)

    # Apply guardrails to the determined text
    clean_output = guard.check_output(txt_to_check, current_input_str)
    logger.debug(f"Output Guardrails: Cleaned output: '{clean_output[:100]}...'")


    # --- Update State Consistently ---
    # Set final_output (primary field for extraction)
    state["final_output"] = clean_output

    # Set output for potential internal use (less critical for final extraction)
    state["output"] = AIMessage(content=clean_output)

    # Ensure the messages list ends with the *cleaned* AIMessage
    if messages:
        last_message = messages[-1]
        # If the last message was the AI message we checked and its content differs, update it
        if isinstance(last_message, AIMessage) and last_message.content != clean_output:
            logger.debug("Output Guardrails: Updating last message content.")
            state["messages"][-1] = AIMessage(content=clean_output)
        # If the last message wasn't an AIMessage (e.g., ToolMessage), append the clean AI message
        elif not isinstance(last_message, AIMessage):
            logger.debug("Output Guardrails: Appending clean message as last was not AI.")
            state["messages"].append(AIMessage(content=clean_output))
        # Else: Last message is AI and already clean, do nothing.
    else:
        # If messages was empty, add the clean message
        logger.debug("Output Guardrails: Appending clean message to empty list.")
        state["messages"] = [AIMessage(content=clean_output)]


    # If patient_response_text exists, update it too (optional, depends if used elsewhere)
    if "patient_response_text" in state:
        state["patient_response_text"] = clean_output

    return state

def perform_human_validation(state: BaseAgentState) -> BaseAgentState:
    # build and append a "please click yes/no" prompt
    output = state.get("output") # Read the potentially guardrail-modified output
    txt = output.content if isinstance(output, AIMessage) else str(output)
    validation_message = f"{txt}\n\n**Human Validation Required:**\n- If you're a healthcare professional: Please validate the output. Select **Yes** or **No**. If No, provide comments.\n- If you're a patient: Simply click Yes to confirm."

    # Update state consistently
    state["output"] = AIMessage(content=validation_message)
    state["final_output"] = validation_message # Set final_output too!
    state["agent_name"] = state.get("agent_name", "") + ",HUMAN_VALIDATION"

    # Also update the last message in the list
    if state.get("messages"):
        state["messages"][-1] = AIMessage(content=validation_message)
    else:
        state["messages"] = [AIMessage(content=validation_message)]

    return state
