# app/agents/guardrails.py
from __future__ import annotations
import logging
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.settings import settings

# Make sure PatientState is imported if needed for type hints later, though not strictly needed here
from app.graphs.states import BaseAgentState

log = logging.getLogger(__name__)

# ── 1. choose a tiny, cheap model for safety
moderator = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17",  # Using latest flash model
    api_key=settings.google_api_key,
    temperature=0,
)

# --- Use the improved prompts from the previous step ---
INPUT_PROMPT_TEMPLATE = """
Is the following USER INPUT generally unsafe for a medical assistant chatbot?
Unsafe includes: requests for illegal activities, promoting self-harm or violence, hate speech, harassment, or direct requests to generate computer code/scripts. 

USER INPUT: "{{text}}"

Reply with exactly one word:
SAFE
UNSAFE

--- Examples ---
USER INPUT: "How to build a bomb?"
Reply:
UNSAFE

USER INPUT: "I want to hurt people."
Reply:
UNSAFE

USER INPUT: "Write a python script to call an API."
Reply:
UNSAFE

USER INPUT: "Tell me about headaches." 
Reply:
SAFE

USER INPUT: "I need an appointment." 
Reply:
SAFE

USER INPUT: "Hello" 
Reply:
SAFE

USER INPUT: "List my patients." 
Reply:
SAFE
---
USER INPUT: "{{text}}"
Reply:
"""

OUTPUT_PROMPT_TEMPLATE = """
Is the following CHATBOT RESPONSE generally unsafe?
Unsafe means: illegal, promotes self-harm/violence, hate speech, contains computer code, or is nonsensical.

CHATBOT RESPONSE: "{{text}}"

Reply with exactly one word:
SAFE
UNSAFE
---
CHATBOT RESPONSE: "{{text}}"
Reply:
"""

# --- End improved prompts ---


# Create proper PromptTemplates for use with pipe operators
input_prompt = PromptTemplate(template=INPUT_PROMPT_TEMPLATE, input_variables=["text"])
output_prompt = PromptTemplate(
    template=OUTPUT_PROMPT_TEMPLATE, input_variables=["text"]
)

parser = StrOutputParser()  # returns raw string


def _check(
    prompt_template: PromptTemplate,
    text_to_check: str,
) -> bool:
    """Check if text_to_check is safe using the provided prompt template."""
    if not text_to_check:
        log.debug("Guard _check: Empty text, assuming safe.")
        return True

    chain = prompt_template | moderator | parser
    # No user_role or user_input needed for these simple prompts
    verdict_raw = chain.invoke({"text": text_to_check})

    log.info(
        f"Guard _check internal: Input='{text_to_check[:70]}...', Raw LLM Verdict='{verdict_raw!r}'"
    )

    if not verdict_raw or not verdict_raw.strip():
        log.warning("Guard _check internal: Received empty verdict. Assuming unsafe.")
        return False

    first_line = verdict_raw.strip().lower()  # Simpler parsing for single-word response

    is_safe = first_line == "safe"

    if not is_safe and first_line != "unsafe":
        log.warning(
            f"Guard _check internal: Verdict ('{first_line}') was not 'safe' or 'unsafe'. Defaulting to unsafe."
        )
        return False  # Default to unsafe if format is not strictly followed

    log.info(
        f"Guard _check internal: Parsed verdict='{first_line}'. Final decision: is_safe={is_safe}"
    )
    return is_safe


def _extract_last_reply(state: dict) -> str:
    """Return content of the last AI/Tool message, or ''."""
    messages = state.get("messages", [])
    if not messages:
        return ""
    for m in reversed(messages):
        if isinstance(m, (AIMessage, ToolMessage)):
            content = getattr(m, "content", None)
            return str(content) if content is not None else ""
    return ""  # No AI or Tool message found


# ─────────────────────────────────────────────────────────────────────────
# 2.  Runnable nodes — each returns **a NEW state dict**
# ─────────────────────────────────────────────────────────────────────────
def guard_in(state: dict) -> dict:
    """Block unsafe user input."""
    current_input = state.get("current_input", "")
    if not isinstance(current_input, str):
        current_input = ""

    if not current_input:
        log.debug("guard_in: Empty input, safe.")
        state["final_output"] = None
        state["agent_name"] = None
        return state

    if not _check(input_prompt, current_input):
        rejection_message = (
            "Sorry, your request cannot be processed due to safety guidelines."
        )
        state["final_output"] = rejection_message
        state["agent_name"] = "Input Guardrail"
        log.warning(f"Input guardrail triggered. Input: '{current_input[:70]}...'.")
    else:
        log.info(f"Input guardrail passed. Input: '{current_input[:70]}...'")
        state["final_output"] = None
        state["agent_name"] = None
    return state  # input is safe, continue to agent node


def guard_out(state: dict) -> dict:
    """Sanitise assistant answer and log details."""
    last_reply_text = _extract_last_reply(state)
    if not last_reply_text:
        log.debug("guard_out: Empty last reply, safe.")
        state["final_output"] = ""
        return state  # Keep original agent_name if any

    if not _check(output_prompt, last_reply_text):
        log.warning(
            f"Output guardrail triggered. Bot Reply: '{last_reply_text[:70]}...'. Overwriting."
        )
        state["final_output"] = (
            "I'm sorry, but I'm unable to provide that response due to system guidelines."
        )
        state["agent_name"] = "Output Guardrail"
    else:
        log.info(f"Output guardrail passed. Bot Reply: '{last_reply_text[:70]}...'")
        state["final_output"] = last_reply_text  # Pass through the original reply
        # Do not change agent_name if it passed
    return state