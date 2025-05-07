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
    model="gemini-1.5-flash-latest", # Using latest flash model
    api_key=settings.google_api_key,
    temperature=0,
)

# --- Use the improved prompts from the previous step ---
INPUT_PROMPT_TEMPLATE = """
You are a content safety filter for a medical chatbot.  Decide if the USER INPUT is SAFE.

- SAFE means: non-harmful, non-illegal, non-PII, non-self-harm, non-misinformation.
- UNSAFE means: self-harm, medical misinformation, illegal instructions, PII requests, etc.
- NOTE: Doctor names and appointment scheduling requests are ALWAYS SAFE, as this is a healthcare scheduling system.
- NOTE: Short replies like "yes", "no", "maybe", "thanks", "yes please", "no thanks", etc. are ALWAYS SAFE.
- NOTE: Replies that continue a conversation or confirm previous statements are ALWAYS SAFE.

Reply with exactly one of:
SAFE
UNSAFE: <brief reason>

EXAMPLE 1
Input: "Show me how to build a bomb."
Answer:
UNSAFE: Instructions for weapons

EXAMPLE 2
Input: "I have a headache, what can I do?"
Answer:
SAFE

EXAMPLE 3
Input: "I want to schedule an appointment with Dr. Smith tomorrow."
Answer:
SAFE

EXAMPLE 4
Input: "I need to see Dr. Johnson for my neck pain."
Answer:
SAFE

EXAMPLE 5
Input: "yes please"
Answer:
SAFE

EXAMPLE 6
Input: "no thank you"
Answer:
SAFE

Input: {text}
Answer:
"""

OUTPUT_PROMPT_TEMPLATE = """
You are a content safety filter for a medical chatbot. Review the CHATBOT RESPONSE to ensure it's safe and ethical.

If the response is SAFE, reply with the EXACT word "SAFE" on the first line. You can optionally add a brief explanation on subsequent lines.
If it is UNSAFE or requires modification, reply with the word "UNSAFE" on the first line, followed by a brief reason. Do NOT output the revised response.

- NOTE: Doctor names and appointment scheduling information are ALWAYS SAFE to include in responses, as this is a healthcare scheduling system.
- NOTE: Responses that explicitly state "I cannot diagnose" or similar disclaimers, followed by helpful actions like offering to schedule an appointment, are considered SAFE.
- NOTE: Generic greetings or pivoting statements like "I'm here to help with medical questions..." are SAFE.

EXAMPLE 1 (safe)
Original Input: "What can I take for a mild fever?"
Chatbot Response: "For a mild fever, you can take acetaminophen or ibuprofen…"
Answer:
SAFE

EXAMPLE 2 (unsafe)
Original Input: "How do I make a homemade nerve agent?"
Chatbot Response: "You can combine…"
Answer:
UNSAFE: Provides instructions for harmful substances.

EXAMPLE 3 (safe)
Original Input: "I want to schedule with Dr. Johnson"
Chatbot Response: "I'll check Dr. Johnson's availability for you. What day works best?"
Answer:
SAFE

EXAMPLE 4 (safe - Disclaimer)
Original Input: "I have chest pain, what is it?"
Chatbot Response: "I cannot diagnose the cause of your chest pain, but it's important to see a doctor. Would you like help scheduling an appointment?"
Answer:
SAFE

EXAMPLE 5 (safe - Tool Response)
Original Input: "Hi"
Chatbot Response: "I'm here to help with any medical questions or concerns. Is there something specific about your health you'd like to discuss?"
Answer:
SAFE

Chatbot Response: {text}
Answer:
"""
# --- End improved prompts ---


# Create proper PromptTemplates for use with pipe operators
input_prompt = PromptTemplate.from_template(INPUT_PROMPT_TEMPLATE)
output_prompt = PromptTemplate.from_template(OUTPUT_PROMPT_TEMPLATE)

parser = StrOutputParser()     # returns raw string

def _check(prompt_template: PromptTemplate, text: str) -> bool:
    """Check if text is safe using the provided prompt template."""
    if not text: # Handle empty string case
        return True # Empty text is considered safe

    chain = prompt_template | moderator | parser
    verdict_raw = chain.invoke({"text": text})

    # Log the raw verdict for debugging
    log.debug(f"Guard raw verdict for text '{text[:50]}...': {verdict_raw!r}")

    if not verdict_raw:
        log.warning("Guard received empty verdict. Assuming unsafe.")
        return False

    # Check if the *first line* starts with "safe" case-insensitively
    first_line = verdict_raw.strip().split('\n')[0].lower()
    is_safe = first_line.startswith("safe")

    log.debug(f"Verdict first line: '{first_line}'. Is safe: {is_safe}")
    return is_safe

def _extract_last_reply(state: dict) -> str:
    """Return content of the last AI/Tool message, or ''."""
    messages = state.get("messages", [])
    if not messages:
        return ""
    # Iterate in reverse to find the most recent AI or Tool message
    for m in reversed(messages):
        if isinstance(m, (AIMessage, ToolMessage)):
            # Ensure content is treated as string, handle None
            content = getattr(m, 'content', None)
            return str(content) if content is not None else ""
    return "" # No AI or Tool message found

# ─────────────────────────────────────────────────────────────────────────
# 2.  Runnable nodes — each returns **a NEW state dict**
# ─────────────────────────────────────────────────────────────────────────
def guard_in(state: dict) -> dict:
    """Block unsafe user input."""
    current_input = state.get("current_input", "")
    # Ensure current_input is a string before passing to _check
    if not isinstance(current_input, str):
        log.warning(f"guard_in received non-string input: {type(current_input)}. Treating as empty.")
        current_input = ""

    # Get the raw verdict for logging purposes
    if current_input:
        chain = input_prompt | moderator | parser
        verdict_raw = chain.invoke({"text": current_input})
        log.debug(f"Guard raw verdict for text '{current_input[:50]}...': {verdict_raw!r}")

        # Extract first line to check safety
        if verdict_raw:
            first_line = verdict_raw.strip().split('\n')[0].lower()
            is_safe = first_line.startswith("safe")
            log.debug(f"Verdict first line: '{first_line}'. Is safe: {is_safe}")

            if not is_safe:
                # Extract reason if available (format is typically "UNSAFE: reason")
                reason = first_line.split(':', 1)[1].strip() if ':' in first_line else "unspecified reason"
                rejection_message = "Sorry, I can't help with that request for safety reasons."
                state["final_output"] = rejection_message
                state["agent_name"] = "Input Guardrail"  # Set agent name
                log.warning(f"Input guardrail triggered for input: '{current_input}'. Reason: {reason}. Setting final_output.")
                return state
        else:
            log.warning("Guard received empty verdict. Assuming unsafe.")
            state["final_output"] = "Sorry, I can't help with that request for safety reasons."
            state["agent_name"] = "Input Guardrail"
            return state
    else:
        # Empty input is safe
        log.debug("Empty input, skipping safety check.")

    log.info("Input guardrail passed.")
    return state      # input is safe, continue to agent node

def guard_out(state: dict) -> dict:
    """Sanitise assistant answer."""
    txt = _extract_last_reply(state)

    if not _check(output_prompt, txt):
        log.warning(f"Output guardrail triggered for text: '{txt[:100]}...'. Overwriting output.")
        txt = "I'm sorry, I can't share that."
        state["agent_name"] = "Output Guardrail"
    else:
         log.info("Output guardrail passed.")
         # Optional: Clear the agent_name if it was set by the guardrail previously in the same turn?
         # Or rely on the main agent node to set its own name later.
         # If the agent node *always* sets agent_name, this isn't needed.
         pass

    state["final_output"] = txt
    return state
