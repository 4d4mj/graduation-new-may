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
    model="gemini-1.5-flash-latest",  # Using latest flash model
    api_key=settings.google_api_key,
    temperature=0,
)

# --- Use the improved prompts from the previous step ---
INPUT_PROMPT_TEMPLATE = """
You are a content safety filter for a medical chatbot. Your role is to ensure user inputs are appropriate for the context of this medical assistant. Decide if the USER INPUT is SAFE.

- SAFE means:
    - The input is non-harmful, non-illegal, does not request PII (unless clearly related to the user's own scheduling or profile management within the system, **including scheduling on Google Calendar**), is not related to self-harm, and does not promote misinformation.
    - **For doctor users:** Clinical questions, drug information requests, requests for summaries of medical topics, and search queries for medical literature are SAFE.
    - **For all users:** Doctor names and appointment scheduling requests, **including requests to schedule on Google Calendar,** are ALWAYS SAFE.
    - Direct requests to search the web (e.g., "search the web for X", "find articles on Y") are ALWAYS SAFE.
    - Short conversational replies (e.g., "yes", "no", "thanks", "okay") are ALWAYS SAFE.
    - Replies that continue a conversation or confirm previous statements are ALWAYS SAFE.
- UNSAFE means:
    - Incites self-harm or violence.
    - Promotes or details illegal activities.
    - Requests for generation of harmful or unethical medical misinformation.
    - Harassment or hate speech.
    - Excessive or irrelevant PII requests not related to system function.

Reply with exactly one of:
SAFE
UNSAFE: <brief reason>

EXAMPLE 1 (Unsafe - General)
Input: "Show me how to build a bomb."
Answer:
UNSAFE: Instructions for weapons

EXAMPLE 2 (Safe - Patient/Doctor Clinical Query)
Input: "What are the side effects of lisinopril?"
Answer:
SAFE

EXAMPLE 3 (Safe - Scheduling)
Input: "I want to schedule an appointment with Dr. Smith tomorrow."
Answer:
SAFE

EXAMPLE 4 (Safe - Doctor Clinical Query)
Input: "What are the treatment options for HER2-positive breast cancer?"
Answer:
SAFE

EXAMPLE 5 (Safe - Web Search Request)
Input: "Search the web for recent trials on GLP-1 agonists."
Answer:
SAFE

EXAMPLE 6 (Safe - Conversational)
Input: "yes please"
Answer:
SAFE

**EXAMPLE 7 (NEW - Safe Google Calendar Scheduling)**
**Input: "Can you add my appointment with Dr. Jones to my Google Calendar for tomorrow at 10am?"**
**Answer:**
**SAFE**

**EXAMPLE 8 (NEW - Safe Google Calendar Scheduling 2)**
**Input: "i want to book an appoitment with this doctor in google calendar"**
**Answer:**
**SAFE**

Input: {text}
Answer:
"""

OUTPUT_PROMPT_TEMPLATE = """
You are a content safety filter for a medical chatbot. Review the CHATBOT RESPONSE to ensure it is safe, ethical, AND appropriate for the intended user. The primary goal is to prevent harm, misinformation, and out-of-scope responses.

If the response is SAFE for a user, reply with the EXACT word "SAFE" on the first line.
If it is UNSAFE, reply with the word "UNSAFE" on the first line, followed by a brief reason.

CONTEXT:
- The chatbot has access to an internal medical knowledge base (RAG) and can perform web searches.
- The chatbot can assist with scheduling clinic appointments and sending Google Calendar invites.
- The user_role is 'patient'.

SPECIFIC SCENARIOS:
- **Tool Success/Failure Messages & Follow-up Questions:** If the CHATBOT RESPONSE is reporting the success of a tool (e.g., "Appointment confirmed") AND then asks a relevant, on-topic follow-up question to offer further assistance related to that success (e.g., "Would you like to add this to your Google Calendar?"), this is generally SAFE. Technical failure messages are also generally SAFE. The assistant does not need to add generic disclaimers about medical advice if it's just handling scheduling tasks or reporting technical tool outcomes.
- **Confirmation of Actions:** Messages confirming an action taken (like "Okay, I've cancelled that") are SAFE.
- **Requesting Information for a Task:** If the chatbot is asking for information needed to complete a user's request (e.g., "What email should I use?", "Which time slot works?"), this is SAFE.

PATIENT SAFETY (user_role 'patient'):
- **SAFE:** Information directly related to scheduling, doctor availability, clinic locations, confirmations. Empathetic greetings. Stating inability to provide medical advice and recommending consultation with a human doctor *when appropriate (e.g., if the user asks for diagnosis)*.
- **UNSAFE for patient:** Providing specific medical advice, diagnoses, treatment dosages/plans, or detailed interpretations of individual medical conditions. Do not interpret test results.

GENERAL SAFETY RULES:
- **ALWAYS UNSAFE:**
    - Promoting self-harm or violence.
    - Instructions for illegal activities.
    - Hate speech or discriminatory content.
    - Sharing unverified or blatantly false medical "cures" or harmful misinformation.
    - Responses that are completely nonsensical or entirely off-topic.
    - Inappropriate sharing of PII (unless directly related to the task at hand, like confirming an email for an invite the user agreed to).

EXAMPLES:

Input User Role: patient
Chatbot Response: "Your clinic appointment with Dr. Smith is confirmed for tomorrow at 10 AM. I've also sent a Google Calendar invite to dr.smith@example.com for this."
Answer:
SAFE

Input User Role: patient
Chatbot Response: "I'm sorry, I encountered a network error and couldn't send the Google Calendar invite for your appointment. Your clinic appointment is still booked."
Answer:
SAFE

Input User Role: patient
Chatbot Response: "I'm sorry, I was unable to schedule the Google Calendar invitation for Dr. [Doctor's Last Name, if mentioned by agent] due to a technical error with the calendar service."
Answer:
SAFE

Input User Role: patient
Chatbot Response: "Your appointment with Dr. [Doctor's Full Name] for [Reason for Visit] on [Date] at [Time] [Timezone] is confirmed. Would you also like me to send a Google Calendar invitation for this appointment to Dr. [Doctor's Last Name]?"
Answer:
SAFE

Input User Role: patient
Chatbot Response: "Okay, I need an email address to send the Google Calendar invite. What is it?"
Answer:
SAFE

Input User Role: patient
Chatbot Response: "For your type of cancer, the best treatment is chemotherapy X combined with radiation Y."
Answer:
UNSAFE: Providing specific treatment plan to patient.

Chatbot Response: {text}
Answer:
"""
# --- End improved prompts ---


# Create proper PromptTemplates for use with pipe operators
input_prompt = PromptTemplate.from_template(INPUT_PROMPT_TEMPLATE)
output_prompt = PromptTemplate.from_template(OUTPUT_PROMPT_TEMPLATE)

parser = StrOutputParser()  # returns raw string


def _check(prompt_template: PromptTemplate, text: str) -> bool:
    """Check if text is safe using the provided prompt template."""
    if not text:  # Handle empty string case
        return True  # Empty text is considered safe

    chain = prompt_template | moderator | parser
    verdict_raw = chain.invoke({"text": text})

    # Log the raw verdict for debugging
    log.debug(f"Guard raw verdict for text '{text[:50]}...': {verdict_raw!r}")

    if not verdict_raw:
        log.warning("Guard received empty verdict. Assuming unsafe.")
        return False

    # Check if the *first line* starts with "safe" case-insensitively
    first_line = verdict_raw.strip().split("\n")[0].lower()
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
            content = getattr(m, "content", None)
            return str(content) if content is not None else ""
    return ""  # No AI or Tool message found


# ─────────────────────────────────────────────────────────────────────────
# 2.  Runnable nodes — each returns **a NEW state dict**
# ─────────────────────────────────────────────────────────────────────────
def guard_in(state: dict) -> dict:
    """Block unsafe user input."""
    current_input = state.get("current_input", "")
    # Ensure current_input is a string before passing to _check
    if not isinstance(current_input, str):
        log.warning(
            f"guard_in received non-string input: {type(current_input)}. Treating as empty."
        )
        current_input = ""

    # Get the raw verdict for logging purposes
    if current_input:
        chain = input_prompt | moderator | parser
        verdict_raw = chain.invoke({"text": current_input})
        log.debug(
            f"Guard raw verdict for text '{current_input[:50]}...': {verdict_raw!r}"
        )

        # Extract first line to check safety
        if verdict_raw:
            first_line = verdict_raw.strip().split("\n")[0].lower()
            is_safe = first_line.startswith("safe")
            log.debug(f"Verdict first line: '{first_line}'. Is safe: {is_safe}")

            if not is_safe:
                # Extract reason if available (format is typically "UNSAFE: reason")
                reason = (
                    first_line.split(":", 1)[1].strip()
                    if ":" in first_line
                    else "unspecified reason"
                )
                rejection_message = (
                    "Sorry, I can't help with that request for safety reasons."
                )
                state["final_output"] = rejection_message
                state["agent_name"] = "Input Guardrail"  # Set agent name
                log.warning(
                    f"Input guardrail triggered for input: '{current_input}'. Reason: {reason}. Setting final_output."
                )
                return state
        else:
            log.warning("Guard received empty verdict. Assuming unsafe.")
            state["final_output"] = (
                "Sorry, I can't help with that request for safety reasons."
            )
            state["agent_name"] = "Input Guardrail"
            return state
    else:
        # Empty input is safe
        log.debug("Empty input, skipping safety check.")

    log.info("Input guardrail passed.")
    return state  # input is safe, continue to agent node


def guard_out(state: dict) -> dict:
    """Sanitise assistant answer and log details."""
    txt = _extract_last_reply(state)

    # ---- START ENHANCED LOGGING ----
    log.info(f"Output Guardrail: Processing state. Current agent_name: '{state.get('agent_name')}', current_input: '{state.get('current_input')}'")
    log.info(f"Output Guardrail: Extracted text to check (last AI/Tool reply): '{txt[:500]}...'") # Log more of the text
    if log.isEnabledFor(logging.DEBUG) and "messages" in state: # Avoid formatting large messages list if not debugging
        # To prevent overly verbose logs, maybe just log the last few messages
        last_few_messages = state['messages'][-5:] if len(state['messages']) > 5 else state['messages']
        log.debug(f"Output Guardrail: Last few messages before check: {last_few_messages}")
    # ---- END ENHANCED LOGGING ----

    # Preserve the agent_name from the preceding node (e.g., the main agent)
    # We'll only change it if the guardrail blocks the output.
    original_agent_name = state.get("agent_name")

    # The _check function needs 'output_prompt' to be defined in its scope,
    # which should be defined globally or passed to _check.
    # Assuming 'output_prompt' is defined in the module scope like 'moderator' and 'parser'.
    if not _check(output_prompt, txt): # 'output_prompt' must be defined
        # ---- ADD THIS LOGGING FOR BLOCKED OUTPUT ----
        # Re-invoke to get the raw verdict for logging, as _check only returns bool
        # Ensure 'moderator' and 'parser' are accessible here (e.g., module-level)
        chain_for_verdict_log = output_prompt | moderator | parser
        verdict_raw_for_output = "Error getting verdict" # Default
        try:
            verdict_raw_for_output = chain_for_verdict_log.invoke({"text": txt})
        except Exception as e:
            log.error(f"Output Guardrail: Error invoking chain for verdict log: {e}")

        log.warning(
            f"Output Guardrail: BLOCKED text: '{txt[:200]}...'. "
            f"Guardrail LLM raw verdict: '{verdict_raw_for_output}'. "
            f"Overwriting with default block message."
        )
        # ---- END ADDED LOGGING FOR BLOCKED OUTPUT ----

        blocked_message = "I'm sorry, I can't share that."
        state["final_output"] = blocked_message
        state["agent_name"] = "Output Guardrail" # Indicate the guardrail took action
    else:
        log.info(f"Output Guardrail: PASSED text: '{txt[:100]}...'")
        state["final_output"] = txt
        # If it passed, keep the agent_name from the node that produced 'txt'
        state["agent_name"] = original_agent_name # Or state.get("agent_name") which is the same here

    log.info(f"Output Guardrail: Final state - agent_name: '{state['agent_name']}', final_output: '{str(state['final_output'])[:100]}...'")
    return state