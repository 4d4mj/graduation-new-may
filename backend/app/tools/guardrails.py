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
    - The input is non-harmful, non-illegal, does not request PII (unless clearly related to the user's own scheduling or profile management within the system), is not related to self-harm, and does not promote misinformation.
    - **For doctor users:** Clinical questions, drug information requests, requests for summaries of medical topics, and search queries for medical literature are SAFE.
    - **For all users:** Doctor names and appointment scheduling requests are ALWAYS SAFE.
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

Input: {text}
Answer:
"""

OUTPUT_PROMPT_TEMPLATE = """
You are a content safety filter for a medical chatbot. Review the CHATBOT RESPONSE to ensure it is safe, ethical, AND appropriate for the intended user. The primary goal is to prevent harm, misinformation, and out-of-scope responses, while allowing factual, helpful information for the intended user.

If the response is SAFE for a user, reply with the EXACT word "SAFE" on the first line.
If it is UNSAFE, reply with the word "UNSAFE" on the first line, followed by a brief reason.

CONTEXT: The chatbot has access to an internal medical knowledge base (RAG) and can perform web searches.

CONSIDER THE INTENDED USER :

-   **If the user_role is 'doctor':**
    -   **SAFE:** Responses providing detailed, factual medical information, including but not limited to:
        -   Descriptions of diseases, symptoms, etiology, pathophysiology (e.g., "symptoms of breast cancer," "mechanism of hypertension").
        -   Drug information (dosages, side effects, contraindications, mechanisms of action).
        -   Diagnostic criteria or investigation pathways from established guidelines.
        -   Treatment options and protocols from established guidelines or medical literature.
        -   Summaries of medical research, clinical trial results, or information from medical textbooks.
        -   Explanation of complex medical terminology.
        -   Information retrieved from web searches when clearly attributed (e.g., with URLs or "Source: Web Search").
    -   **SAFE:** Questions to the doctor for clarification if the bot needs more information to fulfill a request.
    -   **UNSAFE for doctor (rare, but possible):** Responses that are grossly inaccurate medical statements not supported by any source, or responses that directly encourage illegal/unethical medical practices.

-   **If the user_role is 'patient' (interacting with a scheduling/basic info agent):**
    -   **SAFE:** Information directly related to scheduling appointments, doctor availability, clinic locations, confirmation/cancellation messages.
    -   **SAFE:** Empathetic greetings, confirmations, and polite conversational elements.
    -   **SAFE:** Stating inability to provide medical advice, diagnosis, or treatment plans and recommending consultation with a human doctor.
    -   **UNSAFE for patient:** Providing specific medical advice, diagnoses, treatment dosages/plans, or detailed interpretations of individual medical conditions. Do not interpret test results.

GENERAL SAFETY RULES (apply to both user roles):
-   **ALWAYS UNSAFE:**
    -   Promoting self-harm or violence.
    -   Instructions for illegal activities.
    -   Hate speech or discriminatory content.
    -   Sharing unverified or blatantly false medical "cures" or harmful misinformation.
    -   Responses that are completely nonsensical or entirely off-topic from the user's query and the medical context.
    -   Inappropriate sharing of PII (beyond what's strictly necessary for the system's function like scheduling).

EXAMPLES:

Input User Role: doctor
Chatbot Response: "The Frank-Starling mechanism describes how an increase in ventricular preload leads to an increased stroke volume, up to a physiological limit. This is due to more optimal actin-myosin overlap. [Source: Guyton_Physiology.pdf]"
Answer:
SAFE

Input User Role: doctor
Chatbot Response: "Common symptoms of breast cancer include a new lump, skin changes like dimpling or peau d'orange, and nipple retraction or discharge. Many cases are also asymptomatic. [Source: Internal KB; Web Search]"
Answer:
SAFE

Input User Role: doctor
Chatbot Response: "Treatment options for metastatic non-small cell lung cancer with an EGFR exon 19 deletion include Osimertinib. [Source: NCCN Guidelines / WebMD.com/some_article]"
Answer:
SAFE

Input User Role: patient
Chatbot Response: "You mentioned a headache. I can't tell you what's causing it, but Dr. Smith has an opening at 2 PM tomorrow if you'd like to book an appointment to discuss it."
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
    """Sanitise assistant answer."""
    txt = _extract_last_reply(state)

    if not _check(output_prompt, txt):
        log.warning(
            f"Output guardrail triggered for text: '{txt[:100]}...'. Overwriting output."
        )
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
