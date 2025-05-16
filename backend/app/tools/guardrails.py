# app/agents/guardrails.py
from __future__ import annotations
import logging
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.settings import settings

from typing import Optional

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
You are an AI content safety filter for a medical chatbot. Your task is to determine if the USER INPUT is SAFE or UNSAFE based on their role and the content.

**Your entire response MUST be ONLY ONE of the following two formats:**
1.  **SAFE**
2.  **UNSAFE: <brief reason for being unsafe, 10 words or less>**

---
User Role to consider: {{user_role}}
User Input to evaluate: {{text}}
---

**Instructions for Determining Safety:**

*   **Always SAFE Inputs (If the input matches these, respond with only "SAFE"):**
    *   Simple greetings (e.g., "hello", "hi", "hey", "good morning").
    *   Simple expressions of gratitude (e.g., "thanks", "thank you").
    *   Simple affirmatives/negatives (e.g., "yes", "no", "ok", "sure", "please").
    *   Short replies that continue a conversation or confirm previous statements (e.g., providing an email when asked).
    *   Direct requests to search the web (e.g., "search the web for X", "find articles on Y").

*   **General Unsafe Content (If input matches these, respond with "UNSAFE: <reason>"; these apply to ALL roles):**
    *   Incites self-harm or violence.
    *   Promoting or detailing illegal activities.
    *   Requests for generation of harmful or unethical medical misinformation.
    *   Harassment or hate speech.
    *   Generating code or scripts in any programming language.
    *   Excessive or irrelevant PII requests not directly related to the system's function or not directly solicited by the assistant for an ongoing task.

*   **Role-Specific Safety & PII (If not an "Always SAFE Input" or "General Unsafe Content", evaluate based on role and then respond with "SAFE" or "UNSAFE: <reason>"):**

    *   **If user_role is 'doctor':**
        *   SAFE: Clinical questions, drug information requests, medical topic summaries, literature searches.
        *   SAFE: Queries about specific patients they are authorized to access (e.g., "Jane Doe's records?", "Appointments for John Smith?"). Downstream systems handle actual data authorization.
        *   SAFE: Scheduling requests for themselves or patients, including using Google Calendar.
        *   UNSAFE: Any of the "General Unsafe Content" above. Requests for PII of individuals not clearly relevant to patient care duties.

    *   **If user_role is 'patient':**
        *   SAFE: Questions about their own appointments, scheduling, general clinic/doctor info.
        *   SAFE: Descriptions of their own symptoms for scheduling purposes.
        *   SAFE: Requests related to scheduling appointments, including using Google Calendar for themselves or coordinating with a doctor (e.g., "book my appointment with Dr. X on Google Calendar"). PII is acceptable if clearly related to the user's own scheduling or profile management, or if providing information directly solicited by the assistant for scheduling.
        *   UNSAFE: Asking for diagnosis (e.g., "What illness do I have?").
        *   UNSAFE: Asking for specific treatment plans or prescriptions (e.g., "What medication should I take?").
        *   UNSAFE: Detailed medical questions that constitute seeking advice beyond basic information for scheduling or understanding clinic procedures.
        *   UNSAFE: Any of the "General Unsafe Content" above.

**Examples (Follow this format strictly):**

User Role: patient
Input: "hi"
Answer:
SAFE

User Role: doctor
Input: "Search the web for recent trials on GLP-1 agonists."
Answer:
SAFE

User Role: patient
Input: "Show me how to build a bomb."
Answer:
UNSAFE: Instructions for weapons.

User Role: doctor
Input: "What are the side effects of lisinopril?"
Answer:
SAFE

User Role: patient
Input: "I want to schedule an appointment with Dr. Smith tomorrow."
Answer:
SAFE

User Role: patient
Input: "Can you add my appointment with Dr. Jones to my Google Calendar for tomorrow at 10am?"
Answer:
SAFE

User Role: patient
Input: "i want to book an appoitment with this doctor in google calendar"
Answer:
SAFE

User Role: doctor
Input: "Generate a python script to analyze patient data."
Answer:
UNSAFE: Generating code or scripts.

User Role: patient
Input: "I think I have the flu, what should I take?"
Answer:
UNSAFE: Patient seeking treatment advice.

---
Considering the instructions and examples above:
User Role: {{user_role}}
User Input: {{text}}
Answer:
"""

OUTPUT_PROMPT_TEMPLATE = """
You are a content safety filter for a medical chatbot. Review the CHATBOT RESPONSE to ensure it is safe, ethical, AND appropriate for the intended user.
The user's role is: **{{user_role}}**
The original user input was: **{{user_input}}**

If the response is SAFE and appropriate for the {{user_role}}, reply with the EXACT word "SAFE" on the first line.
If it is UNSAFE or requires modification for the {{user_role}}, reply with the word "UNSAFE" on the first line, followed by a brief reason why it's unsafe or inappropriate for this user role.

CONTEXT:
- The chatbot has access to an internal medical knowledge base (RAG), web search, and (for doctors) a patient database.
- The chatbot's primary functions are:
    - For Patients: Appointment scheduling, providing general non-diagnostic information about services/doctors, and sending Google Calendar invites.
    - For Doctors: Clinical information retrieval (RAG/Web), accessing data about *their own* patients from the database.

SAFETY & APPROPRIATENESS RULES:

-   **If user_role is 'doctor':**
    -   **SAFE:** Responses providing detailed, factual medical information from RAG/Web (e.g., diseases, drugs, treatments, diagnostics).
    -   **SAFE:** Responses providing factual data about specific patients retrieved from the database *if the user_input indicated a query about a specific patient*.
    -   **SAFE:** Questions to the doctor for clarification.
    -   **SAFE:** Polite conversational elements.
    -   **UNSAFE (rare):** Grossly inaccurate medical statements, encouraging illegal/unethical medical practices, or attempting to share data about patients the doctor shouldn't have access to (though primary authorization is handled by tools).

-   **If user_role is 'patient':**
    -   **SAFE:** Information directly related to scheduling appointments, doctor availability, clinic locations, confirmations, and follow-up actions like sending Google Calendar invites.
    -   **SAFE:** Tool success/failure messages and relevant, on-topic follow-up questions (e.g., "Appointment confirmed. Would you like me to send a Google Calendar invite?").
    -   **SAFE:** Empathetic greetings, confirmations, polite conversational elements.
    -   **SAFE:** Stating inability to provide medical advice, diagnosis, or treatment plans and recommending consultation with a human doctor when appropriate.
    -   **UNSAFE:** Providing specific medical advice, diagnoses, treatment dosages/plans, or detailed interpretations of individual medical conditions or test results.

GENERAL SAFETY RULES (apply to both):
-   **ALWAYS UNSAFE:**
    - Promoting self-harm/violence, illegal activities, hate speech, harmful misinformation, nonsensical/off-topic responses.
    - Inappropriate sharing of PII (beyond system function).

EXAMPLES:

User Role: doctor
User Input: "What are the latest labs for Jane?"
Chatbot Response: "Jane Doe's latest Creatinine is 1.2 mg/dL (as of 2024-05-10), and Potassium is 4.0 mEq/L."
Answer:
SAFE

User Role: patient
User Input: "I have a headache."
Chatbot Response: "For a headache, you could try taking 2 tablets of SuperDrug X and resting."
Answer:
UNSAFE: Providing specific medication advice to patient.

User Role: patient
User Input: "Hello"
Chatbot Response: "Hello! How can I help you schedule an appointment today?"
Answer:
SAFE

User Role: patient
User Input: "What causes migraines?"
Chatbot Response: "Migraines are complex neurological events often involving ..."
Answer:
UNSAFE: Providing detailed medical explanation to patient that could be misconstrued as diagnosis/advice. Should be: "I can help you find a doctor to discuss migraines. Would you like to schedule an appointment?"

Chatbot Response: {{text_to_check}}
Answer:
"""

# --- End improved prompts ---


# Create proper PromptTemplates for use with pipe operators
input_prompt = PromptTemplate.from_template(INPUT_PROMPT_TEMPLATE)
output_prompt = PromptTemplate.from_template(OUTPUT_PROMPT_TEMPLATE)

parser = StrOutputParser()  # returns raw string


def _check(
    prompt_template: PromptTemplate,
    text_to_check: str,
    user_role: str,
    user_input_for_output_guard: Optional[str] = None,
) -> bool:
    """Check if text_to_check is safe using the provided prompt template."""
    if not text_to_check:  # Handle empty string case
        log.debug(
            f"Guard _check: Empty text_to_check for role '{user_role}', assuming safe."
        )
        return True  # Empty text_to_check is considered safe

    chain = prompt_template | moderator | parser

    invoke_payload = {"text": text_to_check, "user_role": user_role}
    if user_input_for_output_guard is not None:
        invoke_payload["user_input"] = user_input_for_output_guard

    verdict_raw = chain.invoke(invoke_payload)

    # Log the raw verdict for debugging
    log.debug(
        f"Guard raw verdict for role '{user_role}', text '{text_to_check[:50]}...': {verdict_raw!r}"
    )

    if not verdict_raw:
        log.warning(
            f"Guard _check: Received empty verdict for role '{user_role}'. Assuming unsafe."
        )
        return False

    # Check if the *first line* starts with "safe" case-insensitively
    first_line = verdict_raw.strip().split("\n")[0].lower()
    is_safe = first_line.startswith("safe")

    log.debug(
        f"Guard _check: Verdict first line for role '{user_role}': '{first_line}'. Is safe: {is_safe}"
    )
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
    """Block unsafe user input based on role."""
    current_input = state.get("current_input", "")
    user_role = state.get("role")

    if not user_role:
        log.error(
            "guard_in: 'role' not found in state. Defaulting to 'patient'. This indicates an issue in state population."
        )
        user_role = "patient"

    # Ensure current_input is a string before passing to _check
    if not isinstance(current_input, str):
        log.warning(
            f"guard_in received non-string input for role '{user_role}': {type(current_input)}. Treating as empty."
        )
        current_input = ""

    if not current_input:
        log.debug(
            f"guard_in: Empty input for role '{user_role}', skipping safety check (considered safe)."
        )
        return state

    # Call the updated _check function, passing the user_role
    # The 'text' key in the prompt matches the 'text_to_check' variable name in _check
    # and the {{text}} placeholder in INPUT_PROMPT_TEMPLATE.
    if not _check(
        prompt_template=input_prompt, text_to_check=current_input, user_role=user_role
    ):
        # Logic for unsafe input
        rejection_message = "Sorry, I can't help with that request due to safety guidelines for your role."

        # To understand why it was blocked, we can try to get the reason from the LLM's raw output
        # This is optional for functionality but good for debugging.
        chain = input_prompt | moderator | parser
        verdict_raw_for_reason = chain.invoke(
            {"text": current_input, "user_role": user_role}
        )  # invoke_payload expects "text"
        reason = "unspecified"
        if verdict_raw_for_reason and ":" in verdict_raw_for_reason:
            try:
                reason = (
                    verdict_raw_for_reason.strip()
                    .split("\n")[0]
                    .lower()
                    .split(":", 1)[1]
                    .strip()
                )
            except IndexError:
                pass  # Keep 'unspecified' if parsing fails

        state["final_output"] = rejection_message
        state["agent_name"] = (
            f"Input Guardrail ({user_role})"  # Reflect role in agent name
        )
        log.warning(
            f"Input guardrail triggered for role '{user_role}', input: '{current_input[:100]}...'. Reason: {reason}. Setting final_output."
        )
        # IMPORTANT: If input is unsafe, the graph should typically end or route to a specific "unsafe_input_handler" node.
        # Setting final_output is one way to signal this. How LangGraph handles this depends on your graph structure.
        # (The route_after_guard_in function handles this by checking state["final_output"])
    else:
        log.info(
            f"Input guardrail passed for role '{user_role}', input: '{current_input[:50]}...'."
        )
        # Ensure final_output is None if input is safe, so route_after_guard_in proceeds to agent
        state["final_output"] = None
        state["agent_name"] = None  # Clear any guardrail agent name if input is safe

    return state


def guard_out(state: dict) -> dict:
    """Sanitise assistant answer and log details."""
    last_reply_text = _extract_last_reply(state)
    user_role = state.get("role", "patient")  # Default to 'patient' as per friend's version
    original_user_input = state.get("current_input", "")
    original_agent_name = state.get("agent_name") # Preserve original agent name

    # ---- YOUR ENHANCED LOGGING (Slightly adapted) ----
    log.info(f"Output Guardrail: Processing for role '{user_role}'. Current agent_name: '{original_agent_name}', current_input: '{original_user_input}'")
    log.info(f"Output Guardrail: Extracted last AI/Tool reply to check: '{last_reply_text[:500]}...'")
    if log.isEnabledFor(logging.DEBUG) and "messages" in state:
        last_few_messages = state['messages'][-5:] if len(state['messages']) > 5 else state['messages']
        log.debug(f"Output Guardrail: Last few messages before check: {last_few_messages}")
    # ---- END YOUR ENHANCED LOGGING ----

    # Call the _check function (assuming it's adapted for the new arguments)
    # The 'output_prompt' itself needs to be the one that can handle these extra context variables.
    is_safe = _check(
        prompt_template=output_prompt, # Use the correct variable name for the PromptTemplate object
        text_to_check=last_reply_text,
        user_role=user_role,
        user_input_for_output_guard=original_user_input,
    )

    if not is_safe:
        # ---- YOUR LOGGING FOR BLOCKED OUTPUT ----
        # Re-invoke the chain to get the raw verdict string for logging.
        # This assumes 'moderator' and 'parser' are accessible and 'output_prompt' is the correct PromptTemplate.
        chain_for_verdict_log = output_prompt | moderator | parser
        verdict_raw_for_output = "Error getting verdict"
        try:
            # The input to invoke must match the variables expected by output_prompt
            verdict_raw_for_output = chain_for_verdict_log.invoke({
                "text_to_check": last_reply_text, # Or whatever variable name output_prompt expects for the bot's reply
                "user_role": user_role,
                "user_input_for_output_guard": original_user_input,
                # Ensure all variables used in output_prompt's template string are provided here.
                # If output_prompt only uses '{text}', then only pass that:
                # verdict_raw_for_output = chain_for_verdict_log.invoke({"text": last_reply_text})
                # This depends on how output_prompt is defined.
                # For friend's logic, it likely expects all three.
            })
        except Exception as e:
            log.error(f"Output Guardrail: Error invoking chain for verdict log: {e}")
        # ---- END YOUR LOGGING FOR BLOCKED OUTPUT ----

        log.warning(
            f"Output Guardrail: BLOCKED for role '{user_role}', text: '{last_reply_text[:200]}...'. "
            f"Guardrail LLM raw verdict: '{verdict_raw_for_output}'. "
            f"Overwriting output."
        )
        
        # Friend's blocked message and agent_name setting
        final_text_to_user = "I'm sorry, but I'm unable to provide that specific information due to system guidelines."
        state["agent_name"] = f"Output Guardrail ({user_role})"
    else:
        log.info(
            f"Output Guardrail: PASSED for role '{user_role}', output: '{last_reply_text[:100]}...'."
        )
        final_text_to_user = last_reply_text
        # Friend's logic for agent_name on pass
        if not original_agent_name:  # If the main agent itself didn't set a name
            state["agent_name"] = "Assistant" # Or role-specific default
        else:
            state["agent_name"] = original_agent_name # Keep the original agent's name

    state["final_output"] = final_text_to_user
    log.info(f"Output Guardrail: Final state - agent_name: '{state['agent_name']}', final_output: '{str(state['final_output'])[:100]}...'")
    return state