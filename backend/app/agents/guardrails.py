# app/agents/guardrails.py
from __future__ import annotations
import logging
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.settings import settings
from app.agents.states import BaseAgentState     # PatientState, DoctorState, …

log = logging.getLogger(__name__)

# ── 1. choose a tiny, cheap model for safety
moderator = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17",
    api_key=settings.google_api_key,
    temperature=0,
)

INPUT_PROMPT_TEMPLATE = "SAFE or UNSAFE? User said: {text}"
OUTPUT_PROMPT_TEMPLATE = "SAFE or UNSAFE? Assistant will say: {text}"

# Create proper PromptTemplates for use with pipe operators
input_prompt = PromptTemplate.from_template(INPUT_PROMPT_TEMPLATE)
output_prompt = PromptTemplate.from_template(OUTPUT_PROMPT_TEMPLATE)

parser = StrOutputParser()     # returns raw string

def _check(prompt_template: PromptTemplate, text: str) -> bool:
    """Check if text is safe using the provided prompt template."""
    chain = prompt_template | moderator | parser
    verdict = chain.invoke({"text": text})
    return verdict.startswith("SAFE")

# ─────────────────────────────────────────────────────────────────────────
# 2.  Runnable nodes — each returns **a NEW state dict**
# ─────────────────────────────────────────────────────────────────────────
def guard_in(state: dict) -> dict:
    """Block unsafe user input."""
    current_input = state.get("current_input", "")
    if not _check(input_prompt, current_input or ""):
        state["final_output"] = (
            "Sorry, I can't help with that request for safety reasons."
        )
        # Optionally add a single AI message so history isn't empty
        state["messages"] = [AIMessage(content=state["final_output"])]
        return state  # graph can continue straight to END
    return state      # input is safe, continue to agent node

def guard_out(state: dict) -> dict:
    """Sanitise assistant answer."""
    txt = state.get("final_output", "")
    if not _check(output_prompt, txt):
        txt = "I'm sorry, I can't share that."
    state["final_output"] = txt
    return state
