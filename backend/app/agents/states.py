# app/agents/state.py
from typing import Any, List
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState

class BaseAgentState(MessagesState):
    # ─── shared by all roles ───────────────────────────────────
    messages: List[BaseMessage]    = []
    current_input: Any | None      = None
    agent_name:  str | None        = None
    final_output: str | None       = None
    needs_human_validation: bool   = False

class PatientState(BaseAgentState):
    # ─── patient-only fields ──────────────────────────────────
    retrieval_confidence: float    = 0.0
    insufficient_info:    bool     = False
    request_scheduling:   bool     = False
    next_agent:           str | None = None

class DoctorState(BaseAgentState):
    # ─── doctor-only fields ───────────────────────────────────
    retrieval_confidence: float    = 0.0
    web_search_results:  str | None= None
    generate_report_result: str | None = None

def init_state_for_role(role: str) -> BaseAgentState:
    if role == "patient":
        return PatientState()
    elif role == "doctor":
        return DoctorState()
    else:
        return BaseAgentState()
