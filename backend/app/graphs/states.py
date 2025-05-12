# app/agents/state.py
from typing import Any, List, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import Annotated
from langgraph.graph import MessagesState
from pydantic import Field
from datetime import datetime, timezone

class BaseAgentState(MessagesState):
    # ─── shared by all roles ───────────────────────────────────
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    current_input: Any | None = None
    agent_name: str | None = None
    final_output: str | None = None
    needs_human_validation: bool = False
    user_id: str | None = None
    user_tz: str | None = None
    now: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
class PatientState(BaseAgentState):
    # ─── patient-only fields ──────────────────────────────────
    request_scheduling: bool = False
    next_agent: str | None = None
    remaining_steps: int = 10
    # To store the raw output of a tool intended for direct frontend consumption
    raw_tool_output: Any = None
    # Flag to indicate that the response is a direct tool output
    is_direct_tool_response: bool = False

class DoctorState(BaseAgentState):
    # ─── doctor-only fields ───────────────────────────────────
    retrieval_confidence: float = 0.0
    web_search_results: str | None = None
    generate_report_result: str | None = None
    remaining_steps: int = 10

def init_state_for_role(role: str) -> BaseAgentState:
    if role == "patient":
        return PatientState()
    elif role == "doctor":
        return DoctorState()
    else:
        return BaseAgentState()

class SchedulerState(PatientState):          # or DoctorState
    pending_booking: Optional[dict] = None   # {"doctor":..., "starts_at":...}
