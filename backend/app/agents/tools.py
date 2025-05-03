import logging
from typing import Dict, Any, Union
from langchain_core.tools import tool, Tool, StructuredTool
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from typing import Optional

from app.agents.rag.core import DummyRAG
from app.agents.web_search.core import DummyWebSearch
# Import the scheduler tools
from app.agents.scheduler.tools import list_free_slots, book_appointment, cancel_appointment
from app.config.agent import settings as agentSettings

logger = logging.getLogger(__name__)

@tool("rag_query", return_direct=False)
def rag_query(query: str) -> str:
    """
    Retrieve medical knowledge for a patient query from a trusted medical database.
    Use this when the user asks about medical conditions, treatments, or general health information.
    """
    rag = DummyRAG()
    resp = rag.process_query(query)
    return resp["response"].content if hasattr(resp["response"], "content") else str(resp["response"])

@tool("web_search", return_direct=False)
def web_search(query: str) -> str:
    """
    Perform a web search for current medical information that might not be in our database.
    Use this for recent medical news, emerging treatments, or when you need supplementary information.
    """
    web = DummyWebSearch()
    result = web.process_web_search_results(query)
    return result.content if hasattr(result, "content") else str(result)

@tool("small_talk", return_direct=True)
def small_talk(user_message: str) -> str:
    """
    Handle general conversation, greetings, and non-medical chat.
    Use this for casual conversation or when the patient is making small talk.
    """
    return "I'm here to help with any medical questions or concerns. Is there something specific about your health you'd like to discuss?"

# Define Pydantic schemas for the scheduler tools
class ListFreeSlotsInput(BaseModel):
    doctor_name: Optional[str] = Field(None, description="The name of the doctor to find slots for")
    day: Optional[str] = Field(None, description="Date in YYYY-MM-DD format")

class BookAppointmentInput(BaseModel):
    doctor_name: str = Field(..., description="The name of the doctor to book with")
    starts_at: str = Field(..., description="The start time in ISO format (YYYY-MM-DDTHH:MM:SS)")
    duration_minutes: int = Field(30, description="Duration of the appointment in minutes")
    location: str = Field("Main Clinic", description="Location for the appointment")
    notes: Optional[str] = Field(None, description="Optional notes for the appointment")

class CancelAppointmentInput(BaseModel):
    appointment_id: int = Field(..., description="The ID of the appointment to cancel")

# Create the StructuredTool objects directly using the original scheduler functions
list_slots_tool = StructuredTool.from_function(
    func=list_free_slots,
    name="list_free_slots",
    description=list_free_slots.__doc__,
    args_schema=ListFreeSlotsInput,
    coroutine=list_free_slots,
    return_direct=False
)

book_appointment_tool = StructuredTool.from_function(
    func=book_appointment,
    name="book_appointment",
    description=book_appointment.__doc__,
    args_schema=BookAppointmentInput,
    coroutine=book_appointment,
    return_direct=False
)

cancel_appointment_tool = StructuredTool.from_function(
    func=cancel_appointment,
    name="cancel_appointment",
    description=cancel_appointment.__doc__,
    args_schema=CancelAppointmentInput,
    coroutine=cancel_appointment,
    return_direct=False
)
