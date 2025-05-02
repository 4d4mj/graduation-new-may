# app/agents/scheduler/helpers.py
import datetime as dt
import logging
from typing import Dict, Any, List
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import AIMessage
from app.agents.scheduler.parser import create_appointment_parser, AppointmentRequest

logger = logging.getLogger(__name__)

@tool("parse_booking_request")
def parse_booking_request(query: str, user_id: int = 1) -> AppointmentRequest:
    """
    Extract structured appointment data from a natural language booking request.
    Use this tool before calling list_free_slots or book_appointment to extract the date, time preferences, and other details.

    Args:
        query: The user's natural language booking request
        user_id: The ID of the current user (patient)

    Returns:
        A structured representation of the appointment request with patient_id, doctor_id, day, and time_preference
    """
    parser = create_appointment_parser()
    return parser(query, user_id)


def create_mcp_scheduler_tools(mcp_tools: List[BaseTool]) -> List[BaseTool]:
    """
    Create enhanced versions of the MCP scheduler tools that handle natural language better.

    Args:
        mcp_tools: The raw MCP tools from the tool manager

    Returns:
        Enhanced versions of the tools with better descriptions and handling
    """
    # Find the three scheduler tools by name
    list_slots_tool = next((t for t in mcp_tools if t.name == "list_free_slots"), None)
    book_tool = next((t for t in mcp_tools if t.name == "book_appointment"), None)
    cancel_tool = next((t for t in mcp_tools if t.name == "cancel_appointment"), None)

    # If any tool is missing, return the original tools
    if not all([list_slots_tool, book_tool, cancel_tool]):
        logger.warning("One or more scheduler MCP tools are missing")
        return mcp_tools

    # Add the parse_booking_request tool
    enhanced_tools = [parse_booking_request]

    # Enhance the descriptions of the MCP tools
    if list_slots_tool:
        list_slots_tool.description = """
        Find available appointment slots for a doctor on a specific day.
        First use parse_booking_request to extract the doctor_id and day from the patient's request.

        Args:
            doctor_id: The ID of the doctor (integer)
            day: The date to check in ISO format (YYYY-MM-DD)

        Returns:
            A list of available appointment slots
        """
        enhanced_tools.append(list_slots_tool)

    if book_tool:
        book_tool.description = """
        Book a new appointment for a patient with a doctor.
        First use parse_booking_request to understand the patient's scheduling needs.

        Args:
            patient_id: The ID of the patient (integer)
            doctor_id: The ID of the doctor (integer)
            starts_at: The appointment start time (ISO datetime)
            ends_at: The appointment end time (ISO datetime)
            location: The appointment location (string)
            notes: Optional notes about the appointment (string)

        Returns:
            Confirmation of the booking
        """
        enhanced_tools.append(book_tool)

    if cancel_tool:
        cancel_tool.description = """
        Cancel an existing appointment.

        Args:
            appointment_id: The ID of the appointment to cancel (integer)
            patient_id: The ID of the patient for verification (integer)

        Returns:
            Confirmation of the cancellation
        """
        enhanced_tools.append(cancel_tool)

    return enhanced_tools
