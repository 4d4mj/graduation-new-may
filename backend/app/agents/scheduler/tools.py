"""
Database-powered appointment scheduler tools.

These tools interact with the appointments table in the database
to manage appointment scheduling, booking, and cancellation.
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta, date, time, timezone
from typing import Optional

from langchain_core.tools import tool
from sqlalchemy.ext.asyncio import AsyncSession

# Import the CRUD functions and session utility
from app.db.crud.appointment import (
    get_available_slots_for_day,
    create_appointment,
    delete_appointment,
)
from app.db.session import get_db_session
from app.config.settings import settings

logger = logging.getLogger(__name__)

# Helper to parse date string (add more robust parsing if needed)
def parse_iso_date(day_str: str | None) -> date:
    if day_str:
        try:
            return date.fromisoformat(day_str)
        except ValueError:
            logger.warning(f"Invalid date format '{day_str}'. Defaulting to tomorrow.")
    # Default to tomorrow if None or invalid
    return (datetime.now(timezone.utc) + timedelta(days=1)).date()

# Helper to parse time string and combine with date
def parse_datetime_str(datetime_str: str) -> datetime | None:
    """Parses YYYY-MM-DDTHH:MM:SS or similar ISO format"""
    try:
        # Attempt parsing with timezone, fallback to naive then assume UTC
        dt = datetime.fromisoformat(datetime_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc) # Assume UTC if naive
        return dt.astimezone(timezone.utc) # Convert to UTC
    except ValueError:
        logger.error(f"Invalid datetime format for booking: {datetime_str}")
        return None

# ------------------------------------------------------------------ tools
@tool("list_free_slots", return_direct=False)
async def list_free_slots(doctor_id: int = 1, day: str | None = None) -> str:
    """
    Return a *human-readable* list of 30-minute free slots for a doctor on a given ISO date (YYYY-MM-DD).

    Args:
        doctor_id (int): Internal doctor identifier (default 1 for Dr. Smith).
        day (str | None): Date in YYYY-MM-DD format. Defaults to tomorrow if not provided or invalid.
    """
    target_day = parse_iso_date(day)
    logger.info(f"Tool 'list_free_slots' called for doctor {doctor_id} on {target_day.isoformat()}")

    slots = []
    try:
        # Create a DB session just for this tool call
        async for db in get_db_session(str(settings.database_url)):
            slots = await get_available_slots_for_day(db, doctor_id, target_day)
    except Exception as e:
        logger.error(f"Failed to get DB session or query slots: {e}", exc_info=True)
        return "Sorry, I encountered an error trying to check the schedule. Please try again later."

    if not slots:
        return f"There are no available slots found for Dr. Smith (ID: {doctor_id}) on {target_day.isoformat()}."
    else:
        slot_list = "\n • ".join(slots)
        return f"Here are the available slots for Dr. Smith (ID: {doctor_id}) on {target_day.isoformat()}:\n • {slot_list}"


@tool("book_appointment", return_direct=False)
async def book_appointment(
    patient_id: int,
    doctor_id: int,
    starts_at: str, # Expect ISO format string e.g., "2024-06-05T14:00:00Z"
    duration_minutes: int = 30, # Default duration
    location: str = "Main Clinic",
    notes: str | None = None,
) -> str:
    """
    Books a new appointment for a specified patient with a doctor at a given start time and duration.
    Returns a confirmation or error message.

    Args:
        patient_id (int): The ID of the patient making the booking.
        doctor_id (int): The ID of the doctor.
        starts_at (str): The desired start time in ISO format (e.g., "YYYY-MM-DDTHH:MM:SS" or "YYYY-MM-DDTHH:MM:SSZ"). Assumed UTC if no timezone.
        duration_minutes (int): The duration of the appointment in minutes (default 30).
        location (str): The location of the appointment (default "Main Clinic").
        notes (str | None): Optional notes for the appointment.
    """
    logger.info(f"Tool 'book_appointment' called: patient={patient_id}, doctor={doctor_id}, start='{starts_at}'")

    start_dt = parse_datetime_str(starts_at)
    if not start_dt:
         return "Invalid start time format provided. Please use YYYY-MM-DDTHH:MM:SS format (timezone optional, UTC assumed)."

    # Calculate end time based on duration
    end_dt = start_dt + timedelta(minutes=duration_minutes)

    # Hardcode patient_id for now if not easily available from context
    # TODO: Ideally, patient_id should be retrieved from the authenticated user session
    # For now, let's assume patient_id 1 if it's not passed correctly by the LLM
    if not isinstance(patient_id, int) or patient_id <= 0:
        logger.warning(f"Invalid or missing patient_id ({patient_id}) in book_appointment call. Defaulting to 1.")
        patient_id = 1 # Replace with actual user ID retrieval later

    booking_result = None
    try:
        async for db in get_db_session(str(settings.database_url)):
            booking_result = await create_appointment(
                db=db,
                patient_id=patient_id,
                doctor_id=doctor_id,
                starts_at=start_dt,
                ends_at=end_dt,
                location=location,
                notes=notes,
            )
    except Exception as e:
        logger.error(f"Failed to get DB session or create appointment: {e}", exc_info=True)
        return "Sorry, I encountered an error trying to book the appointment. Please try again later."

    if booking_result:
        status = booking_result.get("status")
        if status == "confirmed":
            return (f"OK. Your appointment (ID: {booking_result['id']}) with Dr. Smith (ID: {doctor_id}) "
                    f"on {start_dt.strftime('%Y-%m-%d')} at {start_dt.strftime('%H:%M')} UTC has been confirmed. "
                    f"Location: {booking_result['location']}.")
        elif status == "conflict":
            return booking_result.get("message", "Sorry, that time slot is no longer available.")
        else: # error or other status
            return booking_result.get("message", "An unexpected issue occurred while booking.")
    else:
        return "Sorry, I couldn't book the appointment due to an unexpected error."


@tool("cancel_appointment", return_direct=False)
async def cancel_appointment(appointment_id: int, patient_id: int) -> str:
    """
    Cancels an existing appointment by its ID, verifying the patient ID.

    Args:
        appointment_id (int): The unique ID of the appointment to cancel.
        patient_id (int): The ID of the patient requesting the cancellation (for verification).
    """
    logger.info(f"Tool 'cancel_appointment' called for appointment {appointment_id} by patient {patient_id}")

    # Hardcode patient_id for now if not easily available from context
    # TODO: Retrieve patient_id from authenticated session
    if not isinstance(patient_id, int) or patient_id <= 0:
        logger.warning(f"Invalid or missing patient_id ({patient_id}) in cancel_appointment call. Defaulting to 1.")
        patient_id = 1 # Replace with actual user ID retrieval later

    cancellation_result = None
    try:
        async for db in get_db_session(str(settings.database_url)):
            cancellation_result = await delete_appointment(
                db=db,
                appointment_id=appointment_id,
                patient_id=patient_id
            )
    except Exception as e:
        logger.error(f"Failed to get DB session or cancel appointment: {e}", exc_info=True)
        return "Sorry, I encountered an error trying to cancel the appointment. Please try again later."

    if cancellation_result:
        return cancellation_result.get("message", "Could not process cancellation request.")
    else:
        return "Sorry, I couldn't cancel the appointment due to an unexpected error."

