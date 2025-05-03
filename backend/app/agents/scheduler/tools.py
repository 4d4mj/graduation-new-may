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
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState

# Import the CRUD functions and session utility
from app.db.crud.appointment import (
    get_available_slots_for_day,
    create_appointment,
    delete_appointment,
)
# Import NEW doctor CRUD function
from app.db.crud.doctor import (
    get_doctor_details_by_user_id,
    get_doctor_by_name,
    list_all_doctors,
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
    """
    Parses various datetime formats and standardizes to UTC.
    Supports ISO format (YYYY-MM-DDTHH:MM:SS) with or without timezone.
    Also attempts to handle common non-ISO formats.
    """
    if not datetime_str:
        logger.error("Empty datetime string provided")
        return None

    # Strip any extra whitespace that might cause parsing errors
    datetime_str = datetime_str.strip()

    try:
        # First attempt: Standard ISO format with fromisoformat
        try:
            dt = datetime.fromisoformat(datetime_str)
        except ValueError:
            # Second attempt: Try to handle common non-ISO formats
            formats_to_try = [
                "%Y-%m-%d %H:%M:%S",  # 2025-05-03 14:30:00
                "%Y-%m-%d %H:%M",      # 2025-05-03 14:30
                "%Y/%m/%d %H:%M:%S",   # 2025/05/03 14:30:00
                "%Y/%m/%d %H:%M",      # 2025/05/03 14:30
                "%d-%m-%Y %H:%M:%S",   # 03-05-2025 14:30:00
                "%d-%m-%Y %H:%M",      # 03-05-2025 14:30
                "%d/%m/%Y %H:%M:%S",   # 03/05/2025 14:30:00
                "%d/%m/%Y %H:%M",      # 03/05/2025 14:30
            ]

            for fmt in formats_to_try:
                try:
                    dt = datetime.strptime(datetime_str, fmt)
                    logger.info(f"Parsed datetime using alternative format: {fmt}")
                    break
                except ValueError:
                    continue
            else:  # If no format worked
                logger.error(f"Failed to parse datetime with any supported format: {datetime_str}")
                return None

        # Ensure timezone is set (use UTC if none provided)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        # Convert to UTC for storage
        return dt.astimezone(timezone.utc)

    except Exception as e:
        logger.error(f"Unexpected error parsing datetime '{datetime_str}': {str(e)}")
        return None

# ------------------------------------------------------------------ tools
@tool("list_free_slots", return_direct=False)
async def list_free_slots(doctor_name: Optional[str] = None, day: str | None = None) -> str:
    """
    Return a *human-readable* list of 30-minute free slots for a specific doctor on a given ISO date (YYYY-MM-DD).

    Args:
        doctor_name (Optional[str]): The name of the doctor. If None, will use a default doctor.
        day (str | None): Date in YYYY-MM-DD format. Defaults to tomorrow if not provided or invalid.
    """
    target_day = parse_iso_date(day)
    logger.info(f"Tool 'list_free_slots' called for doctor '{doctor_name}' on {target_day.isoformat()}")

    # Default doctor ID to use if no name provided or doctor not found
    default_doctor_id = 4

    slots = []
    doctor_name_display = f"Doctor" # Default name if lookup fails
    db_session: AsyncSession | None = None
    doctor_id = None

    try:
        async for db in get_db_session(str(settings.database_url)):
            db_session = db
            logger.info(f"Database session obtained for list_free_slots")

            # --- Find doctor by name if provided ---
            if doctor_name:
                doctor = await get_doctor_by_name(db, doctor_name)
                if doctor:
                    doctor_id = doctor.user_id
                    doctor_name_display = f"Dr. {doctor.first_name} {doctor.last_name}"
                    logger.info(f"Found doctor: {doctor_name_display} (ID: {doctor_id})")
                else:
                    logger.warning(f"Doctor with name '{doctor_name}' not found.")
            else:
               return "Sorry, I don't have the doctor id."

            # At this point, we should have a valid doctor_id
            if doctor_id:
                logger.info(f"Checking schedule for {doctor_name_display} (ID: {doctor_id})")
                slots = await get_available_slots_for_day(db, doctor_id, target_day)
                logger.info(f"get_available_slots_for_day returned {len(slots)} slots for {doctor_name_display}.")
            else:
                return "Sorry, I couldn't determine which doctor's schedule to check."

            break # Exit after first successful session use

        if db_session is None:
             logger.error("Failed to obtain a database session for list_free_slots.")
             return "Sorry, I couldn't connect to the scheduling database at the moment."

    except Exception as e:
        logger.error(f"Error during list_free_slots execution for doctor '{doctor_name}': {e}", exc_info=True)
        return f"Sorry, I encountered an error trying to check the schedule. Please try again later."
    finally:
        if db_session:
            try: await db_session.close()
            except Exception: pass # Ignore errors during close

    if not slots:
        # Doctor existence was already checked, so just report no slots
        return f"There are no available slots found for {doctor_name_display} on {target_day.isoformat()}."
    else:
        slot_list = "\n • ".join(slots)
        return f"Here are the available slots for {doctor_name_display} on {target_day.isoformat()}:\n • {slot_list}"


@tool("book_appointment", return_direct=False)
async def book_appointment(
    doctor_name: str,
    starts_at: str,
    patient_id: Annotated[int, InjectedState("user_id")],
    duration_minutes: int = 30,
    location: str = "Main Clinic",
    notes: str | None = None,
) -> str:
    """
    Books a new appointment for the current user with a doctor at a given start time and duration.
    Returns a confirmation or error message.

    Args:
        doctor_name (str): The name of the doctor to book with.
        starts_at (str): The desired start time in ISO format (e.g., "YYYY-MM-DDTHH:MM:SS" or "YYYY-MM-DDTHH:MM:SSZ"). Assumed UTC if no timezone.
        patient_id (str): The ID of the patient to book for.
        duration_minutes (int): The duration of the appointment in minutes (default 30).
        location (str): The location of the appointment (default "Main Clinic").
        notes (str | None): Optional notes for the appointment.
    """
    logger.info(f"Tool 'book_appointment' called: doctor_name='{doctor_name}', start='{starts_at}'")

    if not patient_id:
        logger.error("Failed to extract patient_id from session token")
        return "I couldn't identify you from your session. Please log in again."

    # --- Basic Input Validation ---
    if not doctor_name or not isinstance(doctor_name, str):
        logger.error(f"book_appointment called with invalid doctor_name: {doctor_name}")
        return "I need the doctor's name to book the appointment."

    start_dt = parse_datetime_str(starts_at)
    if not start_dt:
         return "Invalid start time format provided. Please use YYYY-MM-DDTHH:MM:SS format (timezone optional, UTC assumed)."
    # -----------------------------

    end_dt = start_dt + timedelta(minutes=duration_minutes)
    booking_result = None
    doctor_display_name = f"Doctor {doctor_name}" # Default name
    db_session: AsyncSession | None = None
    doctor_id = None  # Will be set if we find the doctor

    try:
        async for db in get_db_session(str(settings.database_url)):
            db_session = db
            logger.info(f"Database session obtained for book_appointment (Patient: {patient_id}, Doctor name: {doctor_name}).")

            # --- Find doctor by name ---
            doctor = await get_doctor_by_name(db, doctor_name)
            if not doctor:
                logger.warning(f"No doctor found with name '{doctor_name}'")
                return f"Sorry, I couldn't find a doctor named '{doctor_name}'. Please check the name and try again."

            doctor_id = doctor.user_id
            doctor_display_name = f"Dr. {doctor.first_name} {doctor.last_name}"
            logger.info(f"Found doctor: {doctor_display_name} (ID: {doctor_id})")
            # -------------------------

            booking_result = await create_appointment(
                db=db, patient_id=patient_id, doctor_id=doctor_id,
                starts_at=start_dt, ends_at=end_dt, location=location, notes=notes,
            )
            logger.info(f"create_appointment returned: {booking_result}")
            break # Exit after first successful session use

        if db_session is None:
             logger.error("Failed to obtain a database session for book_appointment.")
             return "Sorry, I couldn't connect to the scheduling database at the moment."

    except Exception as e:
        logger.error(f"Failed to book appointment for patient {patient_id} with doctor '{doctor_name}': {e}", exc_info=True)
        return "Sorry, I encountered an error trying to book the appointment. Please try again later."
    finally:
        if db_session:
            try: await db_session.close()
            except Exception: pass

    if booking_result:
        status = booking_result.get("status")
        if status == "confirmed":
            return (f"OK. Your appointment (ID: {booking_result['id']}) with {doctor_display_name} "
                    f"on {start_dt.strftime('%Y-%m-%d')} at {start_dt.strftime('%H:%M')} UTC has been confirmed. "
                    f"Location: {booking_result['location']}.")
        elif status == "conflict":
            return booking_result.get("message", "Sorry, that time slot is no longer available.")
        else: # error or other status
            return booking_result.get("message", "An unexpected issue occurred while booking.")
    else:
        # This case might happen if db session failed before booking_result was assigned
        return "Sorry, I couldn't book the appointment due to an unexpected error."


@tool("cancel_appointment", return_direct=False)
async def cancel_appointment(
    appointment_id: int,
    patient_id: Annotated[int, InjectedState("user_id")],
) -> str:
    """
    Cancels an existing appointment by its ID.

    Args:
        appointment_id (int): The unique ID of the appointment to cancel.
        patient_id (str): The ID of the patient to cancel the appointment for.
    """
    logger.info(f"Tool 'cancel_appointment' called for appointment {appointment_id}")


    if not patient_id:
        logger.error("Failed to extract patient_id from session token")
        return "I couldn't identify you from your session. Please log in again."

    # --- Basic Input Validation ---
    if not isinstance(appointment_id, int) or appointment_id <= 0:
        logger.error(f"cancel_appointment called with invalid appointment_id: {appointment_id}")
        return "I need the appointment ID to cancel it."
    # -----------------------------

    cancellation_result = None
    db_session: AsyncSession | None = None
    try:
        async for db in get_db_session(str(settings.database_url)):
             db_session = db
             logger.info(f"Database session obtained for cancel_appointment (Appt ID: {appointment_id}, Patient ID: {patient_id}).")
             cancellation_result = await delete_appointment(
                 db=db, appointment_id=appointment_id, patient_id=patient_id
             )
             logger.info(f"delete_appointment returned: {cancellation_result}")
             break # Exit after first successful session use
        if db_session is None:
             logger.error("Failed to obtain a database session for cancel_appointment.")
             return "Sorry, I couldn't connect to the scheduling database at the moment."

    except Exception as e:
        logger.error(f"Failed to cancel appointment {appointment_id} for patient {patient_id}: {e}", exc_info=True)
        return "Sorry, I encountered an error trying to cancel the appointment. Please try again later."
    finally:
        if db_session:
            try: await db_session.close()
            except Exception: pass

    if cancellation_result:
        return cancellation_result.get("message", "Could not process cancellation request.")
    else:
        # This case might happen if db session failed
        return "Sorry, I couldn't cancel the appointment due to an unexpected error."

