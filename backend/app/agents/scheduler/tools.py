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
# Import NEW doctor CRUD function
from app.db.crud.doctor import get_doctor_details_by_user_id
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
async def list_free_slots(doctor_id: int, day: str | None = None) -> str: # Removed default doctor_id=1
    """
    Return a *human-readable* list of 30-minute free slots for a specific doctor on a given ISO date (YYYY-MM-DD).

    Args:
        doctor_id (int): The user ID of the doctor.
        day (str | None): Date in YYYY-MM-DD format. Defaults to tomorrow if not provided or invalid.
    """
    target_day = parse_iso_date(day)
    logger.info(f"Tool 'list_free_slots' called for doctor_id {doctor_id} on {target_day.isoformat()}")

    slots = []
    doctor_name = f"Doctor (ID: {doctor_id})" # Default name if lookup fails
    db_session: AsyncSession | None = None

    try:
        async for db in get_db_session(str(settings.database_url)):
            db_session = db
            logger.info(f"Database session obtained for list_free_slots (Doctor ID: {doctor_id}).")

            # --- Fetch doctor details ---
            doctor_details = await get_doctor_details_by_user_id(db, doctor_id)
            if not doctor_details:
                logger.warning(f"Doctor with user_id {doctor_id} not found or is not a doctor.")
                return f"Sorry, I couldn't find a doctor with the ID {doctor_id}."
            doctor_name = f"Dr. {doctor_details.first_name} {doctor_details.last_name}"
            # -----------------------------

            logger.info(f"Checking schedule for {doctor_name} (ID: {doctor_id})")
            slots = await get_available_slots_for_day(db, doctor_id, target_day)
            logger.info(f"get_available_slots_for_day returned {len(slots)} slots for {doctor_name}.")
            break # Exit after first successful session use
        if db_session is None:
             logger.error("Failed to obtain a database session for list_free_slots.")
             return "Sorry, I couldn't connect to the scheduling database at the moment."

    except Exception as e:
        logger.error(f"Error during list_free_slots execution for doctor {doctor_id}: {e}", exc_info=True)
        return f"Sorry, I encountered an error trying to check the schedule for {doctor_name}. Please try again later."
    finally:
        if db_session:
            try: await db_session.close()
            except Exception: pass # Ignore errors during close

    if not slots:
        # Doctor existence was already checked, so just report no slots
        return f"There are no available slots found for {doctor_name} on {target_day.isoformat()}."
    else:
        slot_list = "\n • ".join(slots)
        return f"Here are the available slots for {doctor_name} on {target_day.isoformat()}:\n • {slot_list}"


@tool("book_appointment", return_direct=False)
async def book_appointment(
    patient_id: int,
    doctor_id: int,
    starts_at: str,
    duration_minutes: int = 30,
    location: str = "Main Clinic",
    notes: str | None = None,
) -> str:
    """
    Books a new appointment for a specified patient with a doctor at a given start time and duration.
    Returns a confirmation or error message. Requires patient_id and doctor_id (user IDs).

    Args:
        patient_id (int): The user ID of the patient making the booking. Must be provided.
        doctor_id (int): The user ID of the doctor. Must be provided.
        starts_at (str): The desired start time in ISO format (e.g., "YYYY-MM-DDTHH:MM:SS" or "YYYY-MM-DDTHH:MM:SSZ"). Assumed UTC if no timezone.
        duration_minutes (int): The duration of the appointment in minutes (default 30).
        location (str): The location of the appointment (default "Main Clinic").
        notes (str | None): Optional notes for the appointment.
    """
    logger.info(f"Tool 'book_appointment' called: patient_id={patient_id}, doctor_id={doctor_id}, start='{starts_at}'")

    # --- Basic Input Validation ---
    if not isinstance(patient_id, int) or patient_id <= 0:
        logger.error(f"book_appointment called with invalid patient_id: {patient_id}")
        return "I need the patient's user ID to book the appointment."
    if not isinstance(doctor_id, int) or doctor_id <= 0:
        logger.error(f"book_appointment called with invalid doctor_id: {doctor_id}")
        return "I need the doctor's user ID to book the appointment."

    start_dt = parse_datetime_str(starts_at)
    if not start_dt:
         return "Invalid start time format provided. Please use YYYY-MM-DDTHH:MM:SS format (timezone optional, UTC assumed)."
    # -----------------------------

    end_dt = start_dt + timedelta(minutes=duration_minutes)
    booking_result = None
    doctor_name = f"Doctor (ID: {doctor_id})" # Default name
    db_session: AsyncSession | None = None

    try:
        async for db in get_db_session(str(settings.database_url)):
            db_session = db
            logger.info(f"Database session obtained for book_appointment (Patient: {patient_id}, Doctor: {doctor_id}).")

            # --- Fetch doctor details for confirmation message ---
            doctor_details = await get_doctor_details_by_user_id(db, doctor_id)
            if not doctor_details:
                logger.warning(f"Attempting to book with non-existent/non-doctor user_id: {doctor_id}")
                return f"Sorry, I couldn't find a doctor with the ID {doctor_id}."
            doctor_name = f"Dr. {doctor_details.first_name} {doctor_details.last_name}"
            # ----------------------------------------------------

            # --- Optional: Check if patient_id exists ---
            # patient_user = await db.get(UserModel, patient_id)
            # if not patient_user:
            #     logger.warning(f"Attempting to book for non-existent patient_id: {patient_id}")
            #     return f"Sorry, I couldn't find a patient with the ID {patient_id}."
            # ---------------------------------------------

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
        logger.error(f"Failed to book appointment for patient {patient_id} with doctor {doctor_id}: {e}", exc_info=True)
        return "Sorry, I encountered an error trying to book the appointment. Please try again later."
    finally:
        if db_session:
            try: await db_session.close()
            except Exception: pass

    if booking_result:
        status = booking_result.get("status")
        if status == "confirmed":
            return (f"OK. Your appointment (ID: {booking_result['id']}) with {doctor_name} " # Use fetched name
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
async def cancel_appointment(appointment_id: int, patient_id: int) -> str:
    """
    Cancels an existing appointment by its ID, verifying the patient ID. Requires patient_id.

    Args:
        appointment_id (int): The unique ID of the appointment to cancel.
        patient_id (int): The user ID of the patient requesting the cancellation (for verification). Must be provided.
    """
    logger.info(f"Tool 'cancel_appointment' called for appointment {appointment_id} by patient {patient_id}")

    # --- Basic Input Validation ---
    if not isinstance(patient_id, int) or patient_id <= 0:
        logger.error(f"cancel_appointment called with invalid patient_id: {patient_id}")
        return "I need the patient's user ID to cancel the appointment."
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

