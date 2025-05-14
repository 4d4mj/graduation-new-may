"""
Database‑powered appointment scheduler + knowledge‑retrieval tools
──────────────────────────────────────────────────────────────────
Everything here is exposed to the LLM as a *LangChain tool*.

✔ list_doctors       – find doctors by name or specialty
✔ list_free_slots    – see open half‑hour slots for a doctor
✔ book_appointment   – create a new appointment
✔ cancel_appointment – cancel an existing appointment
✔ propose_booking    – create a booking proposal
"""
from __future__ import annotations

import logging
from datetime import datetime, date, timedelta, timezone
from babel.dates import format_date, format_datetime # type: ignore
from zoneinfo import ZoneInfo
import dateparser  # type: ignore
from typing import Optional, Dict, Any, Union, List

from langchain_core.tools import tool
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState  # type: ignore

from app.db.crud.appointment import (
    get_available_slots_for_day,
    create_appointment,
    delete_appointment,
    get_appointment
)
from app.db.crud.doctor import find_doctors
from app.db.session import tool_db_session


logger = logging.getLogger(__name__)

# helper to parse a day string into a date in user timezone or default to tomorrow
def _parse_day(text: str | None, user_tz: str | None) -> date:
    base = datetime.now(ZoneInfo(user_tz)) if user_tz else datetime.utcnow()
    if not text or not user_tz:
        return (base + timedelta(days=1)).date()
    parsed = dateparser.parse(
        text,
        settings={
            'TIMEZONE': user_tz,
            'RETURN_AS_TIMEZONE_AWARE': True,
            'RELATIVE_BASE': base,
            'PREFER_DATES_FROM': 'future',
        },
    )
    return parsed.date() if parsed else (base + timedelta(days=1)).date()


@tool("list_doctors")
async def list_doctors(
    name: str | None = None,
    specialty: str | None = None,
    limit: int = 5
) -> dict:
    """
    Find doctors by name or specialty.

    Parameters
    ----------
    name       : str  – Doctor's name (or part of it) to search for.
    specialty  : str  – Medical specialty to filter doctors.
    limit      : int  – Maximum number of doctors to return (default: 5).
    """
    logger.info(f"Tool 'list_doctors' called with name='{name}' specialty='{specialty}'")

    try:
        async with tool_db_session() as db:
            # Use the unified find_doctors function
            doctors = await find_doctors(
                db,
                name=name,
                specialty=specialty,
                limit=limit,
                return_single=False
            )

        if not doctors:
            return {
                "type": "no_doctors",
                "message": "No doctors found matching your criteria."
            }

        # Return found doctors with their IDs
        doctor_list = []
        for doc in doctors:
            doctor_list.append({
                "id": doc.user_id,
                "name": f"Dr. {doc.first_name} {doc.last_name}",
                "specialty": doc.specialty
            })

        return {
            "type": "doctors",
            "doctors": doctor_list,
            "message": f"Found {len(doctor_list)} doctors matching your criteria."
        }
    except Exception as e:
        logger.error(f"Error executing list_doctors tool: {e}", exc_info=True)
        error_msg = "I encountered an error while trying to find doctors. Please try again later."
        logger.info(f"Tool list_doctors returning error: '{error_msg}'")
        return {"type": "error", "message": error_msg}


@tool("list_free_slots")
async def list_free_slots(
    doctor_id: int = None,
    doctor_name: str = None,
    day: str | None = None,
    user_tz: Annotated[str | None, InjectedState("user_tz")] = None
)-> dict:
    """
    Human readable list of 30‑minute free slots for a doctor on a given day.

    Parameters
    ----------
    doctor_id   : int  – Doctor's ID to check (preferred if available).
    doctor_name : str  – Doctor's name to check (used if doctor_id not provided).
    day         : str  – ISO date (YYYY‑MM‑DD) or natural language date. Tomorrow by default.
    """
    try:
        # determine target day
        target_day = _parse_day(day, user_tz)

        # Make sure doctor_id is an integer if provided
        if doctor_id is not None:
            try:
                doctor_id = int(doctor_id)
            except (ValueError, TypeError):
                logger.warning(f"Invalid doctor_id format: {doctor_id}, attempting to treat as name")
                doctor_name = str(doctor_id)
                doctor_id = None

        logger.info(f"Tool 'list_free_slots' called for doctor_id={doctor_id}, doctor_name={doctor_name} on {target_day}")

        if not doctor_id and not doctor_name:
            return {
                "type": "error",
                "message": "Please provide either a doctor ID or a doctor name."
            }

        async with tool_db_session() as db:
            # Find the doctor by ID or name
            doctor = None
            if doctor_id:
                doctor = await find_doctors(db, doctor_id=doctor_id, return_single=True)
            elif doctor_name:
                # Clean up the doctor_name - strip "Dr." prefix if present
                cleaned_name = doctor_name
                if cleaned_name.lower().startswith("dr."):
                    cleaned_name = cleaned_name[3:].strip()
                elif cleaned_name.lower().startswith("dr "):
                    cleaned_name = cleaned_name[3:].strip()

                doctor = await find_doctors(db, name=cleaned_name, return_single=True)

            if not doctor:
                id_or_name = doctor_id if doctor_id else f"'{doctor_name}'"
                logger.warning(f"Doctor {id_or_name} not found.")
                return {"type": "error", "message": f"Doctor {id_or_name} not found."}

            # Now get available slots using doctor's ID
            logger.debug(f"Found doctor {doctor.user_id}: {doctor.first_name} {doctor.last_name}. Checking slots...")
            slots = await get_available_slots_for_day(db, doctor.user_id, target_day)
            logger.debug(f"Found slots: {slots}")

        if not slots:
            return {
                "type": "no_slots",
                "message": f"Dr. {doctor.first_name} {doctor.last_name} has no available slots on {format_date(target_day, 'long', locale='en')}. Please try another day."
            }

        # Return enhanced response with doctor's full name and ID
        return {
            "type": "slots",
            "doctor_id": doctor.user_id,
            "doctor": f"Dr. {doctor.first_name} {doctor.last_name}",
            "specialty": doctor.specialty,
            "agent": "Scheduler",
            "reply_template": "I choose the appointment slot at ",
            "date": format_date(target_day, "long", locale="en"),
            "options": slots,
        }
    except Exception as e:
        # Log the error, but maintain the expected JSON structure with "type": "error"
        logger.error(f"Error executing list_free_slots tool: {e}", exc_info=True)
        error_msg = "I encountered an error while trying to check the schedule. Please try again later."
        logger.info(f"Tool list_free_slots returning error with proper schema: '{error_msg}'")
        # Return error in the expected schema format for UI
        return {"type": "error", "message": error_msg}


@tool("book_appointment")
async def book_appointment(
    doctor_id: int = None,
    doctor_name: str = None,
    starts_at: str = None,
    patient_id: Annotated[int, InjectedState("user_id")] = None,
    user_tz: Annotated[str | None, InjectedState("user_tz")] = None,
    duration_minutes: int = 30,
    location: str = "Main Clinic",
    notes: str | None = None,
) -> dict:
    """
    Create an appointment and return the DB confirmation / error text.

    Parameters
    ----------
    doctor_id        : int  – Doctor's ID (preferred if available).
    doctor_name      : str  – Doctor's name (used if doctor_id not provided).
    starts_at        : str  – Start time of the appointment (ISO format or natural language).
    duration_minutes : int  – Duration of the appointment in minutes (default: 30).
    location         : str  – Location of the appointment (default: "Main Clinic").
    notes            : str  – Additional notes for the appointment.
    """
    # Make sure doctor_id is an integer if provided
    if doctor_id is not None:
        try:
            doctor_id = int(doctor_id)
        except (ValueError, TypeError):
            logger.warning(f"Invalid doctor_id format: {doctor_id}, attempting to treat as name")
            doctor_name = str(doctor_id)
            doctor_id = None

    logger.info(f"Tool 'book_appointment' called by user {patient_id} for doctor_id={doctor_id}, doctor_name={doctor_name} at {starts_at}")

    if not doctor_id and not doctor_name:
        return {"status": "error", "message": "Please provide either a doctor ID or a doctor name."}

    if not starts_at:
        return {"status": "error", "message": "Please provide a start time for the appointment."}

    if not patient_id:
        logger.warning("Booking tool called without patient_id.")
        return {"status": "error", "message": "I couldn't identify you – please log in again."}

    # Parse start time: try natural language then ISO fallback
    start_dt = None
    if user_tz:
        local_dt = dateparser.parse(starts_at, settings={
            'TIMEZONE': user_tz,
            'RETURN_AS_TIMEZONE_AWARE': True,
            'PREFER_DATES_FROM': 'future'
        })
        if local_dt:
            # Convert to UTC
            start_dt = local_dt.astimezone(timezone.utc)
    # ISO format fallback
    if not start_dt:
        try:
            iso_dt = datetime.fromisoformat(starts_at)
            start_dt = iso_dt if iso_dt.tzinfo else iso_dt.replace(tzinfo=timezone.utc)
        except Exception:
            start_dt = None
    if not start_dt:
        logger.warning(f"Invalid starts_at format received: {starts_at}")
        return {
            "status": "error",
            "message": "Please provide the start time either as an ISO string `YYYY-MM-DDTHH:MM:SS[Z]` or in natural language (e.g., 'next Monday at 2 pm')."
        }

    try:
        async with tool_db_session() as db:
            # Get doctor by ID or name
            doctor = None
            if doctor_id:
                doctor = await find_doctors(db, doctor_id=doctor_id, return_single=True)
            elif doctor_name:
                # Clean up the doctor_name - strip "Dr." prefix if present
                cleaned_name = doctor_name
                if cleaned_name.lower().startswith("dr."):
                    cleaned_name = cleaned_name[3:].strip()
                elif cleaned_name.lower().startswith("dr "):
                    cleaned_name = cleaned_name[3:].strip()

                doctor = await find_doctors(db, name=cleaned_name, return_single=True)

            if not doctor:
                id_or_name = doctor_id if doctor_id else f"'{doctor_name}'"
                logger.warning(f"Doctor {id_or_name} not found during booking.")
                return {"status": "error", "message": f"Doctor {id_or_name} not found."}

            # Calculate end time
            end_dt = start_dt + timedelta(minutes=duration_minutes)

            logger.debug(f"Attempting to create appointment entry in DB for user {patient_id} with doctor {doctor.user_id}")
            appointment = await create_appointment(
                db,
                patient_id=patient_id,
                doctor_id=doctor.user_id,
                starts_at=start_dt,
                ends_at=end_dt,
                location=location,
                notes=notes,
            )

            # Check if result is a dict (error case)
            if isinstance(appointment, dict):
                logger.warning(f"Booking failed for doctor {doctor.user_id} at {starts_at}. Reason: {appointment.get('message', 'Unknown')}")
                return appointment

            # If successful, appointment is an AppointmentModel instance
            logger.info(f"Booking successful: ID {appointment.id}")
            doctor_email_address = "unknown" # Default
            
             # Ensure doctor and doctor.user and doctor.user.email are accessible
            if doctor and doctor.user and hasattr(doctor.user, 'email'): # Add checks
                doctor_email_address = doctor.user.email
            else:
                logger.warning(f"Could not retrieve email for doctor ID {doctor.user_id if doctor else 'N/A'}")

            return {
                "status": "confirmed",
                "id": appointment.id,
                "doctor_id": doctor.user_id,
                "doctor_name": f"Dr. {doctor.first_name} {doctor.last_name}",
                "doctor_email": doctor_email_address,
                "start_dt": format_datetime(start_dt, 'long', locale='en')
            }

    except Exception as e:
        logger.error(f"Error executing book_appointment tool: {e}", exc_info=True)
        return {"status": "error", "message": "I encountered an error while trying to book the appointment. Please try again later."}


@tool("cancel_appointment")
async def cancel_appointment(
    appointment_id: int,
    patient_id: Annotated[int, InjectedState("user_id")],
) -> dict:
    """Cancel an existing appointment owned by the current user."""
    logger.info(f"Tool 'cancel_appointment' called by user {patient_id} for appointment ID {appointment_id}")
    if not patient_id:
        logger.warning("Cancel tool called without patient_id.")
        return {"status": "error", "message": "I couldn't identify you – please log in again."}

    try:
        async with tool_db_session() as db:
            # First verify the appointment exists and belongs to this patient
            try:
                appointment = await get_appointment(db, appointment_id, patient_id, "patient")
            except Exception as e:
                logger.warning(f"Failed to get appointment {appointment_id} for patient {patient_id}: {e}")
                return {"status": "error", "message": "That appointment doesn't exist or doesn't belong to you."}

            # Now delete it
            logger.debug(f"Attempting to delete appointment {appointment_id} for user {patient_id}")
            result = await delete_appointment(db, appointment_id, patient_id, "patient")

            if result:
                logger.info(f"Successfully cancelled appointment {appointment_id}")
                return {"status": "cancelled", "message": f"Appointment #{appointment_id} has been cancelled successfully."}
            else:
                logger.warning(f"Failed to cancel appointment {appointment_id}")
                return {"status": "error", "message": "Sorry – I couldn't cancel that appointment."}

    except Exception as e:
        logger.error(f"Error executing cancel_appointment tool: {e}", exc_info=True)
        return {"status": "error", "message": "I encountered an error while trying to cancel the appointment. Please try again later."}


@tool("propose_booking")
async def propose_booking(
    doctor_id: int = None,
    doctor_name: str = None,
    starts_at: str = None,
    notes: str | None = None,
) -> dict:
    """
    Return a booking proposal without touching the DB.

    Parameters
    ----------
    doctor_id   : int  – Doctor's ID (preferred if available).
    doctor_name : str  – Doctor's name (used if doctor_id not provided).
    starts_at   : str  – Proposed start time of the appointment.
    notes       : str  – Additional notes for the appointment.
    """
    # Make sure doctor_id is an integer if provided
    if doctor_id is not None:
        try:
            doctor_id = int(doctor_id)
        except (ValueError, TypeError):
            logger.warning(f"Invalid doctor_id format: {doctor_id}, attempting to treat as name")
            doctor_name = str(doctor_id)
            doctor_id = None

    if not doctor_id and not doctor_name:
        return {"type": "error", "message": "Please provide either a doctor ID or a doctor name."}

    if not starts_at:
        return {"type": "error", "message": "Please provide a start time for the appointment."}

    logger.info(f"Tool 'propose_booking' called for doctor_id={doctor_id}, doctor_name={doctor_name} at {starts_at}")

    try:
        # Try to get doctor information to enhance the proposal
        async with tool_db_session() as db:
            doctor = None
            if doctor_id:
                doctor = await find_doctors(db, doctor_id=doctor_id, return_single=True)
            elif doctor_name:
                # Clean up the doctor_name - strip "Dr." prefix if present
                cleaned_name = doctor_name
                if cleaned_name.lower().startswith("dr."):
                    cleaned_name = cleaned_name[3:].strip()
                elif cleaned_name.lower().startswith("dr "):
                    cleaned_name = cleaned_name[3:].strip()

                doctor = await find_doctors(db, name=cleaned_name, return_single=True)

        if doctor:
            full_name = f"Dr. {doctor.first_name} {doctor.last_name}"
            return {
                "type": "confirm_booking",
                "doctor_id": doctor.user_id,
                "doctor": full_name,
                "specialty": doctor.specialty,
                "starts_at": starts_at,
                "notes": notes,
            }
    except Exception as e:
        logger.warning(f"Error enriching booking proposal: {e}")

    # Fallback to basic proposal if doctor lookup fails
    doctor_info = doctor_id if doctor_id else doctor_name
    return {
        "type": "confirm_booking",
        "doctor": doctor_info,
        "starts_at": starts_at,
        "notes": notes,
    }
