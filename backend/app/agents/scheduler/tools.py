"""
Database‑powered appointment scheduler + knowledge‑retrieval tools
──────────────────────────────────────────────────────────────────
Everything here is exposed to the LLM as a *LangChain tool*.

✔ list_free_slots     – see open half‑hour slots for a doctor
✔ book_appointment    – create a new appointment
✔ cancel_appointment  – cancel an existing appointment

"""
from __future__ import annotations

import logging
from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo
import dateparser  # type: ignore


from langchain_core.tools import tool
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState  # type: ignore

from app.db.crud.appointment import (
    get_available_slots_for_day,
    create_appointment,
    delete_appointment,
)
from app.db.crud.doctor import get_doctor_by_name
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


@tool("list_free_slots", return_direct=True)
async def list_free_slots(
    doctor_name: str,
    day: str | None = None,
    user_tz: Annotated[str | None, InjectedState("user_tz")] = None
)-> dict:
    """
    Human readable list of 30‑minute free slots for a doctor on a given day.

    Parameters
    ----------
    doctor_name : str  – Which doctor's calendar to check.
    day         : str  – ISO date (YYYY‑MM‑DD).  Tomorrow by default.
    """
    # determine target day
    target_day = _parse_day(day, user_tz)

    logger.info(f"Tool 'list_free_slots' called for Dr. {doctor_name} on {target_day}")

    try:
        async with tool_db_session() as db:
            if not (doc := await get_doctor_by_name(db, doctor_name)):
                logger.warning(f"Doctor '{doctor_name}' not found in tool.")
                return {"type": "error", "message": f"No doctor named '{doctor_name}'."}

            logger.debug(f"Found doctor {doc.user_id}. Checking slots...")
            slots = await get_available_slots_for_day(db, doc.user_id, target_day)
            logger.debug(f"Found slots: {slots}")

        if not slots:
            return {
                "type": "slots",
                "doctor": doctor_name,
                "date": target_day.isoformat(),
                "options": [],
            }

        return {
            "type": "slots",
            "doctor": doctor_name,
            "date": target_day.isoformat(),
            "options": slots,
        }
    except Exception as e:
        logger.error(f"Error executing list_free_slots tool: {e}", exc_info=True)
        error_msg = "I encountered an error while trying to check the schedule. Please try again later."
        logger.info(f"Tool list_free_slots returning error: '{error_msg}'")
        return error_msg


@tool("book_appointment", return_direct=True)
async def book_appointment(
    doctor_name: str,
    starts_at: str,
    patient_id: Annotated[int, InjectedState("user_id")],
    user_tz: Annotated[str | None, InjectedState("user_tz")],
    duration_minutes: int = 30,
    location: str = "Main Clinic",
    notes: str | None = None,
) -> str:
    """Create an appointment and return the DB confirmation / error text."""
    logger.info(f"Tool 'book_appointment' called by user {patient_id} for Dr. {doctor_name} at {starts_at}")
    if not patient_id:
        logger.warning("Booking tool called without patient_id.")
        return "I couldn't identify you – please log in again."

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
        return (
            "Please provide the start time either as an ISO string `YYYY-MM-DDTHH:MM:SS[Z]`"
            " or in natural language (e.g., 'next Monday at 2 pm')."
        )

    try:
        async with tool_db_session() as db:
            doc = await get_doctor_by_name(db, doctor_name)
            if not doc:
                logger.warning(f"Doctor '{doctor_name}' not found during booking.")
                return f"No doctor named '{doctor_name}'."

            logger.debug(f"Attempting to create appointment entry in DB for user {patient_id} with doctor {doc.user_id}")
            result = await create_appointment(
                db,
                patient_id=patient_id,
                doctor_id=doc.user_id,
                starts_at=start_dt,
                ends_at=start_dt + timedelta(minutes=duration_minutes),
                location=location,
                notes=notes,
            )

        if not result or result.get("status") != "confirmed":
            logger.warning(f"Booking failed for Dr. {doctor_name} at {starts_at}. Reason: {result.get('message', 'Unknown') if result else 'None'}")
            return result.get("message", "Could not book – please try another slot.") if result else "Booking failed for an unknown reason."

        logger.info(f"Booking successful: ID {result.get('id')}")
        return (
            f"Your appointment (ID {result['id']}) with Dr. {doctor_name} "
            f"on {start_dt:%Y‑%m‑%d at %H:%M %Z} is confirmed 🎉." # Use %Z for timezone if available
        )
    except Exception as e:
        logger.error(f"Error executing book_appointment tool: {e}", exc_info=True)
        return "I encountered an error while trying to book the appointment. Please try again later."


@tool("cancel_appointment", return_direct=True)
async def cancel_appointment(
    appointment_id: int,
    patient_id: Annotated[int, InjectedState("user_id")],
) -> str:
    """Cancel an existing appointment owned by the current user."""
    logger.info(f"Tool 'cancel_appointment' called by user {patient_id} for appointment ID {appointment_id}")
    if not patient_id:
        logger.warning("Cancel tool called without patient_id.")
        return "I couldn't identify you – please log in again."

    try:
        async with tool_db_session() as db:
            logger.debug(f"Attempting to delete appointment {appointment_id} for user {patient_id}")
            result = await delete_appointment(db, appointment_id, patient_id)

        log_func = logger.info if result.get("status") == "cancelled" else logger.warning
        log_func(f"Cancellation result for appt {appointment_id} / user {patient_id}: {result}")

        return result.get("message", "Sorry – I couldn't cancel that appointment.")
    except Exception as e:
        logger.error(f"Error executing cancel_appointment tool: {e}", exc_info=True)
        return "I encountered an error while trying to cancel the appointment. Please try again later."
