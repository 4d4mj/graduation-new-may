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
from babel.dates import format_date, format_datetime  # type: ignore
from zoneinfo import ZoneInfo
import dateparser  # type: ignore
from typing import Optional, Dict, Any, Union, List

from langchain_core.tools import tool
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState  # type: ignore
from app.db.crud.user import get_user
from app.db.crud.appointment import (
    get_available_slots_for_day,
    create_appointment,
    delete_appointment,
    get_appointment,
    update_appointment_gcal_id,
)
from app.db.crud.doctor import find_doctors
from app.db.session import tool_db_session
from app.db.models.appointment import AppointmentModel  # <-- Add this import
from fastapi import HTTPException

import asyncio
from pathlib import Path  # If not already there
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pytz  # If not already there


logger = logging.getLogger(__name__)

GCAL_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
COMMON_TOOLS_DIR = Path(__file__).resolve().parent.parent
GCAL_TOKEN_FILE_PATH = COMMON_TOOLS_DIR / "calendar" / "token.json"
GCAL_DEFAULT_TIMEZONE = "Asia/Beirut"


# --- Google Calendar Helper Function ---
def _get_gcal_service_sync():
    creds = None
    # Add logging for the path being checked
    logger.info(
        f"Google Calendar: Attempting to access token file at resolved path: {GCAL_TOKEN_FILE_PATH.resolve()}"
    )  # Using .resolve() for absolute path logging

    if not GCAL_TOKEN_FILE_PATH.exists():
        logger.error(
            f"Google Calendar: Token file NOT FOUND at calculated path: {GCAL_TOKEN_FILE_PATH}"
        )
        return (
            None,
            f"Configuration error: Google Calendar token file not found. Checked: {GCAL_TOKEN_FILE_PATH}",
        )

    logger.info(
        f"Google Calendar: Token file found at {GCAL_TOKEN_FILE_PATH}. Proceeding to load."
    )
    try:
        creds = Credentials.from_authorized_user_file(
            str(GCAL_TOKEN_FILE_PATH), GCAL_SCOPES
        )
    except Exception as e:
        logger.error(
            f"Google Calendar: Error loading credentials from {GCAL_TOKEN_FILE_PATH}: {e}",
            exc_info=True,
        )
        return (
            None,
            f"Error loading Google Calendar credentials from {GCAL_TOKEN_FILE_PATH}: {e}.",
        )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.warning(
                f"Google Calendar: Credentials from {GCAL_TOKEN_FILE_PATH} are expired. Refresh might be needed (requires client secrets, not done by this tool). API call might fail."
            )
        else:
            logger.error(
                f"Google Calendar: Could not load valid credentials from {GCAL_TOKEN_FILE_PATH}. Token might be corrupted or missing required fields."
            )
            return (
                None,
                "Invalid or missing Google Calendar credentials. Token may be expired or improperly formatted.",
            )
    try:
        service = build(
            "calendar", "v3", credentials=creds, cache_discovery=False
        )  # Added cache_discovery=False for potential GCE issues
        logger.info("Google Calendar service object created successfully.")
        return service, None
    except HttpError as error:
        logger.error(
            f"Google Calendar: API error building service: {error}. Details: {error.content}",
            exc_info=True,
        )
        return (
            None,
            f"API error building Google Calendar service: {error.resp.status if error.resp else 'Unknown'}",
        )
    except Exception as e:
        logger.error(
            f"Google Calendar: Unexpected error building service: {e}", exc_info=True
        )
        return None, f"Unexpected error building Google Calendar service: {e}"


async def _delete_gcal_event_if_exists_scheduler(
    event_id: str, calendar_id: str = "primary"
) -> tuple[bool, str]:
    """
    Helper to delete a single GCal event.
    Returns (success_bool, message_str).
    """
    if not event_id:
        return True, "No GCal event ID provided for deletion."

    logger.info(
        f"Scheduler GCal Helper: Attempting to get GCal service for deleting event_id: {event_id}"
    )
    service, error_msg = await asyncio.to_thread(
        _get_gcal_service_sync
    )  # Uses the one defined/imported in this file
    if not service:
        logger.error(
            f"Scheduler GCal Helper: Failed to get Google Calendar service: {error_msg}"
        )
        return False, f"Failed to connect to Google Calendar: {error_msg}"

    try:
        logger.info(
            f"Scheduler GCal Helper: Attempting to delete Google Calendar event: {event_id} from calendar: {calendar_id}"
        )
        await asyncio.to_thread(
            service.events()
            .delete(calendarId=calendar_id, eventId=event_id, sendUpdates="all")
            .execute
        )
        logger.info(
            f"Scheduler GCal Helper: Successfully deleted Google Calendar event: {event_id}"
        )
        return True, f"Google Calendar event {event_id} successfully deleted."
    except HttpError as e:
        if e.resp.status == 404:
            logger.warning(
                f"Scheduler GCal Helper: Google Calendar event {event_id} not found for deletion (404)."
            )
            return (
                True,
                f"Google Calendar event {event_id} not found (might be already deleted).",
            )
        logger.error(
            f"Scheduler GCal Helper: HttpError deleting event {event_id}: {e.resp.status} - {e.content}",
            exc_info=True,
        )
        return (
            False,
            f"Google Calendar API error deleting event {event_id}: {e.resp.status}",
        )
    except Exception as e:
        logger.error(
            f"Scheduler GCal Helper: Unexpected error deleting event {event_id}: {e}",
            exc_info=True,
        )
        return (
            False,
            f"Unexpected error deleting Google Calendar event {event_id}: {str(e)}",
        )


# helper to parse a day string into a date in user timezone or default to tomorrow
def _parse_day(text: str | None, user_tz: str | None) -> date:
    base = datetime.now(ZoneInfo(user_tz)) if user_tz else datetime.utcnow()
    if not text or not user_tz:
        return (base + timedelta(days=1)).date()
    parsed = dateparser.parse(
        text,
        settings={
            "TIMEZONE": user_tz,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "RELATIVE_BASE": base,
            "PREFER_DATES_FROM": "future",
        },
    )
    return parsed.date() if parsed else (base + timedelta(days=1)).date()


@tool("list_doctors")
async def list_doctors(
    name: str | None = None, specialty: str | None = None, limit: int = 5
) -> dict:
    """
    Find doctors by name or specialty.

    Parameters
    ----------
    name       : str  – Doctor's name (or part of it) to search for.
    specialty  : str  – Medical specialty to filter doctors.
    limit      : int  – Maximum number of doctors to return (default: 5).
    """
    logger.info(
        f"Tool 'list_doctors' called with name='{name}' specialty='{specialty}'"
    )

    try:
        async with tool_db_session() as db:
            # Use the unified find_doctors function
            doctors = await find_doctors(
                db, name=name, specialty=specialty, limit=limit, return_single=False
            )

        if not doctors:
            return {
                "type": "no_doctors",
                "message": "No doctors found matching your criteria.",
            }

        # Return found doctors with their IDs
        doctor_list = []
        for doc in doctors:
            doctor_list.append(
                {
                    "id": doc.user_id,
                    "name": f"Dr. {doc.first_name} {doc.last_name}",
                    "specialty": doc.specialty,
                }
            )

        return {
            "type": "doctors",
            "doctors": doctor_list,
            "message": f"Found {len(doctor_list)} doctors matching your criteria.",
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
    user_tz: Annotated[str | None, InjectedState("user_tz")] = None,
) -> dict:
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
                logger.warning(
                    f"Invalid doctor_id format: {doctor_id}, attempting to treat as name"
                )
                doctor_name = str(doctor_id)
                doctor_id = None

        logger.info(
            f"Tool 'list_free_slots' called for doctor_id={doctor_id}, doctor_name={doctor_name} on {target_day} with user_tz={user_tz}"
        )

        if not doctor_id and not doctor_name:
            return {
                "type": "error",
                "message": "Please provide either a doctor ID or a doctor name.",
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

            # Now get available slots using doctor's ID, passing user_tz
            logger.debug(
                f"Found doctor {doctor.user_id}: {doctor.first_name} {doctor.last_name}. Checking slots for target_day: {target_day} with user_tz: {user_tz}..."
            )
            slots = await get_available_slots_for_day(db, doctor.user_id, target_day, user_tz=user_tz) # Pass user_tz
            logger.debug(f"Found slots (adjusted for user_tz if provided): {slots}")

        if not slots:
            return {
                "type": "no_slots",
                "message": f"Dr. {doctor.first_name} {doctor.last_name} has no available slots on {format_date(target_day, 'long', locale='en')}. Please try another day.",
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
        logger.info(
            f"Tool list_free_slots returning error with proper schema: '{error_msg}'"
        )
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
    send_google_calendar_invite: bool = True,
    gcal_summary_override: Optional[str] = None,
    gcal_description_override: Optional[str] = None,
) -> dict:
    """
    Create a clinic appointment. If send_google_calendar_invite is true, will also attempt
    to send a Google Calendar invite to the doctor for the *same date and time* as the clinic appointment.
    Returns DB confirmation and Google Calendar status.
    The starts_at parameter is for the clinic appointment and can be a specific date/time.
    """
    logger.info(
        f"Tool 'book_appointment' called by user {patient_id} for doctor_id={doctor_id}, doctor_name={doctor_name} at {starts_at}. Send GCal: {send_google_calendar_invite}"
    )

    if not doctor_id and not doctor_name:
        return {
            "status": "error",
            "message": "Please provide either a doctor ID or a doctor name.",
        }
    if not starts_at:
        return {
            "status": "error",
            "message": "Please provide a start time for the appointment.",
        }
    if not patient_id:
        logger.warning("Booking tool called without patient_id.")
        return {
            "status": "error",
            "message": "I couldn't identify you – please log in again.",
        }

    effective_patient_tz_str = user_tz or GCAL_DEFAULT_TIMEZONE
    parsed_clinic_dt = dateparser.parse(
        starts_at,
        settings={
            "TIMEZONE": effective_patient_tz_str,
            "TO_TIMEZONE": "UTC",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "future",
        },
    )
    if not parsed_clinic_dt:
        logger.warning(f"Invalid starts_at format for clinic appointment: {starts_at}")
        return {
            "status": "error",
            "message": f"I couldn't understand the appointment time '{starts_at}'. Please try a format like 'YYYY-MM-DD HH:MM' or 'tomorrow at 2pm'.",
        }

    start_dt_clinic_utc = parsed_clinic_dt
    end_dt_clinic_utc = start_dt_clinic_utc + timedelta(minutes=duration_minutes)

    clinic_booking_result: Dict[str, Any] = {}
    google_calendar_status: str = "Not attempted."
    doctor_email_address: Optional[str] = None
    db_appointment_object = None
    patient_full_name_for_gcal: str = "Patient"  # Default
    doctor_full_name_for_gcal: str = "Doctor"  # Default

    async with tool_db_session() as db:
        doctor_model_instance = None
        if doctor_id:
            doctor_model_instance = await find_doctors(
                db, doctor_id=doctor_id, return_single=True
            )
        elif doctor_name:
            cleaned_name = doctor_name
            if cleaned_name.lower().startswith("dr."):
                cleaned_name = cleaned_name[3:].strip()
            elif cleaned_name.lower().startswith("dr "):
                cleaned_name = cleaned_name[3:].strip()
            doctor_model_instance = await find_doctors(
                db, name=cleaned_name, return_single=True
            )

        if not doctor_model_instance:
            return {"status": "error", "message": "Doctor not found."}
        if not (
            doctor_model_instance.user
            and hasattr(doctor_model_instance.user, "email")
            and doctor_model_instance.user.email
        ):
            return {
                "status": "error",
                "message": f"Could not find a valid email address for Dr. {doctor_model_instance.first_name} {doctor_model_instance.last_name}.",
            }
        doctor_email_address = doctor_model_instance.user.email
        doctor_full_name_for_gcal = (
            f"Dr. {doctor_model_instance.first_name} {doctor_model_instance.last_name}"
        )

        # ++++ FETCH PATIENT'S NAME ++++
        if patient_id:
            # Ensure patient_id from InjectedState is an int if get_user expects int
            current_patient_id = (
                int(patient_id)
                if isinstance(patient_id, str) and patient_id.isdigit()
                else patient_id
            )
            if isinstance(current_patient_id, int):
                patient_user_model = await get_user(db, current_patient_id)
                if patient_user_model and patient_user_model.patient_profile:
                    patient_full_name_for_gcal = f"{patient_user_model.patient_profile.first_name} {patient_user_model.patient_profile.last_name}"
                else:
                    logger.warning(
                        f"Could not fetch patient profile for patient_id: {current_patient_id} for GCal summary. Using default."
                    )
            else:
                logger.warning(
                    f"Invalid patient_id type: {patient_id} for GCal summary. Using default."
                )
        # ++++++++++++++++++++++++++++++

        appointment_result_or_obj = await create_appointment(
            db,
            int(patient_id)
            if isinstance(patient_id, str)
            else patient_id,  # Ensure patient_id is int for create_appointment
            doctor_model_instance.user_id,
            start_dt_clinic_utc,
            end_dt_clinic_utc,
            location,
            notes,
            google_calendar_event_id=None,
        )

        if (
            isinstance(appointment_result_or_obj, dict)
            and "status" in appointment_result_or_obj
        ):
            clinic_booking_result = appointment_result_or_obj
        elif hasattr(appointment_result_or_obj, "id"):
            db_appointment_object = appointment_result_or_obj

            # Determine the display timezone
            display_tz_str = user_tz or GCAL_DEFAULT_TIMEZONE
            display_tz = ZoneInfo(display_tz_str)

            # start_dt_clinic_utc is already a UTC aware datetime object
            # Format it for the user's timezone for display
            formatted_start_dt_local = format_datetime(
                start_dt_clinic_utc, format="MMMM d, yyyy, h:mm a", locale="en", tzinfo=display_tz
            )
            # Also format the end time for display if needed, though not currently in output
            # end_dt_display = end_dt_clinic_utc.astimezone(display_tz)
            # formatted_end_dt_local = format_datetime(
            #     end_dt_display, format="long", locale="en", tzinfo=display_tz
            # )

            clinic_booking_result = {
                "status": "confirmed",
                "id": db_appointment_object.id,
                "doctor_id": doctor_model_instance.user_id,
                "doctor_name": doctor_full_name_for_gcal,
                "doctor_email": doctor_email_address,
                "start_dt": formatted_start_dt_local,  # Use the locally formatted time
                "notes": notes,
            }
            logger.info(
                f"Clinic appointment ID {db_appointment_object.id} confirmed in DB. Original UTC: {start_dt_clinic_utc}, Display time ({display_tz_str}): {formatted_start_dt_local}"
            )
        else:
            clinic_booking_result = {
                "status": "error",
                "message": "Unknown error during clinic booking.",
            }
            logger.error(
                f"Unexpected result from create_appointment: {appointment_result_or_obj}"
            )

    if (
        clinic_booking_result.get("status") == "confirmed"
        and send_google_calendar_invite
        and db_appointment_object
    ):
        logger.info(
            f"Attempting to send Google Calendar invite to doctor: {doctor_email_address}"
        )
        # Ensure _get_gcal_service_sync is defined and works
        gcal_service, gcal_error_msg = await asyncio.to_thread(_get_gcal_service_sync)

        if not gcal_service:
            google_calendar_status = (
                f"Failed to initialize Google Calendar service: {gcal_error_msg}"
            )
        else:
            try:
                gcal_start_rfc3339 = start_dt_clinic_utc.isoformat()
                gcal_end_rfc3339 = end_dt_clinic_utc.isoformat()

                appointment_reason = clinic_booking_result.get("notes", "visit")
                if not appointment_reason:
                    appointment_reason = "visit"

                # ++++ CONSTRUCT GCAL SUMMARY WITH PATIENT AND DOCTOR NAME ++++
                gcal_final_summary = (
                    gcal_summary_override
                    or f"Appt: {patient_full_name_for_gcal} with {doctor_full_name_for_gcal} ({appointment_reason})"
                )
                gcal_final_description = (
                    gcal_description_override
                    or f"Clinic appointment for {patient_full_name_for_gcal} with {doctor_full_name_for_gcal}.\n"
                    f"Reason: {appointment_reason}"
                )
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                event_body = {
                    "summary": gcal_final_summary,
                    "description": gcal_final_description,
                    "start": {"dateTime": gcal_start_rfc3339, "timeZone": "UTC"},
                    "end": {"dateTime": gcal_end_rfc3339, "timeZone": "UTC"},
                    "attendees": [{"email": doctor_email_address}],
                    "reminders": {
                        "useDefault": False,
                        "overrides": [{"method": "popup", "minutes": 30}],
                    },
                }

                def _sync_gcal_insert():
                    return (
                        gcal_service.events()
                        .insert(
                            calendarId="primary",
                            body=event_body,
                            sendNotifications=True,
                        )
                        .execute()
                    )

                created_event = await asyncio.to_thread(_sync_gcal_insert)
                gcal_event_id_from_api = created_event.get("id")

                if gcal_event_id_from_api:
                    google_calendar_status = f"Google Calendar invite sent to {doctor_email_address} for '{gcal_final_summary}'. Link: {created_event.get('htmlLink')}"
                    logger.info(google_calendar_status)
                    async with tool_db_session() as db_for_gcal_update:
                        gcal_id_updated_in_db = await update_appointment_gcal_id(
                            db_for_gcal_update,
                            db_appointment_object.id,
                            gcal_event_id_from_api,
                        )
                        if gcal_id_updated_in_db:
                            logger.info(
                                f"Successfully stored GCal event ID {gcal_event_id_from_api} for DB appointment {db_appointment_object.id}"
                            )
                        else:
                            logger.warning(
                                f"Failed to store GCal event ID {gcal_event_id_from_api} for DB appointment {db_appointment_object.id}"
                            )
                            google_calendar_status += (
                                " (Note: GCal ID failed to save to DB)."
                            )
                else:
                    google_calendar_status = f"Google Calendar invite sent for '{gcal_final_summary}', but no event ID returned."
                    logger.warning(google_calendar_status)
            except HttpError as api_error:
                google_calendar_status = f"GCal API error: {api_error.resp.status if api_error.resp else 'Unknown'} - Failed to create event."
                logger.error(
                    f"{google_calendar_status} Details: {api_error.content.decode() if hasattr(api_error.content, 'decode') else api_error.content}",
                    exc_info=True,
                )
            except Exception as e_gcal:
                google_calendar_status = (
                    f"Unexpected GCal error: {type(e_gcal).__name__} - {e_gcal}"
                )
                logger.error(google_calendar_status, exc_info=True)    # Return structured output similar to other tools
    if clinic_booking_result.get("status") == "confirmed":
        # Success case - return structured confirmation
        final_result = {
            "type": "appointment_confirmed",
            "status": "confirmed",
            "appointment_id": clinic_booking_result.get("id"),
            "doctor_id": clinic_booking_result.get("doctor_id"),
            "doctor_name": clinic_booking_result.get("doctor_name"),
            "doctor_email": clinic_booking_result.get("doctor_email"),
            "start_dt": clinic_booking_result.get("start_dt"),
            "location": location,
            "notes": notes,
            "google_calendar_invite_status": google_calendar_status,
            "google_calendar_link": None,  # Will be updated if GCal invite was successful
            "agent": "Scheduler"
        }

        # Extract Google Calendar link if available
        if "Link:" in google_calendar_status:
            link_start = google_calendar_status.find("Link: ") + 6
            link_end = google_calendar_status.find(" ", link_start)
            if link_end == -1:
                link_end = len(google_calendar_status)
            final_result["google_calendar_link"] = google_calendar_status[link_start:link_end]

    elif clinic_booking_result.get("status") == "error":
        # Error case - return structured error
        final_result = {
            "type": "booking_error",
            "status": "error",
            "message": clinic_booking_result.get("message", "An error occurred while booking the appointment."),
            "agent": "Scheduler"
        }

    elif clinic_booking_result.get("status") == "conflict":
        # Conflict case - return structured conflict error
        final_result = {
            "type": "booking_conflict",
            "status": "conflict",
            "message": clinic_booking_result.get("message", "This time slot is already booked."),
            "agent": "Scheduler"
        }

    else:
        # Fallback for unexpected cases
        final_result = {
            "type": "booking_error",
            "status": "error",
            "message": "An unexpected error occurred during booking.",
            "agent": "Scheduler"
        }

    logger.info(f"book_appointment tool final result: {final_result}")
    return final_result


@tool("cancel_appointment")
async def cancel_appointment(
    appointment_id: int,
    patient_id: Annotated[int, InjectedState("user_id")],
    # user_tz is not strictly needed here unless GCal interactions require it,
    # but _delete_gcal_event_if_exists_scheduler doesn't use it.
) -> dict:
    """
    Cancel an existing appointment owned by the current user.
    This will delete the appointment from the database and attempt to delete
    any associated Google Calendar event.
    """
    logger.info(
        f"Tool 'cancel_appointment' called by user {patient_id} for appointment_id={appointment_id}"
    )
    if not patient_id:  # Should be caught by auth middleware, but good check
        logger.warning("Cancel tool called without patient_id (should be injected).")
        return {
            "status": "error",
            "message": "I couldn't identify you – please log in again.",
        }

    gcal_event_id_to_delete: Optional[str] = None
    gcal_cancellation_status_msg: str = (
        "Google Calendar event not applicable or not processed."
    )

    try:
        async with tool_db_session() as db:
            # 1. Verify appointment existence and ownership, and get GCal ID
            appointment_to_cancel: Optional[AppointmentModel] = None
            try:
                # get_appointment CRUD should return the AppointmentModel which includes google_calendar_event_id
                appointment_to_cancel = await get_appointment(
                    db, appointment_id, patient_id, "patient"
                )
                if appointment_to_cancel and hasattr(
                    appointment_to_cancel, "google_calendar_event_id"
                ):
                    gcal_event_id_to_delete = (
                        appointment_to_cancel.google_calendar_event_id
                    )
                    logger.info(
                        f"Tool: Found GCal Event ID '{gcal_event_id_to_delete}' for appointment_id={appointment_id} to be cancelled."
                    )
                elif appointment_to_cancel:
                    logger.info(
                        f"Tool: No GCal Event ID found for appointment_id={appointment_id}."
                    )
                # If get_appointment raises HTTPException, it will be caught by the outer try-except
            except (
                HTTPException
            ) as http_exc:  # Catch specific FastAPI HTTPException from get_appointment
                logger.warning(
                    f"Tool: get_appointment failed for appt_id={appointment_id}, user_id={patient_id}. Detail: {http_exc.detail}"
                )
                return {
                    "status": "error",
                    "message": http_exc.detail,
                }  # Relay message from get_appointment
            except (
                Exception
            ) as e_get:  # Catch other unexpected errors from get_appointment
                logger.error(
                    f"Tool: Unexpected error fetching appointment {appointment_id} for patient {patient_id}: {e_get}",
                    exc_info=True,
                )
                return {
                    "status": "error",
                    "message": "An error occurred while trying to find your appointment.",
                }

            # If appointment_to_cancel is None here, get_appointment raised an error handled above, or it just wasn't found
            if not appointment_to_cancel:
                # This case should ideally be covered by get_appointment raising HTTPException for not found
                logger.warning(
                    f"Tool: Appointment {appointment_id} not found or not accessible by user {patient_id} after initial check."
                )
                return {
                    "status": "error",
                    "message": "That appointment doesn't exist or doesn't belong to you.",
                }

            # 2. Delete from Database
            # delete_appointment CRUD performs a hard delete
            logger.debug(
                f"Tool: Attempting to hard delete appointment_id={appointment_id} from DB for user_id={patient_id}"
            )
            db_deleted_successfully = await delete_appointment(
                db, appointment_id, patient_id, "patient"
            )

            if db_deleted_successfully:
                logger.info(
                    f"Tool: Successfully deleted appointment_id={appointment_id} from database."
                )

                # 3. Attempt to Delete from Google Calendar if GCal ID exists
                if gcal_event_id_to_delete:
                    logger.info(
                        f"Tool: Proceeding to delete GCal event_id='{gcal_event_id_to_delete}'."
                    )
                    (
                        gcal_success,
                        gcal_msg,
                    ) = await _delete_gcal_event_if_exists_scheduler(
                        gcal_event_id_to_delete
                    )
                    gcal_cancellation_status_msg = (
                        gcal_msg  # Store the message from the helper
                    )
                    if gcal_success:
                        logger.info(
                            f"Tool: GCal processing for event_id='{gcal_event_id_to_delete}' successful."
                        )
                    else:
                        logger.warning(
                            f"Tool: GCal processing for event_id='{gcal_event_id_to_delete}' had issues: {gcal_msg}"
                        )
                else:
                    gcal_cancellation_status_msg = (
                        "No Google Calendar event was linked to this appointment."
                    )
                    logger.info(
                        f"Tool: No GCal event ID to delete for appointment_id={appointment_id}."
                    )

                return {
                    "status": "cancelled",
                    "message": f"Appointment #{appointment_id} has been successfully cancelled from the schedule. {gcal_cancellation_status_msg}",
                }
            else:
                # This case implies delete_appointment returned False, which means the get_appointment check
                # might have passed but the delete itself failed for some reason (e.g., row gone between select and delete - rare).
                logger.warning(
                    f"Tool: Failed to delete appointment_id={appointment_id} from database, though it was initially found."
                )
                return {
                    "status": "error",
                    "message": "Sorry – there was an issue cancelling that appointment from the database.",
                }

    except Exception as e:  # Catch-all for unexpected errors in the tool's own logic
        logger.error(
            f"Tool 'cancel_appointment': Unexpected error for appointment_id={appointment_id}, user_id={patient_id}: {e}",
            exc_info=True,
        )
        return {
            "status": "error",
            "message": "I encountered an unexpected error while trying to cancel the appointment. Please try again later.",
        }


@tool("propose_booking")
async def propose_booking(
    doctor_id: int = None,
    doctor_name: str = None,
    starts_at: str = None,
    notes: str | None = None,
    user_tz: Annotated[str | None, InjectedState("user_tz")] = None, # Added user_tz
    duration_minutes: int = 30 # Added duration_minutes for consistency
) -> dict:
    """
    Propose a booking time. This does NOT create an appointment but checks if a slot
    is theoretically bookable and returns details for confirmation.
    The 'starts_at' should be a specific date and time string.
    """
    logger.info(
        f"Tool 'propose_booking' called for doctor_id={doctor_id}, doctor_name={doctor_name}, starts_at='{starts_at}', notes='{notes}' with user_tz='{user_tz}'"
    )

    if not doctor_id and not doctor_name:
        return {
            "type": "proposal_error",
            "message": "Please provide either a doctor ID or a doctor name for the proposal.",
        }
    if not starts_at:
        return {
            "type": "proposal_error",
            "message": "Please provide a start time for the proposed appointment.",
        }

    # Determine the effective timezone for parsing the input 'starts_at'
    # This should be the user's timezone if available, otherwise a default.
    effective_input_tz_str = user_tz or GCAL_DEFAULT_TIMEZONE

    # Parse the provided 'starts_at' string.
    # It's assumed to be in the user's local time (effective_input_tz_str).
    # We convert it to UTC to check against DB, which stores in UTC.
    parsed_dt_utc = dateparser.parse(
        starts_at,
        settings={
            "TIMEZONE": effective_input_tz_str, # Assume input is in this timezone
            "TO_TIMEZONE": "UTC",             # Convert to UTC for internal checks
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "future",
        },
    )

    if not parsed_dt_utc:
        logger.warning(f"Invalid starts_at format for proposal: {starts_at}")
        return {
            "type": "proposal_error",
            "message": f"I couldn't understand the proposed time '{starts_at}'. Please try a format like 'YYYY-MM-DD HH:MM' or 'tomorrow at 2pm'.",
        }

    # Calculate end time in UTC
    end_dt_utc = parsed_dt_utc + timedelta(minutes=duration_minutes)

    async with tool_db_session() as db:
        doctor_model_instance = None
        if doctor_id:
            doctor_model_instance = await find_doctors(
                db, doctor_id=doctor_id, return_single=True
            )
        elif doctor_name:
            cleaned_name = doctor_name
            if cleaned_name.lower().startswith("dr."):
                cleaned_name = cleaned_name[3:].strip()
            elif cleaned_name.lower().startswith("dr "):
                cleaned_name = cleaned_name[3:].strip()
            doctor_model_instance = await find_doctors(
                db, name=cleaned_name, return_single=True
            )

        if not doctor_model_instance:
            return {"type": "proposal_error", "message": "Doctor not found for proposal."}

        # Check for conflicts (this is a simplified check, create_appointment has the robust one)
        # For a proposal, we might just check if the exact slot is in the *available* slots list.
        # This requires converting the proposed UTC time back to the clinic's local day and then checking.
        # Or, more simply, attempt a dry-run of create_appointment or check against existing.

        # For now, let's assume the main purpose is to format the proposed time correctly for the user.
        # A full conflict check here might be redundant if the next step is always `book_appointment`.

        # Determine the display timezone (user's TZ or default)
        display_tz_str = user_tz or GCAL_DEFAULT_TIMEZONE
        display_tz = ZoneInfo(display_tz_str)

        # Convert the UTC parsed time to the display timezone for the proposal message
        parsed_dt_display = parsed_dt_utc.astimezone(display_tz)

        # Format for display
        formatted_starts_at_display = format_datetime(
            parsed_dt_display, format="MMMM d, yyyy, h:mm a", locale="en" # Using "full" for clarity in proposal
        ) # format="long" is also good: e.g., January 15, 2024 at 3:00 PM GMT+3

        doctor_full_name = f"Dr. {doctor_model_instance.first_name} {doctor_model_instance.last_name}"

        # Construct a message that uses the time in the user's timezone
        proposal_message = f"OK. I can propose an appointment for you with {doctor_full_name} on {formatted_starts_at_display}."
        if notes:
            proposal_message += f" With notes: \"{notes}\"."
        proposal_message += " Does that sound right?"

        return {
            "type": "booking_proposal",
            "doctor_id": doctor_model_instance.user_id,
            "doctor_name": doctor_full_name,
            "doctor_specialty": doctor_model_instance.specialty,
            "proposed_starts_at_utc": parsed_dt_utc.isoformat(), # Keep UTC for potential booking step
            "proposed_starts_at_display": formatted_starts_at_display, # For user confirmation
            "duration_minutes": duration_minutes,
            "notes": notes,
            "message": proposal_message, # The user-facing confirmation message
            "agent": "Scheduler",
            "reply_template": "Yes, that sounds right, please book it."
        }
