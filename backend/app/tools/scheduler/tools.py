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

import asyncio
from pathlib import Path # If not already there
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pytz # If not already there


logger = logging.getLogger(__name__)

GCAL_SCOPES = ['https://www.googleapis.com/auth/calendar.events']
COMMON_TOOLS_DIR = Path(__file__).resolve().parent.parent
GCAL_TOKEN_FILE_PATH = COMMON_TOOLS_DIR / 'calendar' / 'token.json'
GCAL_DEFAULT_TIMEZONE = 'Asia/Beirut'

# --- Google Calendar Helper Function ---
def _get_gcal_service_sync():
    creds = None
    # Add logging for the path being checked
    logger.info(f"Google Calendar: Attempting to access token file at resolved path: {GCAL_TOKEN_FILE_PATH.resolve()}") # Using .resolve() for absolute path logging
    
    if not GCAL_TOKEN_FILE_PATH.exists():
        logger.error(f"Google Calendar: Token file NOT FOUND at calculated path: {GCAL_TOKEN_FILE_PATH}")
        return None, f"Configuration error: Google Calendar token file not found. Checked: {GCAL_TOKEN_FILE_PATH}"
    
    logger.info(f"Google Calendar: Token file found at {GCAL_TOKEN_FILE_PATH}. Proceeding to load.")
    try:
        creds = Credentials.from_authorized_user_file(str(GCAL_TOKEN_FILE_PATH), GCAL_SCOPES)
    except Exception as e:
        logger.error(f"Google Calendar: Error loading credentials from {GCAL_TOKEN_FILE_PATH}: {e}", exc_info=True)
        return None, f"Error loading Google Calendar credentials from {GCAL_TOKEN_FILE_PATH}: {e}."

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.warning(f"Google Calendar: Credentials from {GCAL_TOKEN_FILE_PATH} are expired. Refresh might be needed (requires client secrets, not done by this tool). API call might fail.")
        else:
            logger.error(f"Google Calendar: Could not load valid credentials from {GCAL_TOKEN_FILE_PATH}. Token might be corrupted or missing required fields.")
            return None, "Invalid or missing Google Calendar credentials. Token may be expired or improperly formatted."
    try:
        service = build('calendar', 'v3', credentials=creds, cache_discovery=False) # Added cache_discovery=False for potential GCE issues
        logger.info("Google Calendar service object created successfully.")
        return service, None
    except HttpError as error:
        logger.error(f'Google Calendar: API error building service: {error}. Details: {error.content}', exc_info=True)
        return None, f"API error building Google Calendar service: {error.resp.status if error.resp else 'Unknown'}"
    except Exception as e:
        logger.error(f'Google Calendar: Unexpected error building service: {e}', exc_info=True)
        return None, f"Unexpected error building Google Calendar service: {e}"

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


# In backend/app/tools/scheduler/tools.py

@tool("book_appointment")
async def book_appointment(
    doctor_id: int = None,
    doctor_name: str = None,
    starts_at: str = None, # This is for the clinic appointment
    patient_id: Annotated[int, InjectedState("user_id")] = None,
    user_tz: Annotated[str | None, InjectedState("user_tz")] = None,
    duration_minutes: int = 30,
    location: str = "Main Clinic",
    notes: str | None = None,
    send_google_calendar_invite: bool = False, # NEW PARAMETER
    # Optional overrides for Google Calendar details if LLM wants to be specific
    gcal_summary_override: Optional[str] = None,
    gcal_description_override: Optional[str] = None,
    gcal_event_time_override_hhmm: Optional[str] = None # For "TOMORROW HH:MM" if different from clinic time
) -> dict:
    """
    Create a clinic appointment. If send_google_calendar_invite is true, will also attempt
    to send a Google Calendar invite for TOMORROW to the doctor.
    Returns DB confirmation and Google Calendar status.
    The starts_at parameter is for the clinic appointment and can be a specific date/time.
    The Google Calendar invite will be for TOMORROW using gcal_event_time_override_hhmm if provided,
    otherwise it will try to use the time from the clinic appointment if it's for tomorrow,
    or a default time for tomorrow as a reminder if the clinic appointment is on a different day.
    """
    logger.info(f"Tool 'book_appointment' called by user {patient_id} for doctor_id={doctor_id}, doctor_name={doctor_name} at {starts_at}. Send GCal: {send_google_calendar_invite}")

    # --- 1. Clinic Appointment Booking Logic (mostly as before) ---
    if not doctor_id and not doctor_name: # Basic validation
        return {"status": "error", "message": "Please provide either a doctor ID or a doctor name for the clinic appointment."}
    if not starts_at:
        return {"status": "error", "message": "Please provide a start time for the clinic appointment."}
    if not patient_id:
        logger.warning("Booking tool called without patient_id.")
        return {"status": "error", "message": "I couldn't identify you – please log in again."}

    # Parse start_dt for clinic appointment (as before)
    # ... (your existing start_dt parsing logic using dateparser and user_tz)
    # For brevity, assuming start_dt_clinic (UTC datetime object) is correctly parsed here
    
    # Example placeholder for clinic start time parsing
    parsed_clinic_dt = dateparser.parse(starts_at, settings={
        'TIMEZONE': user_tz or GCAL_DEFAULT_TIMEZONE, # Use user_tz or a default
        'TO_TIMEZONE': 'UTC', # Ensure it's converted to UTC
        'RETURN_AS_TIMEZONE_AWARE': True,
        'PREFER_DATES_FROM': 'future'
    })
    if not parsed_clinic_dt:
        logger.warning(f"Invalid starts_at format for clinic appointment: {starts_at}")
        return {"status": "error", "message": "Invalid start time for clinic appointment."}
    start_dt_clinic_utc = parsed_clinic_dt # This should be a timezone-aware datetime object in UTC

    end_dt_clinic_utc = start_dt_clinic_utc + timedelta(minutes=duration_minutes)
    
    clinic_booking_result = {}
    google_calendar_status = "Not attempted."
    doctor_email_address = None # Will be populated if clinic booking succeeds

    async with tool_db_session() as db:
        doctor = None # Fetch doctor as before
        if doctor_id:
            doctor = await find_doctors(db, doctor_id=doctor_id, return_single=True)
        elif doctor_name:
            cleaned_name = doctor_name # ... (your name cleaning logic) ...
            doctor = await find_doctors(db, name=cleaned_name, return_single=True)

        if not doctor:
            return {"status": "error", "message": f"Doctor not found for clinic appointment."}
        
        if not (doctor.user and hasattr(doctor.user, 'email')):
             return {"status": "error", "message": f"Could not find email address for Dr. {doctor.first_name} {doctor.last_name}."}
        doctor_email_address = doctor.user.email


        appointment = await create_appointment(
            db, patient_id, doctor.user_id, start_dt_clinic_utc, end_dt_clinic_utc, location, notes
        )

        if isinstance(appointment, dict) and "status" in appointment: # Error from create_appointment
            clinic_booking_result = appointment # e.g. {"status": "conflict", "message": "..."}
        elif hasattr(appointment, 'id'): # Success
            clinic_booking_result = {
                "status": "confirmed",
                "id": appointment.id,
                "doctor_id": doctor.user_id,
                "doctor_name": f"Dr. {doctor.first_name} {doctor.last_name}",
                "doctor_email": doctor_email_address,
                "start_dt": format_datetime(start_dt_clinic_utc, 'long', locale='en', tzinfo=pytz.utc), # Ensure it's formatted as UTC
                "notes": notes
            }
            logger.info(f"Clinic appointment ID {appointment.id} confirmed.")
        else: # Unexpected result
            clinic_booking_result = {"status": "error", "message": "Unknown error during clinic booking."}


    # --- 2. Google Calendar Invite Logic (if clinic booking was successful and requested) ---
    if clinic_booking_result.get("status") == "confirmed" and send_google_calendar_invite:
                logger.info(f"Attempting to send Google Calendar invite to doctor: {doctor_email_address}")
                
                gcal_service, gcal_error_msg = await asyncio.to_thread(_get_gcal_service_sync) # Run sync in thread

                if not gcal_service:
                    google_calendar_status = f"Failed to initialize Google Calendar service: {gcal_error_msg}"
                else:
                    try:
                        # Determine event time for Google Calendar (TOMORROW)
                        event_tz_str = user_tz or GCAL_DEFAULT_TIMEZONE # user_tz from InjectedState
                        gcal_event_pytz = pytz.timezone(event_tz_str)
                        
                        # CORRECTED: Use datetime.now() directly, not datetime.datetime.now()
                        now_gcal_local = datetime.now(gcal_event_pytz)
                        # CORRECTED: Use timedelta() directly, not datetime.timedelta()
                        tomorrow_gcal_local = now_gcal_local + timedelta(days=1)

                        event_hour, event_minute = 9, 0 # Default time for tomorrow's GCal event
                        
                        # Logic to determine summary and time for GCal event
                        gcal_final_summary = gcal_summary_override
                        final_event_time_str_hhmm = gcal_event_time_override_hhmm

                        # Check if clinic appointment is for tomorrow
                        # start_dt_clinic_utc should be a timezone-aware datetime object (in UTC) from earlier in the function
                        is_clinic_appt_tomorrow = (start_dt_clinic_utc.astimezone(gcal_event_pytz).date() == tomorrow_gcal_local.date())

                        if not gcal_final_summary:
                            gcal_final_summary = f"Appointment: Patient & {clinic_booking_result['doctor_name']}"
                            if not is_clinic_appt_tomorrow:
                                clinic_appt_formatted_for_summary = start_dt_clinic_utc.astimezone(gcal_event_pytz).strftime('%b %d at %I:%M %p %Z')
                                gcal_final_summary = f"REMINDER: {gcal_final_summary} (Actual appt: {clinic_appt_formatted_for_summary})"

                        if not final_event_time_str_hhmm: # If LLM didn't specify a GCal time
                            if is_clinic_appt_tomorrow:
                                final_event_time_str_hhmm = start_dt_clinic_utc.astimezone(gcal_event_pytz).strftime('%H:%M')
                            else: # Clinic appt not tomorrow, use default 9 AM for reminder
                                final_event_time_str_hhmm = "09:00"
                        
                        try:
                            event_hour, event_minute = map(int, final_event_time_str_hhmm.split(':'))
                        except ValueError:
                            logger.warning(f"Invalid GCal event time format '{final_event_time_str_hhmm}', using 09:00.")
                            event_hour, event_minute = 9, 0

                        gcal_start_dt_local = tomorrow_gcal_local.replace(hour=event_hour, minute=event_minute, second=0, microsecond=0)
                        # Use clinic appointment duration for GCal event
                        gcal_duration_hours = duration_minutes / 60.0 
                        # CORRECTED: Use timedelta() directly
                        gcal_end_dt_local = gcal_start_dt_local + timedelta(hours=gcal_duration_hours)

                        gcal_start_rfc3339 = gcal_start_dt_local.isoformat()
                        gcal_end_rfc3339 = gcal_end_dt_local.isoformat()

                        gcal_final_description = gcal_description_override or clinic_booking_result.get("notes") or gcal_final_summary

                        event_body = {
                            'summary': gcal_final_summary,
                            'description': gcal_final_description,
                            'start': {'dateTime': gcal_start_rfc3339, 'timeZone': event_tz_str},
                            'end': {'dateTime': gcal_end_rfc3339, 'timeZone': event_tz_str},
                            'attendees': [{'email': doctor_email_address}], # Invite the doctor
                            'reminders': {'useDefault': False, 'overrides': [{'method': 'popup', 'minutes': 30}]},
                        }
                        
                        def _sync_gcal_insert(): # Inner sync function for asyncio.to_thread
                            # Ensure gcal_service is in the closure or passed
                            return gcal_service.events().insert(calendarId='primary', body=event_body, sendNotifications=True).execute()
                        
                        created_event = await asyncio.to_thread(_sync_gcal_insert)
                        
                        google_calendar_status = f"Google Calendar invite sent to {doctor_email_address} for {gcal_final_summary}. Link: {created_event.get('htmlLink')}"
                        logger.info(google_calendar_status)

                    except HttpError as api_error:
                        google_calendar_status = f"Google Calendar API error: {api_error.resp.status if api_error.resp else 'Unknown'} - Failed to create event."
                        # Log the full content for debugging API errors
                        error_content = api_error.content.decode('utf-8') if hasattr(api_error, 'content') and isinstance(api_error.content, bytes) else str(api_error.content)
                        logger.error(f"{google_calendar_status} Details: {error_content}", exc_info=True) # Added exc_info for full traceback
                    except Exception as e_gcal:
                        google_calendar_status = f"Unexpected error during Google Calendar scheduling: {type(e_gcal).__name__} - {e_gcal}" # More specific error type
                        logger.error(google_calendar_status, exc_info=True)
    
    # --- 3. Consolidate and Return ---
    final_result = {**clinic_booking_result} # Start with clinic booking status
    final_result["google_calendar_invite_status"] = google_calendar_status
    return final_result


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
