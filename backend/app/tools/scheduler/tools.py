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
    get_appointment,
    update_appointment_gcal_id
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

async def _delete_gcal_event_if_exists_scheduler(event_id: str, calendar_id: str = 'primary') -> tuple[bool, str]:
    """
    Helper to delete a single GCal event.
    Returns (success_bool, message_str).
    """
    if not event_id:
        return True, "No GCal event ID provided for deletion."

    logger.info(f"Scheduler GCal Helper: Attempting to get GCal service for deleting event_id: {event_id}")
    service, error_msg = await asyncio.to_thread(_get_gcal_service_sync) # Uses the one defined/imported in this file
    if not service:
        logger.error(f"Scheduler GCal Helper: Failed to get Google Calendar service: {error_msg}")
        return False, f"Failed to connect to Google Calendar: {error_msg}"

    try:
        logger.info(f"Scheduler GCal Helper: Attempting to delete Google Calendar event: {event_id} from calendar: {calendar_id}")
        await asyncio.to_thread(
            service.events().delete(calendarId=calendar_id, eventId=event_id, sendUpdates='all').execute
        )
        logger.info(f"Scheduler GCal Helper: Successfully deleted Google Calendar event: {event_id}")
        return True, f"Google Calendar event {event_id} successfully deleted."
    except HttpError as e:
        if e.resp.status == 404:
            logger.warning(f"Scheduler GCal Helper: Google Calendar event {event_id} not found for deletion (404).")
            return True, f"Google Calendar event {event_id} not found (might be already deleted)."
        logger.error(f"Scheduler GCal Helper: HttpError deleting event {event_id}: {e.resp.status} - {e.content}", exc_info=True)
        return False, f"Google Calendar API error deleting event {event_id}: {e.resp.status}"
    except Exception as e:
        logger.error(f"Scheduler GCal Helper: Unexpected error deleting event {event_id}: {e}", exc_info=True)
        return False, f"Unexpected error deleting Google Calendar event {event_id}: {str(e)}"

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
    send_google_calendar_invite: bool = False,
    gcal_summary_override: Optional[str] = None,
    gcal_description_override: Optional[str] = None,
    gcal_event_time_override_hhmm: Optional[str] = None
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

    # --- Basic Validations ---
    if not doctor_id and not doctor_name:
        return {"status": "error", "message": "Please provide either a doctor ID or a doctor name for the clinic appointment."}
    if not starts_at:
        return {"status": "error", "message": "Please provide a start time for the clinic appointment."}
    if not patient_id:
        logger.warning("Booking tool called without patient_id.")
        return {"status": "error", "message": "I couldn't identify you – please log in again."}

    # --- 1. Parse Clinic Appointment Time (ensure it's UTC) ---
    parsed_clinic_dt = dateparser.parse(starts_at, settings={
        'TIMEZONE': user_tz or GCAL_DEFAULT_TIMEZONE,
        'TO_TIMEZONE': 'UTC',
        'RETURN_AS_TIMEZONE_AWARE': True,
        'PREFER_DATES_FROM': 'future'
    })
    if not parsed_clinic_dt:
        logger.warning(f"Invalid starts_at format for clinic appointment: {starts_at}")
        return {"status": "error", "message": "Invalid start time for clinic appointment."}
    start_dt_clinic_utc = parsed_clinic_dt
    end_dt_clinic_utc = start_dt_clinic_utc + timedelta(minutes=duration_minutes)

    # --- 2. Clinic Appointment Booking in DB (Initial Step) ---
    clinic_booking_result: Dict[str, Any] = {}
    google_calendar_status: str = "Not attempted."
    doctor_email_address: Optional[str] = None
    db_appointment_object = None # <<<<< Will hold the created AppointmentModel instance

    async with tool_db_session() as db:
        doctor_model_instance = None
        if doctor_id:
            doctor_model_instance = await find_doctors(db, doctor_id=doctor_id, return_single=True)
        elif doctor_name:
            cleaned_name = doctor_name
            if cleaned_name.lower().startswith("dr."):
                cleaned_name = cleaned_name[3:].strip()
            elif cleaned_name.lower().startswith("dr "):
                cleaned_name = cleaned_name[3:].strip()
            doctor_model_instance = await find_doctors(db, name=cleaned_name, return_single=True)

        if not doctor_model_instance:
            return {"status": "error", "message": f"Doctor not found for clinic appointment."}

        if not (doctor_model_instance.user and hasattr(doctor_model_instance.user, 'email') and doctor_model_instance.user.email):
             return {"status": "error", "message": f"Could not find a valid email address for Dr. {doctor_model_instance.first_name} {doctor_model_instance.last_name}."}
        doctor_email_address = doctor_model_instance.user.email

        # Call create_appointment from your CRUD, passing None for google_calendar_event_id initially
        # It should return Union[AppointmentModel, Dict]
        appointment_result_or_obj = await create_appointment(
            db,
            patient_id=patient_id,
            doctor_id=doctor_model_instance.user_id,
            starts_at=start_dt_clinic_utc,
            ends_at=end_dt_clinic_utc,
            location=location,
            notes=notes,
            google_calendar_event_id=None # Explicitly pass None here
        )

        if isinstance(appointment_result_or_obj, dict) and "status" in appointment_result_or_obj:
            clinic_booking_result = appointment_result_or_obj # Error from create_appointment
        elif hasattr(appointment_result_or_obj, 'id'): # Successfully created AppointmentModel
            db_appointment_object = appointment_result_or_obj # <<<<< STORE THE DB OBJECT
            clinic_booking_result = {
                "status": "confirmed",
                "id": db_appointment_object.id,
                "doctor_id": doctor_model_instance.user_id,
                "doctor_name": f"Dr. {doctor_model_instance.first_name} {doctor_model_instance.last_name}",
                "doctor_email": doctor_email_address,
                "start_dt": format_datetime(start_dt_clinic_utc, 'long', locale='en', tzinfo=pytz.utc),
                "notes": notes
            }
            logger.info(f"Clinic appointment ID {db_appointment_object.id} confirmed in DB.")
        else: # Unexpected result
            clinic_booking_result = {"status": "error", "message": "Unknown error during clinic booking."}
            logger.error(f"Unexpected result from create_appointment: {appointment_result_or_obj}")


    # --- 3. Google Calendar Invite Logic (if clinic booking was successful AND requested AND db_appointment_object exists) ---
    if clinic_booking_result.get("status") == "confirmed" and send_google_calendar_invite and db_appointment_object: # <<<<< ADDED CHECK FOR db_appointment_object
        logger.info(f"Attempting to send Google Calendar invite to doctor: {doctor_email_address}")

        gcal_service, gcal_error_msg = await asyncio.to_thread(_get_gcal_service_sync)

        if not gcal_service:
            google_calendar_status = f"Failed to initialize Google Calendar service: {gcal_error_msg}"
        else:
            try:
                # ... (Your existing logic for:
                #       event_tz_str, gcal_event_pytz, now_gcal_local, tomorrow_gcal_local,
                #       gcal_final_summary, final_event_time_str_hhmm, is_clinic_appt_tomorrow,
                #       parsing event_hour/minute,
                #       gcal_start_dt_local, gcal_duration_hours, gcal_end_dt_local,
                #       gcal_start_rfc3339, gcal_end_rfc3339,
                #       gcal_final_description, event_body
                #      This all looks correct in your pasted code)
                event_tz_str = user_tz or GCAL_DEFAULT_TIMEZONE
                gcal_event_pytz = pytz.timezone(event_tz_str)
                now_gcal_local = datetime.now(gcal_event_pytz)
                tomorrow_gcal_local = now_gcal_local + timedelta(days=1)
                gcal_final_summary = gcal_summary_override
                final_event_time_str_hhmm = gcal_event_time_override_hhmm
                is_clinic_appt_tomorrow = (start_dt_clinic_utc.astimezone(gcal_event_pytz).date() == tomorrow_gcal_local.date())

                if not gcal_final_summary:
                    gcal_final_summary = f"Appointment: Patient & {clinic_booking_result['doctor_name']}"
                    if not is_clinic_appt_tomorrow:
                        clinic_appt_formatted_for_summary = start_dt_clinic_utc.astimezone(gcal_event_pytz).strftime('%b %d at %I:%M %p %Z')
                        gcal_final_summary = f"REMINDER: {gcal_final_summary} (Actual appt: {clinic_appt_formatted_for_summary})"
                if not final_event_time_str_hhmm:
                    if is_clinic_appt_tomorrow:
                        final_event_time_str_hhmm = start_dt_clinic_utc.astimezone(gcal_event_pytz).strftime('%H:%M')
                    else:
                        final_event_time_str_hhmm = "09:00"
                try:
                    event_hour, event_minute = map(int, final_event_time_str_hhmm.split(':'))
                except ValueError:
                    logger.warning(f"Invalid GCal event time format '{final_event_time_str_hhmm}', using 09:00.")
                    event_hour, event_minute = 9, 0
                gcal_start_dt_local = tomorrow_gcal_local.replace(hour=event_hour, minute=event_minute, second=0, microsecond=0)
                gcal_duration_hours = duration_minutes / 60.0
                gcal_end_dt_local = gcal_start_dt_local + timedelta(hours=gcal_duration_hours)
                gcal_start_rfc3339 = gcal_start_dt_local.isoformat()
                gcal_end_rfc3339 = gcal_end_dt_local.isoformat()
                gcal_final_description = gcal_description_override or clinic_booking_result.get("notes") or gcal_final_summary
                event_body = {
                    'summary': gcal_final_summary,
                    'description': gcal_final_description,
                    'start': {'dateTime': gcal_start_rfc3339, 'timeZone': event_tz_str},
                    'end': {'dateTime': gcal_end_rfc3339, 'timeZone': event_tz_str},
                    'attendees': [{'email': doctor_email_address}],
                    'reminders': {'useDefault': False, 'overrides': [{'method': 'popup', 'minutes': 30}]},
                }

                def _sync_gcal_insert():
                    return gcal_service.events().insert(calendarId='primary', body=event_body, sendNotifications=True).execute()

                created_event = await asyncio.to_thread(_sync_gcal_insert)
                gcal_event_id_from_api = created_event.get('id') # <<<< GET THE GCAL EVENT ID

                if gcal_event_id_from_api:
                    google_calendar_status = f"Google Calendar invite sent to {doctor_email_address} for {gcal_final_summary}. Link: {created_event.get('htmlLink')}"
                    logger.info(google_calendar_status)

                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # ++++ THIS IS THE BLOCK YOU NEED TO ADD/ENSURE IS CORRECT +++++++++++++
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # Now, update the database record with this gcal_event_id_from_api
                    async with tool_db_session() as db_for_gcal_update:
                        gcal_id_updated_in_db = await update_appointment_gcal_id(
                            db_for_gcal_update,
                            db_appointment_object.id, # Use the ID from the appointment object created earlier
                            gcal_event_id_from_api
                        )
                        if gcal_id_updated_in_db:
                            logger.info(f"Successfully stored GCal event ID {gcal_event_id_from_api} for DB appointment {db_appointment_object.id}")
                        else:
                            logger.warning(f"Failed to store GCal event ID {gcal_event_id_from_api} for DB appointment {db_appointment_object.id}")
                            google_calendar_status += " (Note: GCal ID failed to save to DB)."
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                else: # if no gcal_event_id_from_api
                    google_calendar_status = f"Google Calendar invite sent to {doctor_email_address} for {gcal_final_summary}, but no event ID was returned by Google."
                    logger.warning(google_calendar_status)

            except HttpError as api_error:
                # ... (your existing HttpError handling, looks correct)
                google_calendar_status = f"Google Calendar API error: {api_error.resp.status if api_error.resp else 'Unknown'} - Failed to create event."
                error_content = api_error.content.decode('utf-8') if hasattr(api_error, 'content') and isinstance(api_error.content, bytes) else str(api_error.content)
                logger.error(f"{google_calendar_status} Details: {error_content}", exc_info=True)
            except Exception as e_gcal:
                # ... (your existing general GCal Exception handling, looks correct)
                google_calendar_status = f"Unexpected error during Google Calendar scheduling: {type(e_gcal).__name__} - {e_gcal}"
                logger.error(google_calendar_status, exc_info=True)
    else: # Reasons for not attempting GCal invite
        if clinic_booking_result.get("status") != "confirmed":
            logger.info("Skipping Google Calendar invite because clinic booking was not successful.")
        elif not send_google_calendar_invite:
            logger.info("Skipping Google Calendar invite because send_google_calendar_invite is false.")
        elif not db_appointment_object:
            logger.info("Skipping Google Calendar invite because db_appointment_object is None (unexpected after confirmed booking).")


    # --- 4. Consolidate and Return ---
    final_result = {**clinic_booking_result}
    final_result["google_calendar_invite_status"] = google_calendar_status
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
    logger.info(f"Tool 'cancel_appointment' called by user {patient_id} for appointment_id={appointment_id}")
    if not patient_id: # Should be caught by auth middleware, but good check
        logger.warning("Cancel tool called without patient_id (should be injected).")
        return {"status": "error", "message": "I couldn't identify you – please log in again."}

    gcal_event_id_to_delete: Optional[str] = None
    gcal_cancellation_status_msg: str = "Google Calendar event not applicable or not processed."

    try:
        async with tool_db_session() as db:
            # 1. Verify appointment existence and ownership, and get GCal ID
            appointment_to_cancel: Optional[AppointmentModel] = None
            try:
                # get_appointment CRUD should return the AppointmentModel which includes google_calendar_event_id
                appointment_to_cancel = await get_appointment(db, appointment_id, patient_id, "patient")
                if appointment_to_cancel and hasattr(appointment_to_cancel, 'google_calendar_event_id'):
                    gcal_event_id_to_delete = appointment_to_cancel.google_calendar_event_id
                    logger.info(f"Tool: Found GCal Event ID '{gcal_event_id_to_delete}' for appointment_id={appointment_id} to be cancelled.")
                elif appointment_to_cancel:
                    logger.info(f"Tool: No GCal Event ID found for appointment_id={appointment_id}.")
                # If get_appointment raises HTTPException, it will be caught by the outer try-except
            except HTTPException as http_exc: # Catch specific FastAPI HTTPException from get_appointment
                logger.warning(f"Tool: get_appointment failed for appt_id={appointment_id}, user_id={patient_id}. Detail: {http_exc.detail}")
                return {"status": "error", "message": http_exc.detail} # Relay message from get_appointment
            except Exception as e_get: # Catch other unexpected errors from get_appointment
                logger.error(f"Tool: Unexpected error fetching appointment {appointment_id} for patient {patient_id}: {e_get}", exc_info=True)
                return {"status": "error", "message": "An error occurred while trying to find your appointment."}

            # If appointment_to_cancel is None here, get_appointment raised an error handled above, or it just wasn't found
            if not appointment_to_cancel:
                 # This case should ideally be covered by get_appointment raising HTTPException for not found
                logger.warning(f"Tool: Appointment {appointment_id} not found or not accessible by user {patient_id} after initial check.")
                return {"status": "error", "message": "That appointment doesn't exist or doesn't belong to you."}


            # 2. Delete from Database
            # delete_appointment CRUD performs a hard delete
            logger.debug(f"Tool: Attempting to hard delete appointment_id={appointment_id} from DB for user_id={patient_id}")
            db_deleted_successfully = await delete_appointment(db, appointment_id, patient_id, "patient")

            if db_deleted_successfully:
                logger.info(f"Tool: Successfully deleted appointment_id={appointment_id} from database.")

                # 3. Attempt to Delete from Google Calendar if GCal ID exists
                if gcal_event_id_to_delete:
                    logger.info(f"Tool: Proceeding to delete GCal event_id='{gcal_event_id_to_delete}'.")
                    gcal_success, gcal_msg = await _delete_gcal_event_if_exists_scheduler(gcal_event_id_to_delete)
                    gcal_cancellation_status_msg = gcal_msg # Store the message from the helper
                    if gcal_success:
                        logger.info(f"Tool: GCal processing for event_id='{gcal_event_id_to_delete}' successful.")
                    else:
                        logger.warning(f"Tool: GCal processing for event_id='{gcal_event_id_to_delete}' had issues: {gcal_msg}")
                else:
                    gcal_cancellation_status_msg = "No Google Calendar event was linked to this appointment."
                    logger.info(f"Tool: No GCal event ID to delete for appointment_id={appointment_id}.")

                return {
                    "status": "cancelled",
                    "message": f"Appointment #{appointment_id} has been successfully cancelled from the schedule. {gcal_cancellation_status_msg}"
                }
            else:
                # This case implies delete_appointment returned False, which means the get_appointment check
                # might have passed but the delete itself failed for some reason (e.g., row gone between select and delete - rare).
                logger.warning(f"Tool: Failed to delete appointment_id={appointment_id} from database, though it was initially found.")
                return {"status": "error", "message": "Sorry – there was an issue cancelling that appointment from the database."}

    except Exception as e: # Catch-all for unexpected errors in the tool's own logic
        logger.error(f"Tool 'cancel_appointment': Unexpected error for appointment_id={appointment_id}, user_id={patient_id}: {e}", exc_info=True)
        return {"status": "error", "message": "I encountered an unexpected error while trying to cancel the appointment. Please try again later."}


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
