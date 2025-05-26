# backend/app/tools/bulk_cancel_tool.py
import logging
import asyncio
from datetime import datetime, timedelta, date as DateClass # Alias date
from typing import Optional, List
from zoneinfo import ZoneInfo

from langchain_core.tools import tool
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState

from app.db.session import tool_db_session
from app.db.crud.appointment import (
    get_appointments_for_doctor_on_date, # This function is perfect for this
    delete_appointment
)
# Ensure this path is correct for your project structure
from app.tools.scheduler.tools import _get_gcal_service_sync, GCAL_DEFAULT_TIMEZONE
from googleapiclient.errors import HttpError
import dateparser # For parsing natural language dates

logger = logging.getLogger(__name__)

# _delete_gcal_event_if_exists helper function (as defined before, ensure it's in this file or imported)
async def _delete_gcal_event_if_exists(event_id: str, calendar_id: str = 'primary') -> tuple[bool, str]:
    if not event_id:
        return True, "No GCal event ID provided for deletion."
    logger.info(f"GCal Helper: Attempting to get GCal service for deleting event_id: {event_id}")
    service, error_msg = await asyncio.to_thread(_get_gcal_service_sync)
    if not service:
        logger.error(f"GCal Helper: Failed to get Google Calendar service: {error_msg}")
        return False, f"Failed to connect to Google Calendar: {error_msg}"
    try:
        logger.info(f"GCal Helper: Attempting to delete Google Calendar event: {event_id} from calendar: {calendar_id}")
        await asyncio.to_thread(
            service.events().delete(calendarId=calendar_id, eventId=event_id, sendUpdates='all').execute
        )
        logger.info(f"GCal Helper: Successfully deleted Google Calendar event: {event_id}")
        return True, f"Google Calendar event {event_id} deleted."
    except HttpError as e:
        if e.resp.status == 404:
            logger.warning(f"GCal Helper: Google Calendar event {event_id} not found for deletion (404).")
            return True, f"Google Calendar event {event_id} not found (might be already deleted)."
        logger.error(f"GCal Helper: HttpError deleting Google Calendar event {event_id}: {e.resp.status} - {e.content}", exc_info=True)
        return False, f"Google Calendar API error deleting event {event_id}: {e.resp.status}"
    except Exception as e:
        logger.error(f"GCal Helper: Unexpected error deleting Google Calendar event {event_id}: {e}", exc_info=True)
        return False, f"Unexpected error deleting Google Calendar event {event_id}: {str(e)}"


@tool("cancel_doctor_appointments_for_date") # <<<< RENAMED TOOL
async def cancel_doctor_appointments_for_date( # <<<< RENAMED FUNCTION
    date_query: Annotated[str, "The date for which to cancel all of the doctor's appointments (e.g., 'today', 'tomorrow', 'July 10th', '2025-07-10'). This tool should ONLY be called AFTER the doctor has EXPLICITLY CONFIRMED the cancellation for this specific date in a previous conversational turn."],
    doctor_user_id: Annotated[int, InjectedState("user_id")],
    user_tz_str: Annotated[Optional[str], InjectedState("user_tz")]
) -> str:
    """
    Cancels (hard deletes) ALL of the calling doctor's 'scheduled' appointments for a SPECIFIED date.
    The date is parsed from the date_query based on the doctor's current time and timezone.
    IMPORTANT: The AI assistant MUST have ALREADY VERBALLY CONFIRMED with the doctor for the *specific target date*
    (e.g., "You have X appointments on {date}, are you sure you want to cancel all of them?")
    and received a 'yes' BEFORE calling this tool.

    Args:
        date_query: The date string (e.g., "today", "tomorrow", "YYYY-MM-DD") for which appointments are to be cancelled.
        doctor_user_id: The ID of the doctor whose appointments are to be cancelled.
        user_tz_str: The timezone of the user for correct date parsing.
    """
    logger.info(
        f"Tool 'cancel_doctor_appointments_for_date' invoked by doctor_id='{doctor_user_id}' "
        f"for date_query: '{date_query}', user_tz: '{user_tz_str}'"
    )

    # 1. Parse the date_query
    effective_user_tz_str = user_tz_str or GCAL_DEFAULT_TIMEZONE
    try:
        effective_user_tz = ZoneInfo(effective_user_tz_str)
    except Exception:
        logger.warning(f"Invalid user_tz '{user_tz_str}', defaulting to {GCAL_DEFAULT_TIMEZONE} for date parsing.")
        effective_user_tz = ZoneInfo(GCAL_DEFAULT_TIMEZONE)
        effective_user_tz_str = GCAL_DEFAULT_TIMEZONE # Update string if defaulted

    now_in_user_tz = datetime.now(effective_user_tz)
    parsed_datetime_obj = dateparser.parse(
        date_query,
        settings={
            "PREFER_DATES_FROM": "current_period", # Helps with "today", "tomorrow"
            "RELATIVE_BASE": now_in_user_tz,     # Crucial for relative dates
            "TIMEZONE": effective_user_tz_str,       # Tells dateparser the input's context if ambiguous
        }
    )

    if not parsed_datetime_obj:
        logger.warning(f"Tool: Could not parse date_query: '{date_query}' for doctor_id '{doctor_user_id}'.")
        return f"Sorry, I could not understand the date '{date_query}' for cancellation. Please specify a clear date like 'tomorrow', 'next Monday', or 'July 10th 2025'."

    target_date_obj: DateClass = parsed_datetime_obj.date() # Get the date part
    formatted_target_date_str = target_date_obj.strftime('%Y-%m-%d (%A, %B %d)')
    logger.info(
        f"Tool: Executing confirmed cancellation of all 'scheduled' appointments for doctor_id {doctor_user_id} on calculated date: {formatted_target_date_str}"
    )

    # --- Initialize counters and result lists ---
    deleted_db_count = 0
    total_appointments_on_date = 0
    processed_gcal_event_ids = set()
    successful_gcal_deletions = 0
    failed_gcal_deletions_details = []
    problematic_db_deletions_details = []

    async with tool_db_session() as db:
        try:
            # 2. Fetch 'scheduled' appointments for this doctor on this date
            appointments_on_date = await get_appointments_for_doctor_on_date(
                db, doctor_id=doctor_user_id, target_date=target_date_obj
            )
            total_appointments_on_date = len(appointments_on_date)

            if not appointments_on_date:
                return f"No 'scheduled' appointments were found for you on {formatted_target_date_str} to cancel."

            logger.info(
                f"Tool: Found {total_appointments_on_date} 'scheduled' appointments to cancel for doctor {doctor_user_id} on {target_date_obj}."
            )

            # 3. Proceed with cancellation for each appointment
            for appt in appointments_on_date:
                gcal_event_id_for_this_appt = getattr(appt, 'google_calendar_event_id', None)

                # a. Hard Delete from Database
                db_deleted_successfully = await delete_appointment(
                    db,
                    appointment_id=appt.id,
                    user_id=doctor_user_id,
                    role="doctor"
                )

                if db_deleted_successfully:
                    deleted_db_count += 1
                    logger.info(f"Tool: DB - Successfully hard deleted appointment_id={appt.id}.")

                    # b. Attempt to Delete from Google Calendar
                    if gcal_event_id_for_this_appt and gcal_event_id_for_this_appt not in processed_gcal_event_ids:
                        logger.info(f"Tool: GCal - Attempting to delete event_id={gcal_event_id_for_this_appt} for (now deleted) appointment_id={appt.id}")
                        processed_gcal_event_ids.add(gcal_event_id_for_this_appt)
                        gcal_success, gcal_msg = await _delete_gcal_event_if_exists(gcal_event_id_for_this_appt)
                        if gcal_success:
                            successful_gcal_deletions += 1
                        else:
                            failed_gcal_deletions_details.append(f"For original Appt ID {appt.id} (GCal ID {gcal_event_id_for_this_appt}): {gcal_msg}")
                    # ... (logging for already processed or no GCal ID) ...
                else:
                    logger.warning(f"Tool: DB - Failed to hard delete appointment_id={appt.id}.")
                    problematic_db_deletions_details.append(f"Appt ID {appt.id}: DB delete failed or not authorized.")
        
        except Exception as e:
            logger.error(
                f"Tool 'cancel_doctor_appointments_for_date': General error for doctor_id '{doctor_user_id}', date '{target_date_obj}': {e}",
                exc_info=True,
            )
            return "An unexpected error occurred while trying to cancel your appointments. Please check system logs."

    # 4. Formulate response message
    response_parts = []
    if total_appointments_on_date == 0 and deleted_db_count == 0:
        response_parts.append(f"No 'scheduled' appointments were found for you for {formatted_target_date_str} to cancel.")
    elif deleted_db_count > 0:
        response_parts.append(f"Successfully deleted {deleted_db_count} out of {total_appointments_on_date} 'scheduled' appointments from the database for {formatted_target_date_str}.")
    elif total_appointments_on_date > 0:
        response_parts.append(f"Found {total_appointments_on_date} 'scheduled' appointments for {formatted_target_date_str}, but none could be deleted from the database.")

    if successful_gcal_deletions > 0:
        response_parts.append(f"Successfully processed {successful_gcal_deletions} associated Google Calendar events.")
    if failed_gcal_deletions_details:
        response_parts.append(f"Could not successfully process the following Google Calendar events:")
        for failure_detail in failed_gcal_deletions_details:
            response_parts.append(f"- {failure_detail}")
    if problematic_db_deletions_details:
        response_parts.append(f"Issues encountered with database deletions:")
        for db_issue in problematic_db_deletions_details:
            response_parts.append(f"- {db_issue}")
    
    if not response_parts:
        return f"No action taken for appointments on {formatted_target_date_str}."
        
    return "\n".join(response_parts)