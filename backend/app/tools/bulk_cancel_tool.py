# backend/app/tools/bulk_cancel_tool.py
import logging
import asyncio
from datetime import datetime, timedelta, date as DateClass # Alias date
from typing import Optional, List
from zoneinfo import ZoneInfo # For robust timezone handling

from langchain_core.tools import tool
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState

from app.db.session import tool_db_session
from app.db.crud.appointment import (
    get_appointments_for_doctor_on_date,
    cancel_appointment_by_id_for_doctor # This CRUD updates status
)
# Make sure this path is correct to import your GCal helper
from app.tools.scheduler.tools import _get_gcal_service_sync, GCAL_DEFAULT_TIMEZONE
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

async def _delete_gcal_event_if_exists(event_id: str, calendar_id: str = 'primary') -> tuple[bool, str]:
    """
    Helper to delete a single GCal event.
    Returns (success_bool, message_str).
    """
    if not event_id:
        return True, "No GCal event ID provided for deletion."

    logger.info(f"GCal Helper: Attempting to get GCal service for deleting event_id: {event_id}")
    # Assuming _get_gcal_service_sync is synchronous
    service, error_msg = await asyncio.to_thread(_get_gcal_service_sync)
    if not service:
        logger.error(f"GCal Helper: Failed to get Google Calendar service: {error_msg}")
        return False, f"Failed to connect to Google Calendar: {error_msg}"

    try:
        logger.info(f"GCal Helper: Attempting to delete Google Calendar event: {event_id} from calendar: {calendar_id}")
        # Run the synchronous Google API call in a separate thread
        await asyncio.to_thread(
            service.events().delete(calendarId=calendar_id, eventId=event_id, sendUpdates='all').execute
        )
        logger.info(f"GCal Helper: Successfully deleted Google Calendar event: {event_id}")
        return True, f"Google Calendar event {event_id} deleted."
    except HttpError as e:
        if e.resp.status == 404: # Event not found
            logger.warning(f"GCal Helper: Google Calendar event {event_id} not found for deletion (404). Assuming already deleted or never existed.")
            return True, f"Google Calendar event {event_id} not found (might be already deleted)."
        logger.error(f"GCal Helper: HttpError deleting Google Calendar event {event_id}: {e.resp.status} - {e.content}", exc_info=True)
        return False, f"Google Calendar API error deleting event {event_id}: {e.resp.status}"
    except Exception as e:
        logger.error(f"GCal Helper: Unexpected error deleting Google Calendar event {event_id}: {e}", exc_info=True)
        return False, f"Unexpected error deleting Google Calendar event {event_id}: {str(e)}"


@tool("cancel_tomorrows_appointments")
async def cancel_tomorrows_appointments(
    doctor_user_id: Annotated[int, InjectedState("user_id")],
    user_tz_str: Annotated[Optional[str], InjectedState("user_tz")]
) -> str:
    """
    Cancels all of the calling doctor's appointments scheduled for "tomorrow".
    "Tomorrow" is determined based on the doctor's current time and timezone.
    This will update the appointment status in the database to 'cancelled_by_doctor'
    and attempt to delete corresponding Google Calendar events if their IDs are stored.
    """
    logger.info(f"Tool 'cancel_tomorrows_appointments' invoked by doctor_id='{doctor_user_id}' with timezone='{user_tz_str}'")

    effective_user_tz_str = user_tz_str or GCAL_DEFAULT_TIMEZONE # Use the imported default if user_tz is None
    try:
        user_timezone = ZoneInfo(effective_user_tz_str)
    except Exception:
        logger.warning(f"Invalid timezone '{effective_user_tz_str}' provided for doctor {doctor_user_id}. Defaulting to {GCAL_DEFAULT_TIMEZONE}.")
        user_timezone = ZoneInfo(GCAL_DEFAULT_TIMEZONE)

    # Determine "tomorrow" based on the doctor's timezone
    current_time_in_user_tz = datetime.now(user_timezone)
    tomorrow_date_in_user_tz = (current_time_in_user_tz + timedelta(days=1)).date()
    logger.info(f"Tool: Targeting appointments for cancellation on date: {tomorrow_date_in_user_tz} (Doctor's tomorrow in {user_timezone})")

    cancelled_db_count = 0
    total_appointments_for_tomorrow = 0
    processed_gcal_event_ids = set() # To avoid trying to delete the same GCal event multiple times if, hypothetically, multiple DB appts linked to it
    successful_gcal_deletions = 0
    failed_gcal_cancellations_details = []
    problematic_db_cancellations = []

    async with tool_db_session() as db:
        try:
            # 1. Get 'scheduled' appointments for tomorrow
            appointments_for_tomorrow = await get_appointments_for_doctor_on_date(
                db, doctor_id=doctor_user_id, target_date=tomorrow_date_in_user_tz
            )
            total_appointments_for_tomorrow = len(appointments_for_tomorrow)

            if not appointments_for_tomorrow:
                return f"No appointments found scheduled for you for tomorrow ({tomorrow_date_in_user_tz.strftime('%Y-%m-%d')})."

            logger.info(f"Tool: Found {total_appointments_for_tomorrow} 'scheduled' appointments for doctor {doctor_user_id} on {tomorrow_date_in_user_tz}.")

            # 2. Cancel each appointment
            for appt in appointments_for_tomorrow:
                # a. Cancel in Database (soft delete by updating status)
                db_cancelled_successfully = await cancel_appointment_by_id_for_doctor(db, appt.id, doctor_user_id)

                if db_cancelled_successfully:
                    cancelled_db_count += 1
                    logger.info(f"Tool: DB status for appointment_id={appt.id} updated to 'cancelled_by_doctor'.")

                    # b. Attempt to Cancel in Google Calendar if event ID exists and is valid
                    if hasattr(appt, 'google_calendar_event_id') and appt.google_calendar_event_id and appt.google_calendar_event_id not in processed_gcal_event_ids:
                        gcal_event_id = appt.google_calendar_event_id
                        logger.info(f"Tool: GCal - Attempting to cancel event_id={gcal_event_id} for appointment_id={appt.id}")
                        processed_gcal_event_ids.add(gcal_event_id) # Mark as processed
                        gcal_success, gcal_msg = await _delete_gcal_event_if_exists(gcal_event_id)
                        if gcal_success:
                            successful_gcal_deletions += 1
                            logger.info(f"Tool: GCal - Successfully processed GCal event_id={gcal_event_id} (message: {gcal_msg})")
                        else:
                            failed_gcal_cancellations_details.append(f"Appt ID {appt.id} (GCal ID {gcal_event_id}): {gcal_msg}")
                            logger.warning(f"Tool: GCal - Failed to process GCal event_id={gcal_event_id} for appointment_id={appt.id}: {gcal_msg}")
                    elif hasattr(appt, 'google_calendar_event_id') and appt.google_calendar_event_id in processed_gcal_event_ids:
                        logger.info(f"Tool: GCal - Already processed GCal event_id={appt.google_calendar_event_id} for appointment_id={appt.id}. Skipping duplicate deletion attempt.")
                    elif not hasattr(appt, 'google_calendar_event_id') or not appt.google_calendar_event_id:
                        logger.info(f"Tool: GCal - No Google Calendar event ID found for appointment_id={appt.id}. Skipping GCal cancellation for this appointment.")
                else:
                    logger.warning(f"Tool: DB - Failed to update status for appointment_id={appt.id} (may not exist, not belong to doctor, or not in 'scheduled' state).")
                    problematic_db_cancellations.append(f"Appt ID {appt.id}: DB update failed.")

        except Exception as e:
            logger.error(f"Tool 'cancel_tomorrows_appointments': Error during DB operations for doctor_id='{doctor_user_id}': {e}", exc_info=True)
            return "An unexpected error occurred while retrieving or updating appointments in the database. Please check system logs."

    # 3. Formulate response message
    response_parts = []
    if cancelled_db_count > 0:
        response_parts.append(f"Successfully cancelled {cancelled_db_count} out of {total_appointments_for_tomorrow} appointments in the database for tomorrow ({tomorrow_date_in_user_tz.strftime('%Y-%m-%d')}).")
    elif total_appointments_for_tomorrow > 0:
        response_parts.append(f"Found {total_appointments_for_tomorrow} appointments for tomorrow, but none could be cancelled in the database (they might have already been cancelled or there was an issue).")
    else: # This case should be caught earlier, but as a fallback
        response_parts.append(f"No appointments were scheduled for you for tomorrow ({tomorrow_date_in_user_tz.strftime('%Y-%m-%d')}) to cancel.")

    if successful_gcal_deletions > 0:
        response_parts.append(f"Successfully processed (deleted or confirmed not found) {successful_gcal_deletions} Google Calendar events.")
    if failed_gcal_cancellations_details:
        response_parts.append(f"Could not successfully process the following Google Calendar events:")
        for failure_detail in failed_gcal_cancellations_details:
            response_parts.append(f"- {failure_detail}")
    if problematic_db_cancellations:
        response_parts.append(f"Issues encountered with database cancellations:")
        for db_issue in problematic_db_cancellations:
            response_parts.append(f"- {db_issue}")
    
    return "\n".join(response_parts)