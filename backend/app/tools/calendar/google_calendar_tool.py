# backend/app/tools/calendar/google_calendar_tool.py

import datetime
import os.path
import pytz
import asyncio  # For running blocking IO in a thread
from pathlib import Path
from typing import Optional, Dict, Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState  # If you need to inject state like user_tz

# Google API Client libraries
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import logging

logger = logging.getLogger(__name__)

# --- Configuration for the tool ---
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
# TOKEN_FILE_NAME should be relative to this file or an absolute path
# For deployment, consider making TOKEN_FILE_PATH configurable via environment variable
TOKEN_FILE_PATH = Path(__file__).parent / "token.json"
DEFAULT_TIMEZONE_TOOL = "Asia/Beirut"  # Default if not provided by agent/user


# --- Pydantic Input Schema for the Tool ---
class ScheduleGoogleCalendarInput(BaseModel):
    attendee_email: str = Field(
        ...,
        description="Email address of the person to invite to the Google Calendar event.",
    )
    summary: str = Field(
        ...,
        description="The title or summary of the Google Calendar event (e.g., 'Meeting with Dr. Smith').",
    )
    event_time_str: str = Field(
        description="The time for the appointment for TOMORROW, in HH:MM format (e.g., '14:30' for 2:30 PM)."
    )
    duration_hours: float = Field(
        default=1.0,
        description="Duration of the appointment in hours (e.g., 0.5 for 30 minutes, 1 for 1 hour).",
    )
    description: Optional[str] = Field(
        None, description="Optional detailed description or message for the event body."
    )
    # timezone_str is optional here; if not provided by LLM, we'll try to use injected user_tz or a default
    timezone_str: Optional[str] = Field(
        None,
        description="Optional: IANA timezone name for the event (e.g., 'America/New_York', 'Asia/Beirut'). If omitted, the user's configured timezone or a system default will be used.",
    )
    send_notifications: bool = Field(
        default=True, description="Whether to send email notifications to attendees."
    )
    calendar_id: str = Field(
        default="primary",
        description="The calendar ID to add the event to. Defaults to 'primary'.",
    )


# --- Helper function to get Calendar Service (modified from your script) ---
def _get_calendar_service_sync():
    """
    Synchronous version for use with asyncio.to_thread.
    Authenticates and returns Google Calendar service object using token.json.
    """
    creds = None
    if not TOKEN_FILE_PATH.exists():
        logger.error(f"Google Calendar: {TOKEN_FILE_PATH} not found.")
        return (
            None,
            f"Configuration error: Google Calendar token file not found at {TOKEN_FILE_PATH}.",
        )

    try:
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE_PATH), SCOPES)
    except Exception as e:
        logger.error(
            f"Google Calendar: Error loading credentials from {TOKEN_FILE_PATH}: {e}"
        )
        return (
            None,
            f"Error loading Google Calendar credentials: {e}. Ensure the token file is valid.",
        )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.warning(
                f"Google Calendar: Credentials in {TOKEN_FILE_PATH} are expired. Attempting refresh (might fail without client secrets)."
            )
            # The library might attempt to refresh if it can, but we are not explicitly calling refresh here
            # as the original script intended to bypass client_secret.json.
            # If refresh fails, the API call will likely fail.
        else:
            logger.error(
                f"Google Calendar: Could not load valid credentials from {TOKEN_FILE_PATH}."
            )
            return (
                None,
                "Invalid or missing Google Calendar credentials. Token may be expired or corrupted.",
            )

    try:
        service = build("calendar", "v3", credentials=creds)
        logger.info(
            "Google Calendar service object created successfully using token.json."
        )
        return service, None
    except HttpError as error:
        logger.error(
            f"Google Calendar: An API error occurred building the service: {error}"
        )
        if error.resp.status == 401:
            return (
                None,
                "Google Calendar authentication error (401): The token is likely invalid or expired, and refresh failed or was not possible.",
            )
        return (
            None,
            f"API error building Google Calendar service: {error.resp.status} - {error.content}",
        )
    except Exception as e:
        logger.error(
            f"Google Calendar: An unexpected error occurred during service build: {e}"
        )
        return None, f"Unexpected error building Google Calendar service: {e}"


# --- The Langchain Tool ---
@tool("schedule_google_calendar_event", args_schema=ScheduleGoogleCalendarInput)
async def schedule_google_calendar_event(
    attendee_email: str,
    summary: str,
    event_time_str: str,
    duration_hours: float = 1.0,
    description: Optional[str] = None,
    timezone_str: Annotated[
        Optional[str], InjectedState("user_tz")
    ] = None,  # Injects user_tz from agent state
    send_notifications: bool = True,
    calendar_id: str = "primary",
) -> str:
    """
    Schedules an event on Google Calendar for TOMORROW with the specified attendee.
    This tool uses a pre-existing authorization token (token.json) and cannot
    perform initial authorization or refresh tokens that require client secrets.
    Ensure token.json is valid and present in the server's configured path.
    The event time is specified in HH:MM format for tomorrow.
    The timezone_str, if provided by the LLM call, will be used; otherwise, it defaults to the user's timezone from the chat session or a system default.
    """
    logger.info(
        f"Tool 'schedule_google_calendar_event' called for attendee: {attendee_email}, "
        f"summary: '{summary}', time: {event_time_str} for TOMORROW."
    )

    # Determine timezone: LLM-provided > Injected user_tz > Tool's default
    final_timezone_str = timezone_str  # This comes from Pydantic model (LLM input or InjectedState if field name matches)
    if (
        final_timezone_str is None
    ):  # Check if it was actually provided or resolved by InjectedState
        # If InjectedState value was None, it would be None here.
        # This logic might be redundant if InjectedState provides a value or None.
        # The key is that 'timezone_str' in the function signature is what gets the InjectedState.
        # Let's assume if LLM provides it via the input schema, it overrides the injected one.
        # This needs careful testing with how LangGraph injects.
        # For simplicity now: if the `timezone_str` argument (which might be from LLM or InjectedState) is None, use default.
        final_timezone_str = DEFAULT_TIMEZONE_TOOL
        logger.info(
            f"No specific timezone provided or injected, using default: {final_timezone_str}"
        )
    else:
        logger.info(f"Using timezone: {final_timezone_str}")

    # Run the synchronous Google API calls in a separate thread
    # to avoid blocking the asyncio event loop.
    def _sync_book_appointment():
        service, error_msg = _get_calendar_service_sync()
        if not service:
            return f"Failed to get Calendar service: {error_msg}"

        try:
            tz = pytz.timezone(final_timezone_str)
            now_local = datetime.datetime.now(tz)
            tomorrow_local = now_local + datetime.timedelta(days=1)

            try:
                event_hour, event_minute = map(int, event_time_str.split(":"))
            except ValueError:
                logger.warning(
                    f"Invalid event_time_str format: '{event_time_str}'. Using 09:00."
                )
                event_hour, event_minute = 9, 0

            start_datetime_local = tomorrow_local.replace(
                hour=event_hour, minute=event_minute, second=0, microsecond=0
            )
            end_datetime_local = start_datetime_local + datetime.timedelta(
                hours=duration_hours
            )

            start_time_rfc3339 = start_datetime_local.isoformat()
            end_time_rfc3339 = end_datetime_local.isoformat()

            logger.info(
                f"Attempting to book Google Calendar event for: {start_datetime_local.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"
            )

            event_body = {
                "summary": summary,
                "description": description if description else summary,
                "start": {
                    "dateTime": start_time_rfc3339,
                    "timeZone": final_timezone_str,
                },
                "end": {"dateTime": end_time_rfc3339, "timeZone": final_timezone_str},
                "attendees": [{"email": attendee_email}],
                "reminders": {
                    "useDefault": False,
                    "overrides": [
                        {"method": "email", "minutes": 24 * 60},
                        {"method": "popup", "minutes": 30},
                    ],
                },
            }

            created_event = (
                service.events()
                .insert(
                    calendarId=calendar_id,
                    body=event_body,
                    sendNotifications=send_notifications,
                )
                .execute()
            )

            event_link = created_event.get("htmlLink")
            confirmation_msg = f"Successfully scheduled Google Calendar event: '{created_event.get('summary')}' for tomorrow. Link: {event_link}"
            logger.info(confirmation_msg)
            return confirmation_msg

        except HttpError as api_error:
            logger.error(
                f"Google Calendar API error: {api_error}. Details: {api_error.content}"
            )
            if api_error.resp.status == 401:
                return (
                    "Google Calendar authentication error: The token is likely invalid or expired. "
                    "Please ensure token.json is up-to-date."
                )
            return f"Google Calendar API error: {api_error.resp.status} - Failed to create event."
        except pytz.UnknownTimeZoneError:
            logger.error(f"Error: Unknown timezone '{final_timezone_str}'.")
            return f"Error: Unknown timezone '{final_timezone_str}'. Please use a valid IANA timezone name."
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during Google Calendar booking: {e}",
                exc_info=True,
            )
            return f"An unexpected error occurred: {e}"

    # Execute the synchronous booking logic in a thread
    return await asyncio.to_thread(_sync_book_appointment)


# --- Example of how you might test it standalone (requires token.json) ---
async def main_test():
    print("--- Google Calendar Booking Tool Test (Token Only) ---")
    # Ensure token.json exists in the same directory as this script (or where TOKEN_FILE_PATH points)

    invite_email = "test@gmail.com"  # *** CHANGE THIS ***
    if invite_email == "your_test_email@example.com":
        print("Please change 'your_test_email@example.com' before running the test.")
        return

    # Simulate InjectedState for testing (if your agent provides it)
    # If you don't inject user_tz, timezone_str in the call below will be used or the default.
    class MockState:  # For testing InjectedState
        user_tz: Optional[str] = "America/Los_Angeles"  # Or None to test default

    # To test InjectedState, you'd typically call this from within a graph context
    # For standalone, we pass timezone_str directly or let it default.
    # Here, we'll test passing it directly to the tool to simulate LLM input.
    result = await schedule_google_calendar_event.ainvoke(
        {
            "attendee_email": invite_email,
            "summary": "Async Tool Test Event",
            "description": "Testing the async Langchain tool for Google Calendar.",
            "event_time_str": "11:45",
            "duration_hours": 0.75,
            "timezone_str": "Asia/Beirut",  # LLM provides this
            # If you were testing InjectedState and this was None, it would use MockState.user_tz
        }
        # If you want to test the injected user_tz, you would set timezone_str=None in the dict
        # and ensure `InjectedState("user_tz")` could resolve (which is hard standalone).
        # The current tool logic prioritizes timezone_str from the input dict if present.
    )
    print(f"\nTool invocation result:\n{result}")

    print("\n--- Test finished ---")


if __name__ == "__main__":
    # Setup basic logging for standalone test
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # asyncio.run(main_test())
    print(
        "To run the test, uncomment `asyncio.run(main_test())` and configure `invite_email` and `token.json`."
    )
    print(f"Tool expects token.json at: {TOKEN_FILE_PATH.resolve()}")
