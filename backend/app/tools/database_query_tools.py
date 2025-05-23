import logging
from typing import Optional, Annotated, List, Dict, Any, Union
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import interrupt

from app.db.session import tool_db_session
from app.db.crud.patient import (
    find_patients_by_name_and_verify_doctor_link,
    get_patients_for_doctor,
)
from app.db.crud.appointment import (
    get_appointments,
    get_doctor_schedule_for_date,
    delete_appointment,
    get_appointment,
)
from app.db.models import PatientModel

from app.db.crud.allergy import get_allergies_for_patient

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud.salary import get_doctor_financial_summary_by_user_id

from datetime import (
    datetime,
    timedelta,
    date as DateClass,
    timezone as TZ,
)  # Alias date
from zoneinfo import ZoneInfo  # Preferred for IANA timezones
import dateparser
import decimal

logger = logging.getLogger(__name__)


@tool("get_patient_info")
async def get_patient_info(
    patient_full_name: str, user_id: Annotated[int, InjectedState("user_id")]
) -> str:
    """
    Fetches basic demographic information (Date of Birth, sex, phone number, address)
    for a specific patient if they have an appointment record with the requesting doctor
    The patient_full_name should be the first and last name of the patient

    """
    logger.info(
        f"Tool 'get_patient_info' invoked by doctor_id '{user_id}' for patient patient: '{patient_full_name}'"
    )

    if not patient_full_name or not patient_full_name.strip():
        return "Please provide the full name of the patient you are looking for"

    async with tool_db_session() as db:
        try:
            # user_id here is the requesting_doctor_id from the agent's state
            patients = await find_patients_by_name_and_verify_doctor_link(
                db, full_name=patient_full_name, requesting_doctor_id=user_id
            )

            if not patients:
                return f"'No patient named '{patient_full_name}' found with an appointment record associated with you"

            if len(patients) > 1:
                # If multiple patients with the same name are linked to this doctor,
                # provide enough info for the doctor (via LLM) to disambiguate.
                response_lines = [
                    f"Multiple patients named '{patient_full_name}' found who have had appointments with you. Please specify using their date of birth: "
                ]
                for p in patients:
                    dob_str = (
                        p.dob.strftime("%Y-%m-%d") if p.dob else "DOB not available"
                    )
                    response_lines.append(
                        f"- {p.first_name} {p.last_name} (DOB: {dob_str})"
                    )

                return "\n".join(response_lines)

            # Exact;y onr patient found
            patient = patients[0]
            dob_str = patient.dob.strftime("%Y-%m-%d") if patient.dob else "N/A"
            sex_str = patient.sex or "N/A"
            phone_str = patient.phone or "N/A"
            address_str = patient.address or "N/A"

            return (
                f"Patient Information for {patient.first_name} {patient.last_name}:\n"
                f"- Date of Birth: {dob_str}\n"
                f"- Sex: {sex_str}\n"
                f"- Phone: {phone_str}\n"
                f"- Address: {address_str}"
            )

        except Exception as e:
            logger.error(
                f"Tool: 'get_patient_info': Error processing request for doctor_id '{user_id}', patient '{patient_full_name}': {e}",
                exc_info=True,
            )
            return "An unexpected error occurred while trying to retrieve patient information. Please try again later."


@tool("list_my_patients")
async def list_my_patients(
    user_id: Annotated[int, InjectedState("user_id")],
    page: Optional[int] = 1,
    page_size: Optional[int] = 10,
) -> str:
    """
    Lists all patients who have an appointment record with the currently logged-in doctor.
    supports pagination.

    Args:
        user_id (Annotated[int, InjectedState): id of the logged-in doctor
        page (Optional[int], optional): the page number to retrieve starting from 1, Defaults to 1
        page_size (Optional[int], optional): the number of patients to retrieve per page, Defaults to 10
    """

    logger.info(
        f"Tool 'list_my_patients' invoked by doctor_id '{user_id}' with page {page}, page_size {page_size}"
    )

    current_page = page if page and page > 0 else 1
    current_page_size = page_size if page_size and page_size > 0 else 10
    offset = (current_page - 1) * current_page_size

    async with tool_db_session() as db:
        try:
            # user_id here is the requesting_doctor_id
            patients = await get_patients_for_doctor(
                db, requesting_doctor_id=user_id, limit=current_page_size, offset=offset
            )

            if not patients:
                if current_page == 1:
                    return "You dont have any patients with appointmnent records in the system"
                else:
                    return "No more patients found for the given page"

            response_lines = [f"Listing your patients (Page {current_page}):"]
            for p_idx, patient in enumerate(patients):
                dob_str = patient.dob.strftime("%Y-%m-%d") if patient.dob else "N/A"
                # Using patient.user_id  as an identifier in the list for now
                response_lines.append(
                    f"{offset + p_idx + 1}. {patient.first_name} {patient.last_name} (ID: {patient.user_id}), DOB: {dob_str}"
                )

            if len(patients) < current_page_size:
                response_lines.append("\n(End of list)")
            else:
                response_lines.append(
                    f"\n (Showing {len(patients)} patients. To see more, ask for page {current_page + 1})"
                )

            return "\n".join(response_lines)
        except Exception as e:
            logger.error(
                f"Tool: 'list_my_patients: Error processing request for doctor_id '{user_id}': {e}",
                exc_info=True,
            )
            return "An unexpected error occurred while trying to retrieve your patient list. Please try again later."


@tool("get_patient_allergies_info")
async def get_patient_allergies_info(
    patient_full_name: str, user_id: Annotated[int, InjectedState("user_id")]
) -> str:
    """
    Fetches recorded allergies for a specific patient if they have an appointment
    record with the requesting doctor.
    The patient_full_name should be the first and last name of the patient.
    """
    logger.info(
        f"Tool 'get_patient_allergies_info' invoked by doctor_id '{user_id}' for patient: '{patient_full_name}'"
    )

    if not patient_full_name or not patient_full_name.strip():
        return "Please provide the full name of the patient you are looking for"

    async with tool_db_session() as db:
        try:
            patients = await find_patients_by_name_and_verify_doctor_link(
                db, full_name=patient_full_name, requesting_doctor_id=user_id
            )

            if not patients:
                return f"No patients named {patient_full_name} found with an appointment record associated with you"

            if len(patients) > 1:
                response_lines = [
                    f"Multiple patients named '{patient_full_name}' found who have had appointments with you. Please specify using their date of birth"
                ]

                for p in patients:
                    dob_str = (
                        p.dob.strftime("%Y-%m-%d") if p.dob else "DOB not available"
                    )
                    response_lines.append(
                        f"- {p.first_name} {p.last_name} (DOB: {dob_str})"
                    )
                return "\n".join(response_lines)

            patient = patients[0]

            allergies = await get_allergies_for_patient(
                db, patient_user_id=patient.user_id
            )

            if not allergies:
                return f"No known allergies recorded for {patient.first_name} {patient.last_name}"

            response_lines = [
                f"Recorded allergies for patient {patient.first_name} {patient.last_name}"
            ]
            for allergy in allergies:
                substance = allergy.substance or "N/A"
                reaction = allergy.reaction or "N/A"
                severity = allergy.severity or "N/A"
                response_lines.append(
                    f"- Allergy to {substance} (Reaction: {reaction}, Severity: {severity})"
                )

            return "\n".join(response_lines)

        except Exception as e:
            logger.error(
                f"Tool: 'get_patient_allergies_info': Error for doctor_id '{user_id}', patient '{patient_full_name}': {e}",
                exc_info=True,
            )
            return "An unexpected error occurred while trying to retrieve patient allergies. Please try again later."


@tool("get_patient_appointment_history")
async def get_patient_appointment_history(
    patient_full_name: str,
    user_id: Annotated[int, InjectedState("user_id")],
    user_tz: Annotated[
        Optional[str], InjectedState("user_tz")
    ],  # Get user_tz from state
    date_filter: Optional[
        str
    ] = None,  # e.g., "upcoming", "past_7_days", "past_30_days", "all"
    specific_date_str: Optional[str] = None,  # e.g., "today", "tomorrow", "2024-07-15"
    limit: Optional[int] = 10,  # Default limit for appointments listed
) -> str:
    """
    Fetches appointment history for a specific patient linked to the requesting doctor.
    Can filter by general periods (upcoming, past_7_days, all) or a specific date.
    """
    logger.info(
        f"Tool 'get_patient_appointment_history' invoked by doctor_id '{user_id}' for patient '{patient_full_name}', "
        f"date_filter='{date_filter}', specific_date_str='{specific_date_str}', user_tz='{user_tz}'"
    )

    if not patient_full_name or not patient_full_name.strip():
        return "Please provide the full name of the patient."

    async with tool_db_session() as db:
        try:
            patients = await find_patients_by_name_and_verify_doctor_link(
                db, full_name=patient_full_name, requesting_doctor_id=user_id
            )

            if not patients:
                return f"No patient named '{patient_full_name}' found with an appointment record associated with you."
            if len(patients) > 1:
                # ... (ambiguity handling as in get_patient_info) ...
                response_lines = [
                    f"Multiple patients named '{patient_full_name}' found who have had appointments with you. Please specify using their date of birth:"
                ]
                for p in patients:
                    dob_str = (
                        p.dob.strftime("%Y-%m-%d") if p.dob else "DOB not available"
                    )
                    response_lines.append(
                        f"- {p.first_name} {p.last_name} (DOB: {dob_str})"
                    )
                return "\n".join(response_lines)

            patient = patients[0]

            # --- Date Logic ---
            now_utc = datetime.now(TZ.utc)
            effective_user_tz_str = user_tz or "UTC"  # Default to UTC if not provided
            try:
                effective_user_tz = ZoneInfo(effective_user_tz_str)
            except Exception:
                logger.warning(f"Invalid user_tz '{user_tz}', defaulting to UTC.")
                effective_user_tz = ZoneInfo("UTC")

            now_user_tz = datetime.now(effective_user_tz)

            date_from_dt_utc: Optional[datetime] = None
            date_to_dt_utc: Optional[datetime] = None
            filter_description = "all"  # Default description

            if specific_date_str:
                # Use dateparser for natural language dates
                parsed_date = dateparser.parse(
                    specific_date_str,
                    settings={
                        "TIMEZONE": effective_user_tz_str,
                        "TO_TIMEZONE": "UTC",  # Ask dateparser to convert to UTC
                        "RETURN_AS_TIMEZONE_AWARE": True,
                        "PREFER_DATES_FROM": "current_period",
                        "RELATIVE_BASE": now_user_tz,  # Parse relative to user's current time
                    },
                )
                if parsed_date:
                    # For a specific date, get the whole day in UTC
                    # Ensure parsed_date is timezone-aware; if not, assume it's in user_tz
                    if parsed_date.tzinfo is None:
                        parsed_date = effective_user_tz.localize(
                            parsed_date
                        )  # Localize if naive

                    start_of_day_user = parsed_date.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    end_of_day_user = parsed_date.replace(
                        hour=23, minute=59, second=59, microsecond=999999
                    )
                    date_from_dt_utc = start_of_day_user.astimezone(TZ.utc)
                    date_to_dt_utc = end_of_day_user.astimezone(TZ.utc)
                    filter_description = f"on {start_of_day_user.strftime('%Y-%m-%d')}"
                else:
                    return f"Could not understand the date: '{specific_date_str}'. Please use YYYY-MM-DD or terms like 'today', 'last Monday'."

            elif date_filter:
                date_filter_lower = date_filter.lower()
                if date_filter_lower == "upcoming":
                    date_from_dt_utc = now_utc
                    filter_description = "upcoming"
                elif date_filter_lower == "past_7_days":
                    date_from_dt_utc = (now_utc - timedelta(days=7)).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    date_to_dt_utc = now_utc  # End of today (UTC)
                    filter_description = "in the past 7 days"
                elif date_filter_lower == "past_30_days":
                    date_from_dt_utc = (now_utc - timedelta(days=30)).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    date_to_dt_utc = now_utc  # End of today (UTC)
                    filter_description = "in the past 30 days"
                elif date_filter_lower == "all":
                    filter_description = "all recorded"
                    # date_from_dt_utc and date_to_dt_utc remain None
                else:
                    return f"Unknown date_filter: '{date_filter}'. Try 'upcoming', 'past_7_days', 'past_30_days', or 'all', or provide a 'specific_date_str'."
            else:  # Default if neither specific_date nor date_filter provided
                date_from_dt_utc = now_utc  # Default to upcoming
                filter_description = "upcoming (default)"
            # --- End Date Logic ---

            appointments_data = await get_appointments(
                db,
                user_id=user_id,  # doctor_id
                role="doctor",
                patient_id=patient.user_id,
                date_from=date_from_dt_utc,
                date_to=date_to_dt_utc,
                limit=limit,
            )

            if not appointments_data:
                return f"No {filter_description} appointments found for {patient.first_name} {patient.last_name} with you."

            response_lines = [
                f"Appointments for {patient.first_name} {patient.last_name} with you ({filter_description}):"
            ]
            for appt_dict in appointments_data:
                # starts_at from DB is already UTC if stored correctly
                starts_at_utc = appt_dict["starts_at"]
                # Convert to user's timezone for display
                starts_at_user_tz = starts_at_utc.astimezone(effective_user_tz)
                display_time = starts_at_user_tz.strftime("%Y-%m-%d %I:%M %p %Z")

                response_lines.append(
                    f"- Date: {display_time}, Location: {appt_dict['location']}, Notes: {appt_dict.get('notes', 'N/A')}"
                )

            if len(appointments_data) == limit:
                response_lines.append(
                    f"\n(Showing up to {limit} appointments. More may exist.)"
                )

            return "\n".join(response_lines)

        except Exception as e:
            logger.error(
                f"Tool 'get_patient_appointment_history': Error for doctor_id '{user_id}', patient '{patient_full_name}': {e}",
                exc_info=True,
            )
            return "An unexpected error occurred while trying to retrieve patient appointment history."


@tool("get_my_schedule")
async def get_my_schedule(
    date_query: Annotated[
        str,
        "The date for which to fetch the schedule (e.g., 'today', 'tomorrow', '2025-07-10', 'next Monday'). Defaults to 'today' if not specified or ambiguous.",
    ],
    user_id: Annotated[int, InjectedState("user_id")],  # This is the doctor's ID
    user_tz: Annotated[Optional[str], InjectedState("user_tz")],
) -> str:
    """
    Fetches the calling doctor's own appointment schedule for a specified date.
    Use this when a doctor asks about their own appointments for a particular day.
    For example: 'What's my schedule for today?', 'Do I have any appointments tomorrow?', 'Show my schedule for October 26th'.
    """
    logger.info(
        f"Tool 'get_my_schedule' invoked by doctor_id '{user_id}' for date_query: '{date_query}' with user_tz: '{user_tz}'"
    )

    effective_user_tz_str = (
        user_tz or "UTC"
    )  # Default to UTC if user_tz is not available
    try:
        effective_user_tz = ZoneInfo(effective_user_tz_str)
    except Exception:
        logger.warning(
            f"Invalid user_tz '{user_tz}', defaulting to UTC for date parsing."
        )
        effective_user_tz = ZoneInfo("UTC")
        effective_user_tz_str = "UTC"

    now_user_tz = datetime.now(effective_user_tz)

    if (
        not date_query
        or date_query.lower() == "what date?"
        or date_query.lower() == "what day?"
    ):
        # If LLM asks for clarification or sends an empty query, default to today
        target_date_dt = now_user_tz
        date_query_for_log = "today (defaulted)"
    else:
        # Parse the date_query string
        # RELATIVE_BASE is important for "today", "tomorrow" to be relative to user's timezone
        target_date_dt = dateparser.parse(
            date_query,
            settings={
                "PREFER_DATES_FROM": "future",  # Slightly prefer future for ambiguous queries like "Monday"
                "RELATIVE_BASE": now_user_tz,  # Critical for "today", "tomorrow"
                "TIMEZONE": effective_user_tz_str,  # Interpret query in user's TZ
                # 'TO_TIMEZONE': 'UTC' # Not strictly needed here as we only need the date part
            },
        )

    if not target_date_dt:
        logger.warning(
            f"Could not parse date_query: '{date_query}' for doctor_id '{user_id}'. Defaulting to today."
        )
        target_date_dt = now_user_tz  # Default to today if parsing fails
        date_query_for_log = f"{date_query} (defaulted to today)"
    else:
        date_query_for_log = date_query

    target_date_obj: DateClass = target_date_dt.date()  # Extract the date part

    logger.info(
        f"Tool 'get_my_schedule': Parsed '{date_query_for_log}' to date: {target_date_obj} for doctor_id '{user_id}'"
    )

    async with tool_db_session() as db:
        try:
            appointments = await get_doctor_schedule_for_date(
                db, doctor_id=user_id, target_date=target_date_obj
            )

            if not appointments:
                return f"You have no appointments scheduled for {target_date_obj.strftime('%A, %B %d, %Y')}."

            # Format the schedule into a readable string
            # Times should be displayed in the doctor's local timezone
            schedule_lines = [
                f"Your schedule for {target_date_obj.strftime('%A, %B %d, %Y')}:"
            ]
            for appt in appointments:
                starts_at_utc = appt["starts_at"].replace(
                    tzinfo=TZ.utc
                )  # Ensure it's UTC
                ends_at_utc = appt["ends_at"].replace(tzinfo=TZ.utc)  # Ensure it's UTC

                starts_at_local = starts_at_utc.astimezone(effective_user_tz)
                ends_at_local = ends_at_utc.astimezone(effective_user_tz)

                patient_info = f"Patient: {appt['patient_name']}"
                notes_info = f"(Reason: {appt['notes']})" if appt["notes"] else ""

                schedule_lines.append(
                    f"- {starts_at_local.strftime('%I:%M %p')} - {ends_at_local.strftime('%I:%M %p')}: {patient_info} {notes_info}"
                )

            return "\n".join(schedule_lines)

        except Exception as e:
            logger.error(
                f"Tool 'get_my_schedule': Error processing request for doctor_id '{user_id}', date '{target_date_obj}': {e}",
                exc_info=True,
            )
            return "An unexpected error occurred while trying to retrieve your schedule. Please try again later."


@tool("execute_doctor_day_cancellation_confirmed")
async def execute_doctor_day_cancellation_confirmed(
    date_query: Annotated[
        str,
        "The date for which to cancel all of the doctor's appointments (e.g., 'today', 'tomorrow', '2025-07-10'). This tool should ONLY be called AFTER the doctor has EXPLICITLY CONFIRMED the cancellation in a previous conversational turn.",
    ],
    user_id: Annotated[int, InjectedState("user_id")],  # Doctor's ID
    user_tz: Annotated[Optional[str], InjectedState("user_tz")],
) -> str:
    """
    Cancels ALL of the calling doctor's appointments for a specified date.
    IMPORTANT: This tool performs the cancellation directly. The AI assistant MUST have ALREADY VERBALLY CONFIRMED with the doctor (e.g., "You have X appointments on {date}, are you sure you want to cancel all of them?") and received a 'yes' BEFORE calling this tool.
    Do NOT call this tool without prior explicit confirmation from the doctor in the conversation.

    Args:
        date_query: The date (e.g., "today", "tomorrow", "YYYY-MM-DD") for which appointments are to be cancelled.
        user_id: The ID of the doctor whose appointments are to be cancelled.
        user_tz: The timezone of the user for correct date parsing.
    """
    logger.info(
        f"Tool 'execute_doctor_day_cancellation_confirmed' invoked by doctor_id '{user_id}' for date_query: '{date_query}'"
    )

    effective_user_tz_str = user_tz or "UTC"
    try:
        effective_user_tz = ZoneInfo(effective_user_tz_str)
    except Exception:
        logger.warning(
            f"Invalid user_tz '{user_tz}', defaulting to UTC for date parsing in cancellation tool."
        )
        effective_user_tz = ZoneInfo("UTC")
        # effective_user_tz_str = "UTC" # Not strictly needed to re-assign

    now_user_tz = datetime.now(effective_user_tz)

    parsed_date_dt = dateparser.parse(
        date_query,
        settings={
            "PREFER_DATES_FROM": "current_period",  # For "today", "tomorrow"
            "RELATIVE_BASE": now_user_tz,
            "TIMEZONE": effective_user_tz_str,
        },
    )

    if not parsed_date_dt:
        logger.warning(
            f"Could not parse date_query for cancellation: '{date_query}' for doctor_id '{user_id}'."
        )
        return f"Sorry, I could not understand the date '{date_query}' for cancellation. Please specify a clear date."

    target_date_obj: DateClass = parsed_date_dt.date()
    logger.info(
        f"Executing confirmed cancellation of all appointments for doctor_id {user_id} on date: {target_date_obj}"
    )

    async with tool_db_session() as db:
        try:
            # 1. Fetch appointments for this doctor on this date to get their IDs
            appointments_on_schedule_details = await get_doctor_schedule_for_date(
                db, doctor_id=user_id, target_date=target_date_obj
            )

            if not appointments_on_schedule_details:
                return f"No appointments were found for you on {target_date_obj.strftime('%A, %B %d, %Y')} to cancel."

            appointment_ids_to_cancel = [
                appt["id"] for appt in appointments_on_schedule_details
            ]
            logger.info(
                f"Found {len(appointment_ids_to_cancel)} appointments to cancel for doctor {user_id} on {target_date_obj}: IDs {appointment_ids_to_cancel}"
            )

            # 2. Proceed with cancellation for each appointment ID
            cancelled_count = 0
            failed_ids = []

            for appt_id in appointment_ids_to_cancel:
                try:
                    # The delete_appointment CRUD function checks if the appointment belongs to the user (doctor in this case)
                    # and has the correct role.
                    success = await delete_appointment(
                        db, appointment_id=appt_id, user_id=user_id, role="doctor"
                    )
                    if success:
                        cancelled_count += 1
                    else:
                        logger.warning(
                            f"delete_appointment returned False for appointment_id {appt_id} for doctor {user_id}."
                        )
                        failed_ids.append(appt_id)
                except Exception as e_del:
                    logger.error(
                        f"Error deleting appointment_id {appt_id} for doctor {user_id}: {e_del}",
                        exc_info=True,
                    )
                    failed_ids.append(appt_id)

            formatted_date = target_date_obj.strftime("%A, %B %d, %Y")
            response_message = f"Successfully cancelled {cancelled_count} appointment(s) for {formatted_date}."
            if failed_ids:
                response_message += f" Failed to cancel {len(failed_ids)} appointment(s) with IDs: {failed_ids} (they may have been already cancelled or an error occurred)."

            logger.info(
                f"Cancellation result for doctor {user_id} on {target_date_obj}: {response_message}"
            )
            return response_message

        except Exception as e:
            logger.error(
                f"Tool 'execute_doctor_day_cancellation_confirmed': General error for doctor_id '{user_id}', date '{target_date_obj}': {e}",
                exc_info=True,
            )
            return "An unexpected error occurred while trying to cancel your appointments. Please try again later."


@tool("get_my_financial_summary")
async def get_my_financial_summary(
    user_id: Annotated[int, InjectedState("user_id")],
) -> str:
    """
    Retrieves a summary of the calling doctor's financial information from the clinic's records,
    including salary, recent bonuses, and raises.
    Use this tool when the doctor inquires about their salary, compensation, recent bonuses, or raises.
    """
    logger.info(
        f"Tool 'get_my_financial_summary' invoked by doctor_id '{user_id}' - using DB"
    )

    async with tool_db_session() as db:
        financial_summary_model = await get_doctor_financial_summary_by_user_id(
            db, doctor_user_id=user_id
        )

    if not financial_summary_model:
        return "I'm sorry, but I could not find financial information for your profile in our records. For official details, please consult the HR department."

    # Formatting the response string
    doctor_name = "doctor"  # Default
    if financial_summary_model.user and financial_summary_model.user.doctor_profile:
        doctor_name = f"Dr. {financial_summary_model.user.doctor_profile.first_name} {financial_summary_model.user.doctor_profile.last_name}"

    response_lines = [
        f"Here's a summary of the financial information for {doctor_name} from our records:"
    ]

    # Helper to format currency
    def format_currency(value: Optional[decimal.Decimal]) -> str:
        if value is None:
            return "N/A"
        return f"${value:,.2f}"  # Format with commas and 2 decimal places

    response_lines.append(
        f"- Annual Base Salary: {format_currency(financial_summary_model.base_salary_annual)}"
    )

    if (
        financial_summary_model.last_bonus_amount
        and financial_summary_model.last_bonus_date
    ):
        bonus_reason = financial_summary_model.last_bonus_reason or "Not specified"
        response_lines.append(
            f"- Last Bonus: {format_currency(financial_summary_model.last_bonus_amount)} on {financial_summary_model.last_bonus_date.strftime('%Y-%m-%d')}. "
            f"Reason: {bonus_reason}"
        )
    else:
        response_lines.append(
            "- Last Bonus: No recent bonus information found in records."
        )

    if (
        financial_summary_model.last_raise_percentage
        and financial_summary_model.last_raise_date
    ):
        raise_reason = financial_summary_model.last_raise_reason or "Not specified"
        response_lines.append(
            f"- Last Raise: {financial_summary_model.last_raise_percentage:.2f}% on {financial_summary_model.last_raise_date.strftime('%Y-%m-%d')}. "
            f"Reason: {raise_reason}"
        )
    else:
        response_lines.append(
            "- Last Raise: No recent raise information found in records."
        )

    if financial_summary_model.next_review_period:
        response_lines.append(
            f"- Next Performance Review Period: {financial_summary_model.next_review_period}"
        )

    response_lines.append(
        "\nPlease note: For official and complete details, please always refer to the HR department or your employment contract."
    )

    return "\n".join(response_lines)
