import logging
from datetime import datetime, date, time, timedelta, timezone
from typing import List, Optional, Dict, Any, Union
from zoneinfo import ZoneInfo  # Added import

from sqlalchemy import select, insert, delete, update, and_, or_, Date, cast
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, NoResultFound
from fastapi import HTTPException
from app.db.crud.user import get_user

from app.db.models.appointment import AppointmentModel
from app.db.models.user import UserModel

logger = logging.getLogger(__name__)


async def create_appointment(
    db: AsyncSession,
    patient_id: int,
    doctor_id: int,
    starts_at: datetime,  # Should be UTC datetime
    ends_at: datetime,  # Should be UTC datetime
    location: str,
    notes: Optional[str] = None,
    google_calendar_event_id: Optional[str] = None,
) -> Union[AppointmentModel, Dict[str, Any]]:
    """
    Create a new appointment in the database.

    Checks for doctor existence and scheduling conflicts before creation.
    Sets the default status of the new appointment to 'scheduled'.
    Optionally stores a Google Calendar event ID if provided.

    Args:
        db (AsyncSession): The database session.
        patient_id (int): The ID of the patient for the appointment.
        doctor_id (int): The ID of the doctor for the appointment.
        starts_at (datetime): The UTC start date and time of the appointment.
        ends_at (datetime): The UTC end date and time of the appointment.
        location (str): The location of the appointment.
        notes (Optional[str]): Additional notes for the appointment.
        google_calendar_event_id (Optional[str]): The event ID from Google Calendar, if any.

    Returns:
        Union[AppointmentModel, Dict[str, Any]]: The created AppointmentModel instance on success,
        or a dictionary with 'status' and 'message' keys on failure (e.g., conflict, error).
    """
    logger.info(
        f"CRUD: Attempting to create appointment for patient_id={patient_id} with doctor_id={doctor_id} "
        f"from {starts_at} to {ends_at}. GCal ID: {google_calendar_event_id}"
    )
    try:
        # 1. Validate Doctor
        doctor_user = await get_user(
            db, doctor_id
        )  # Assuming get_user fetches UserModel
        if not doctor_user or doctor_user.role != "doctor":
            logger.warning(
                f"CRUD: Doctor validation failed for doctor_id={doctor_id}. User role: {doctor_user.role if doctor_user else 'None'}"
            )
            return {
                "status": "error",
                "message": "Doctor not found or the specified user is not a doctor.",
            }
        # 2. Check for Scheduling Conflicts
        # Only check against other 'scheduled' appointments for the same doctor at the overlapping time.
        conflict_stmt = select(
            AppointmentModel
        ).where(
            and_(
                AppointmentModel.doctor_id == doctor_id,
                AppointmentModel.status
                == "scheduled",  # Important: only conflict with active appointments
                starts_at
                < AppointmentModel.ends_at,  # New appointment starts before an existing one ends
                ends_at
                > AppointmentModel.starts_at,  # New appointment ends after an existing one starts
            )
        )
        conflict_result = await db.execute(conflict_stmt)
        conflicting_appointment = conflict_result.scalars().first()

        if conflicting_appointment:
            logger.warning(
                f"CRUD: Scheduling conflict detected for doctor_id={doctor_id} at {starts_at}. "
                f"Conflicts with appointment_id={conflicting_appointment.id}"
            )
            return {
                "status": "conflict",
                "message": "This time slot is already booked with a scheduled appointment.",
            }
        # 3. Create New Appointment Instance
        new_appointment_data = {
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "starts_at": starts_at,  # Ensure this is UTC
            "ends_at": ends_at,  # Ensure this is UTC
            "location": location,
            "notes": notes,
            "status": "scheduled",  # << SET DEFAULT STATUS
        }
        if google_calendar_event_id:  # << STORE GCAL ID IF PROVIDED
            new_appointment_data["google_calendar_event_id"] = google_calendar_event_id

        new_appointment = AppointmentModel(**new_appointment_data)

        # 4. Add to DB and Commit
        db.add(new_appointment)
        await db.commit()
        await db.refresh(
            new_appointment
        )  # To get DB-generated values like ID and created_at

        logger.info(
            f"CRUD: Successfully created appointment_id={new_appointment.id} with status='{new_appointment.status}'."
        )
        return new_appointment  # << RETURN THE MODEL INSTANCE ON SUCCESS

    except IntegrityError as e:  # Catches DB-level unique constraint violations
        await db.rollback()
        # This might be redundant if the conflict check above is thorough and the unique constraint includes status.
        # However, it's a good safety net.
        logger.error(
            f"CRUD: Database integrity error during appointment creation: {e}",
            exc_info=True,
        )
        return {
            "status": "conflict",
            "message": "A database integrity error occurred, possibly due to a conflicting appointment. Please try a different time.",
        }
    except Exception as e:
        await db.rollback()
        logger.error(
            f"CRUD: Unexpected error creating appointment for patient_id={patient_id}, doctor_id={doctor_id}. "
            f"Error: {type(e).__name__} - {e}",
            exc_info=True,
        )
        return {
            "status": "error",
            "message": f"An unexpected error occurred while attempting to book the appointment: {str(e)}",
        }


async def get_appointments(
    db: AsyncSession,
    user_id: int,  # ID of the user making the request
    role: str,  # Role of the user making the request
    skip: int = 0,
    limit: int = 100,
    doctor_id: Optional[int] = None,  # Optional filter: for which doctor's appointments
    patient_id: Optional[
        int
    ] = None,  # Optional filter: for which patient's appointments
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    target_specific_date: Optional[
        date
    ] = None,  # <<< THIS IS THE NEW/REQUIRED PARAMETER
) -> List[Dict[str, Any]]:
    """
    Get appointments based on filters with role-based access control.
    Prioritizes target_specific_date if provided; otherwise, uses date_from/date_to.
    """
    logger.debug(
        f"CRUD get_appointments: user_id={user_id}, role='{role}', doctor_id={doctor_id}, patient_id={patient_id}, "
        f"date_from={date_from}, date_to={date_to}, target_specific_date={target_specific_date}, "
        f"skip={skip}, limit={limit}"
    )

    query = select(AppointmentModel)

    # Apply primary role-based filtering (who is asking and what they can see by default)
    if role == "doctor":
        query = query.where(AppointmentModel.doctor_id == int(user_id))
        # If a doctor is asking, and they also specify a patient_id, further filter for that patient.
        if patient_id is not None:
            query = query.where(AppointmentModel.patient_id == int(patient_id))
        # A doctor generally shouldn't be using the doctor_id filter for anyone but themselves unless it's an admin feature
        if doctor_id is not None and int(doctor_id) != int(user_id):
            logger.warning(
                f"Doctor {user_id} querying for appointments of another doctor {doctor_id}. This might be an admin action or misconfiguration."
            )
            # If you want to strictly enforce a doctor only sees their own, even if doctor_id is passed:
            # query = query.where(AppointmentModel.doctor_id == int(user_id))
            # Or if you allow this (e.g. for admins impersonating or specific cross-doctor views):
            query = query.where(AppointmentModel.doctor_id == int(doctor_id))

    elif role == "patient":
        query = query.where(AppointmentModel.patient_id == int(user_id))
        # If a patient is asking, and they also specify a doctor_id, filter for that doctor.
        if doctor_id is not None:
            query = query.where(AppointmentModel.doctor_id == int(doctor_id))
        # A patient cannot query for other patients' appointments via the patient_id filter
        if patient_id is not None and int(patient_id) != int(user_id):
            logger.warning(
                f"Patient {user_id} trying to query for another patient {patient_id}. Denying."
            )
            return []

    elif role == "admin":
        # Admin can filter by specific doctor_id or patient_id if provided
        if doctor_id is not None:
            query = query.where(AppointmentModel.doctor_id == int(doctor_id))
        if patient_id is not None:
            query = query.where(AppointmentModel.patient_id == int(patient_id))
    else:
        logger.error(f"Unknown role '{role}' attempting to get appointments.")
        raise HTTPException(
            status_code=403, detail="Insufficient permissions for an unknown role."
        )

    # --- DATE FILTERING LOGIC ---
    if target_specific_date:
        logger.debug(
            f"Filtering appointments for specific date: {target_specific_date}"
        )
        query = query.where(
            cast(AppointmentModel.starts_at, Date) == target_specific_date
        )
    elif date_from and date_to:
        logger.debug(f"Filtering appointments from {date_from} to {date_to}")
        # Ensure date_from and date_to are timezone-aware if starts_at is
        query = query.where(
            AppointmentModel.starts_at >= date_from,
            AppointmentModel.starts_at <= date_to,
        )
    elif date_from:  # Only start date provided (e.g., upcoming)
        logger.debug(f"Filtering appointments from {date_from} onwards")
        query = query.where(AppointmentModel.starts_at >= date_from)
    elif date_to:  # Only end date provided
        logger.debug(f"Filtering appointments up to {date_to}")
        query = query.where(AppointmentModel.starts_at <= date_to)

    query = query.order_by(AppointmentModel.starts_at).offset(skip).limit(limit)

    result = await db.execute(query)
    appointments_models = result.scalars().all()

    enhanced_appointments = []
    for appt_model in appointments_models:
        # Initialize with defaults
        doctor_profile_data = {"first_name": "N/A", "last_name": "", "specialty": "N/A"}
        patient_name_str = "N/A"

        # Fetch doctor for this appointment to get profile
        # This assumes AppointmentModel has a 'doctor' relationship to UserModel
        doc_user = await get_user(
            db, appt_model.doctor_id
        )  # Using the minimal get_user
        if doc_user and doc_user.doctor_profile:
            doctor_profile_data = {
                "first_name": doc_user.doctor_profile.first_name,
                "last_name": doc_user.doctor_profile.last_name,
                "specialty": doc_user.doctor_profile.specialty,
            }

        # Fetch patient for this appointment to get name
        # This assumes AppointmentModel has a 'patient' relationship to UserModel
        pat_user = await get_user(
            db, appt_model.patient_id
        )  # Using the minimal get_user
        if pat_user and pat_user.patient_profile:
            patient_name_str = f"{pat_user.patient_profile.first_name} {pat_user.patient_profile.last_name}".strip()

        enhanced_appointments.append(
            {
                "id": appt_model.id,
                "patient_id": appt_model.patient_id,
                "patient_name": patient_name_str,  # Include patient name
                "doctor_id": appt_model.doctor_id,
                "doctor_profile": doctor_profile_data,
                "starts_at": appt_model.starts_at,
                "ends_at": appt_model.ends_at,
                "location": appt_model.location,
                "notes": appt_model.notes,
                "created_at": appt_model.created_at,
                "is_discharged": getattr(appt_model, "is_discharged", False),
            }
        )

    logger.info(
        f"CRUD get_appointments: Found {len(enhanced_appointments)} appointments matching criteria."
    )
    return enhanced_appointments


async def get_appointment(
    db: AsyncSession, appointment_id: int, user_id: int, role: str
) -> Optional[AppointmentModel]:  # Return type changed to AppointmentModel
    """
    Get a specific appointment by ID with permission checks.
    Returns the AppointmentModel instance if found and user has access.
    """
    logger.info(
        f"Fetching appointment model {appointment_id} for user {user_id} with role {role}"
    )
    result = await db.execute(
        select(AppointmentModel).where(AppointmentModel.id == appointment_id)
    )
    appointment = result.scalars().first()

    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")

    logger.info(
        f"Appointment model {appointment_id} details: Patient ID {appointment.patient_id}, Doctor ID {appointment.doctor_id}"
    )
    # Check permissions - user must be the patient, doctor, or admin
    if (
        appointment.patient_id != user_id
        and appointment.doctor_id != user_id
        and role != "admin"
    ):
        raise HTTPException(
            status_code=403, detail="Not authorized to access this appointment"
        )

    # Return the raw model instance. Formatting is moved to the route handler.
    return appointment


async def update_appointment(
    db: AsyncSession,
    appointment_id: int,
    user_id: int,
    role: str,
    update_data: Dict[str, Any],
) -> AppointmentModel:
    """
    Update an existing appointment

    Args:
        db: Database session
        appointment_id: ID of the appointment to update
        user_id: ID of the current user
        role: Role of the current user (patient, doctor, admin)
        update_data: Dictionary of fields to update

    Returns:
        The updated appointment
    """
    # Get the appointment
    appointment = await get_appointment(db, appointment_id, user_id, role)

    # Check if changing time or doctor, check for conflicts
    if (
        "starts_at" in update_data
        or "ends_at" in update_data
        or "doctor_id" in update_data
    ):
        doctor_id = update_data.get("doctor_id", appointment.doctor_id)
        starts_at = update_data.get("starts_at", appointment.starts_at)
        ends_at = update_data.get("ends_at", appointment.ends_at)

        result = await db.execute(
            select(AppointmentModel).where(
                and_(
                    AppointmentModel.id != appointment_id,
                    AppointmentModel.doctor_id == doctor_id,
                    starts_at < AppointmentModel.ends_at,
                    ends_at > AppointmentModel.starts_at,
                )
            )
        )
        conflict = result.scalars().first()

        if conflict:
            raise HTTPException(status_code=409, detail="Time slot already booked")

    # Apply the updates
    for key, value in update_data.items():
        setattr(appointment, key, value)

    await db.commit()
    await db.refresh(appointment)
    return appointment


async def delete_appointment(
    db: AsyncSession,
    appointment_id: int,
    user_id: int,  # This is the ID of the user initiating the delete
    role: str,  # Role of the user
) -> bool:
    """
    Delete an appointment.
    Ensures the user has permission based on their role.
    Patient can delete their own. Doctor can delete appointments they are part of.
    Admin can delete any.
    """
    logger.info(
        f"CRUD: User {user_id} (role: {role}) attempting to hard delete appointment_id={appointment_id}"
    )

    # First, get the appointment to check ownership/permissions
    # get_appointment already handles role-based access checks and raises HTTPException if not authorized or not found
    try:
        appointment_to_delete = await get_appointment(db, appointment_id, user_id, role)
    except HTTPException as http_exc:
        # If get_appointment raises an error (e.g., not found, not authorized), propagate a failure.
        logger.warning(
            f"CRUD: Permission check failed or appointment not found for hard delete. Appt ID: {appointment_id}, User: {user_id}, Role: {role}. Detail: {http_exc.detail}"
        )
        return False  # Indicate failure

    # If get_appointment didn't raise an exception, then appointment_to_delete exists and user is authorized.
    if appointment_to_delete:
        await db.delete(appointment_to_delete)
        await db.commit()
        logger.info(f"CRUD: Successfully hard deleted appointment_id={appointment_id}")
        return True
    else:
        # This case should ideally be caught by get_appointment raising an error for "not found".
        # But as a fallback:
        logger.warning(
            f"CRUD: Appointment {appointment_id} not found for deletion (unexpected after get_appointment call)."
        )
        return False


async def get_doctor_availability(
    db: AsyncSession, doctor_id: int, date: datetime, slot_duration: int = 30
) -> List[datetime]:
    """
    Get available time slots for a doctor on a specific date

    Args:
        db: Database session
        doctor_id: ID of the doctor
        date: Date to check availability for
        slot_duration: Duration of each slot in minutes

    Returns:
        List of available datetime slots
    """
    # Check if doctor exists and is a doctor
    doctor = await get_user(db, doctor_id)
    if not doctor or doctor.role != "doctor":
        raise HTTPException(status_code=404, detail="Doctor not found")

    # Define working hours (8 AM to 5 PM)
    start_hour = 5
    end_hour = 12

    # Get the start and end of the requested date - MAKE TIMEZONE AWARE
    date_start = datetime(
        date.year, date.month, date.day, start_hour, 0, tzinfo=timezone.utc
    )
    date_end = datetime(
        date.year, date.month, date.day, end_hour, 0, tzinfo=timezone.utc
    )

    # Get existing appointments for the doctor on that date
    result = await db.execute(
        select(AppointmentModel).where(
            and_(
                AppointmentModel.doctor_id == doctor_id,
                AppointmentModel.starts_at >= date_start,
                AppointmentModel.starts_at < date_start + timedelta(days=1),
            )
        )
    )
    existing_appointments = result.scalars().all()

    # Create a list of all possible time slots - MAKE TIMEZONE AWARE
    all_slots = []
    current_slot = date_start
    while current_slot < date_end:
        all_slots.append(current_slot)
        current_slot += timedelta(minutes=slot_duration)

    # Filter out booked slots
    available_slots = []
    for slot in all_slots:
        slot_end = slot + timedelta(minutes=slot_duration)
        is_available = True

        for appointment in existing_appointments:
            # Ensure appointment times are timezone-aware for comparison
            appt_starts_at = appointment.starts_at
            appt_ends_at = appointment.ends_at

            # Compare timezone-aware datetimes
            if slot < appt_ends_at and slot_end > appt_starts_at:
                is_available = False
                break

        if is_available:
            available_slots.append(slot)

    return available_slots


async def get_available_slots_for_day(
    db: AsyncSession,
    doctor_id: int,
    target_date: date,  # This is a date object, representing the day in user's perspective
    user_tz: Optional[str],  # Added user_tz parameter
    slot_duration: int = 30,
    format_time: bool = True,
) -> List[str]:  # Return type is List[str] if format_time is True, otherwise List[datetime]
    """
    Get available time slots for a doctor on a specific date formatted as strings
    or as datetime objects, adjusted for the user's timezone if provided.

    Args:
        db: Database session
        doctor_id: ID of the doctor
        target_date: Date to check availability for (date object, user's perspective)
        user_tz: The user's timezone string (e.g., 'America/New_York')
        slot_duration: Duration of each slot in minutes
        format_time: Whether to return times as formatted strings (e.g., "9:00 AM")

    Returns:
        List of available time slots (strings or datetimes).
    """
    logger.info(f"Getting available slots for doctor {doctor_id} on {target_date} for user_tz {user_tz}")

    try:
        # target_datetime represents the start of the target_date in a naive way,
        # get_doctor_availability will handle UTC conversions for working hours.
        target_datetime_naive = datetime.combine(target_date, time(0, 0))

        # Get available slots as UTC datetime objects
        available_utc_slots: List[datetime] = await get_doctor_availability(
            db=db,
            doctor_id=doctor_id,
            date=target_datetime_naive,  # Pass the naive datetime; get_doctor_availability assumes it's for UTC working hours
            slot_duration=slot_duration,
        )

        if not format_time:
            # If not formatting, consumer needs to be aware these are UTC
            # Or, we could convert them here too if a user_tz is provided.
            # For now, returning raw UTC datetimes if not formatting.
            return available_utc_slots

        # Format times, converting to user's timezone if provided
        formatted_slots = []
        display_tz = ZoneInfo(user_tz) if user_tz else timezone.utc

        for slot_utc in available_utc_slots:
            slot_display_tz = slot_utc.astimezone(display_tz)
            # Format as "9:00 AM", "2:30 PM", etc.
            formatted_time = slot_display_tz.strftime("%-I:%M %p").strip()  # strip potential leading space from %-I on some systems
            if formatted_time.startswith("0"):  # handle cases like "09:00 AM" -> "9:00 AM"
                formatted_time = formatted_time[1:]
            formatted_slots.append(formatted_time)

        logger.debug(f"Formatted slots for doctor {doctor_id} on {target_date} in tz {display_tz}: {formatted_slots}")
        return formatted_slots

    except Exception as e:
        logger.error(
            f"Error getting available slots for doctor {doctor_id} on {target_date} with user_tz {user_tz}: {e}", exc_info=True
        )
        return []


async def get_doctor_schedule_for_date(
    db: AsyncSession,
    doctor_id: int,
    target_date: date,
) -> List[Dict[str, Any]]:
    """
    Retrieves appointments for a specific doctor on a given date.

    Args:
        db (AsyncSession): The database session.
        doctor_id (int): The user_id of the doctor.
        target_date (date): The specific date to fetch appointments for.

    Returns:
        List[Dict[str, Any]]: A list of appointment details, including patient information.
    """
    logger.info(f"CRUD: Fetching schedule for doctor_id {doctor_id} on {target_date}")

    try:
        stmt = (
            select(AppointmentModel)
            .where(
                AppointmentModel.doctor_id == doctor_id,
                # Cast the stored UTC datetime to a DATE for comparison with target_date
                # This assumes starts_at is a DateTime field.
                cast(AppointmentModel.starts_at, Date) == target_date,
            )
            .options(
                selectinload(AppointmentModel.patient).selectinload(
                    UserModel.patient_profile
                )  # Eager load patient user and then their profile
            )
            .order_by(AppointmentModel.starts_at)
        )

        result = await db.execute(stmt)
        appointments = result.scalars().all()

        schedule = []
        if appointments:
            for appt in appointments:
                patient_name = "N/A"
                if appt.patient and appt.patient.patient_profile:
                    patient_name = f"{appt.patient.patient_profile.first_name} {appt.patient.patient_profile.last_name}"

                schedule.append(
                    {
                        "id": appt.id,
                        "starts_at": appt.starts_at,  # Keep as datetime for now, tool will format
                        "ends_at": appt.ends_at,  # Keep as datetime
                        "patient_id": appt.patient_id,
                        "patient_name": patient_name,
                        "location": appt.location,
                        "notes": appt.notes,
                    }
                )

        logger.info(
            f"CRUD: Found {len(schedule)} appointments for doctor_id {doctor_id} on {target_date}"
        )
        return schedule

    except Exception as e:
        logger.error(
            f"CRUD: Error fetching schedule for doctor {doctor_id} on {target_date}: {e}",
            exc_info=True,
        )
        return []


async def update_appointment_gcal_id(
    db: AsyncSession, appointment_id: int, google_calendar_event_id: str
) -> bool:
    """Updates the google_calendar_event_id for a given appointment."""
    logger.info(
        f"CRUD: Updating GCal event ID for appointment {appointment_id} to {google_calendar_event_id}"
    )
    stmt = (
        update(AppointmentModel)
        .where(AppointmentModel.id == appointment_id)
        .values(google_calendar_event_id=google_calendar_event_id)
        .returning(AppointmentModel.id)  # To check if a row was updated
    )
    result = await db.execute(stmt)
    updated_id = result.scalar_one_or_none()
    if updated_id:
        await db.commit()
        logger.info(
            f"CRUD: Successfully updated GCal event ID for appointment {appointment_id}"
        )
        return True
    else:
        logger.warning(
            f"CRUD: Failed to update GCal event ID for appointment {appointment_id} (appointment not found)."
        )
        await db.rollback()  # Good practice
        return False


async def get_appointments_for_doctor_on_date(
    db: AsyncSession,
    doctor_id: int,
    target_date: date,  # Pass a date object
) -> List[AppointmentModel]:
    """
    Retrieves all 'scheduled' appointments for a specific doctor on a given date.
    """
    logger.info(
        f"CRUD: Fetching 'scheduled' appointments for doctor {doctor_id} on date {target_date}"
    )

    start_of_day_utc = datetime.combine(target_date, time.min, tzinfo=timezone.utc)
    end_of_day_utc = datetime.combine(target_date, time.max, tzinfo=timezone.utc)

    stmt = (
        select(AppointmentModel)
        .where(
            and_(  # Make sure 'and_' is imported from sqlalchemy
                AppointmentModel.doctor_id == doctor_id,
                AppointmentModel.starts_at >= start_of_day_utc,
                AppointmentModel.starts_at <= end_of_day_utc,
                AppointmentModel.status == "scheduled",
            )
        )
        .order_by(AppointmentModel.starts_at)
    )

    result = await db.execute(stmt)
    appointments = result.scalars().all()
    logger.info(
        f"CRUD: Found {len(appointments)} 'scheduled' appointments for doctor {doctor_id} on {target_date}"
    )
    return appointments


async def mark_appointment_discharged(
    db: AsyncSession,
    appointment_id: int,
    doctor_id: int,  # To ensure the doctor owns this appointment
) -> Optional[AppointmentModel]:
    """
    Marks a specific appointment as discharged.
    Ensures the appointment belongs to the requesting doctor.
    """
    logger.info(
        f"CRUD: Attempting to mark appointment_id {appointment_id} as discharged by doctor_id {doctor_id}"
    )

    # Use the existing get_appointment which has permission checks
    try:
        # Role 'doctor' for permission check within get_appointment
        appointment_to_update = await get_appointment(
            db, appointment_id=appointment_id, user_id=doctor_id, role="doctor"
        )
    except HTTPException as e:
        if e.status_code == 404:
            logger.warning(
                f"CRUD: Appointment {appointment_id} not found for doctor {doctor_id}."
            )
            return None
        elif e.status_code == 403:
            logger.warning(
                f"CRUD: Doctor {doctor_id} not authorized to modify appointment {appointment_id}."
            )
            return None
        raise  # Re-raise other HTTPExceptions

    if not appointment_to_update:
        logger.warning(
            f"CRUD: Appointment {appointment_id} not found or not associated with doctor_id {doctor_id}."
        )
        return None

    if appointment_to_update.is_discharged:
        logger.info(
            f"CRUD: Appointment {appointment_id} is already marked as discharged."
        )
        # Optionally return it anyway, or a specific message
        return appointment_to_update

    appointment_to_update.is_discharged = True
    try:
        await db.commit()
        await db.refresh(appointment_to_update)
        logger.info(
            f"CRUD: Successfully marked appointment {appointment_id} as discharged."
        )
        return appointment_to_update
    except Exception as e:
        await db.rollback()
        logger.error(
            f"CRUD: Error committing discharge for appointment {appointment_id}: {e}",
            exc_info=True,
        )
        return None
