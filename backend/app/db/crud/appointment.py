# app/db/crud/appointment.py
import logging
from datetime import datetime, date, time, timedelta, timezone
from typing import List, Optional, Dict, Any, Union

from sqlalchemy import select, insert, delete, update, and_, or_, Date, cast
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, NoResultFound
from fastapi import HTTPException
from babel.dates import format_datetime
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
    user_id: int,
    role: str,
    skip: int = 0,
    limit: int = 100,
    doctor_id: Optional[int] = None,
    patient_id: Optional[int] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Get appointments based on filters with role-based access control and enhanced response.

    Args:
        db: Database session
        user_id: ID of the current user
        role: Role of the current user (patient, doctor, admin)
        skip: Number of records to skip
        limit: Maximum number of records to return
        doctor_id: Filter by doctor ID
        patient_id: Filter by patient ID
        date_from: Filter by appointments starting at or after this date
        date_to: Filter by appointments starting at or before this date

    Returns:
        List of appointments with formatted dates and doctor profiles.
    """
    query = select(AppointmentModel)

    # Apply role-based filtering
    if role == "doctor":
        query = query.where(AppointmentModel.doctor_id == int(user_id))
    elif role == "patient":
        query = query.where(AppointmentModel.patient_id == int(user_id))
    elif role != "admin":
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    # Apply optional filters
    if doctor_id:
        query = query.where(AppointmentModel.doctor_id == int(doctor_id))
    if patient_id:
        query = query.where(AppointmentModel.patient_id == int(patient_id))
    if date_from:
        query = query.where(AppointmentModel.starts_at >= date_from)
    if date_to:
        query = query.where(AppointmentModel.starts_at <= date_to)

    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    appointments = result.scalars().all()

    enhanced_appointments = []
    for appointment in appointments:
        # Fetch the doctor profile
        doctor = await get_user(db, appointment.doctor_id)
        if not doctor or not doctor.doctor_profile:
            continue  # Skip appointments with missing doctor profiles

        logger.info(
            f"Doctor profile for appointment ID {appointment.id}: {doctor.doctor_profile}"
        )

        logger.debug(f"Processing appointment ID {appointment.id}")
        logger.debug(f"Doctor data: {doctor}")
        logger.debug(f"Doctor profile: {doctor.doctor_profile}")

        enhanced_appointments.append(
            {
                "id": appointment.id,
                "patient_id": appointment.patient_id,
                "doctor_id": appointment.doctor_id,  # Add doctor_id to the response
                "doctor_profile": {
                    "first_name": doctor.doctor_profile.first_name,
                    "last_name": doctor.doctor_profile.last_name,
                    "specialty": doctor.doctor_profile.specialty,
                },
                "starts_at": appointment.starts_at,  # Return datetime object
                "ends_at": appointment.ends_at,  # Return datetime object
                "location": appointment.location,
                "notes": appointment.notes,
                "created_at": appointment.created_at,  # Return datetime object
            }
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
    start_hour = 8
    end_hour = 17

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
    target_date: date,
    slot_duration: int = 30,
    format_time: bool = True,
) -> List[str]:
    """
    Get available time slots for a doctor on a specific date formatted as strings

    Args:
        db: Database session
        doctor_id: ID of the doctor
        target_date: Date to check availability for (date object, not datetime)
        slot_duration: Duration of each slot in minutes
        format_time: Whether to return times as formatted strings (e.g., "9:00 AM") or datetime objects

    Returns:
        List of available time slots as formatted strings or datetime objects
    """
    logger.info(f"Getting available slots for doctor {doctor_id} on {target_date}")

    try:
        # Convert date to datetime for availability check with UTC timezone
        target_datetime = datetime.combine(target_date, time(0, 0))

        # Get available slots as datetime objects
        available_slots = await get_doctor_availability(
            db=db,
            doctor_id=doctor_id,
            date=target_datetime,
            slot_duration=slot_duration,
        )

        # Format times if requested
        if format_time:
            formatted_slots = []
            for slot in available_slots:
                # Format as "9:00 AM", "2:30 PM", etc.
                formatted_time = slot.strftime("%-I:%M %p")
                formatted_slots.append(formatted_time)
            return formatted_slots

        return available_slots

    except Exception as e:
        # Log the error but don't re-raise - return a properly structured error response
        logger.error(
            f"Error getting available slots for doctor {doctor_id}: {e}", exc_info=True
        )
        # Return empty list which will be handled by the tool to show "no slots available"
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
