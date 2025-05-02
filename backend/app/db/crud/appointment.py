# backend/app/db/crud/appointment.py
import logging
from datetime import datetime, date, time, timedelta, timezone
from typing import List, Optional, Dict, Any

from sqlalchemy import select, insert, delete, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, NoResultFound

from app.db.models.appointment import AppointmentModel
from app.db.models.user import UserModel
from app.config.settings import settings

logger = logging.getLogger(__name__)

async def get_available_slots_for_day(db: AsyncSession, doctor_id: int, target_day: date) -> List[str]:
    """
    Finds available 30-minute slots for a doctor on a given day.
    Assumes working hours e.g., 9 AM to 5 PM.
    """
    logger.info(f"Checking available slots for doctor {doctor_id} on {target_day}")
    slots = []
    working_start_time = time(9, 0)
    working_end_time = time(17, 0)
    slot_duration = timedelta(minutes=30)

    try:
        # Combine date and time for start and end of the target day
        day_start = datetime.combine(target_day, working_start_time, tzinfo=timezone.utc)
        day_end = datetime.combine(target_day, working_end_time, tzinfo=timezone.utc)

        # Fetch existing appointments for the doctor on that day
        stmt = select(AppointmentModel.starts_at, AppointmentModel.ends_at).where(
            AppointmentModel.doctor_id == doctor_id,
            AppointmentModel.starts_at >= day_start,
            AppointmentModel.starts_at < day_end # Appointments starting before 5 PM
        ).order_by(AppointmentModel.starts_at)

        result = await db.execute(stmt)
        booked_slots = result.fetchall() # List of (starts_at, ends_at) tuples

        # Iterate through potential slots and check for overlaps
        current_slot_start = day_start
        while current_slot_start < day_end:
            current_slot_end = current_slot_start + slot_duration
            is_free = True
            for booked_start, booked_end in booked_slots:
                # Check for overlap: (StartA < EndB) and (EndA > StartB)
                if (current_slot_start < booked_end) and (current_slot_end > booked_start):
                    is_free = False
                    break # Overlaps with this booked slot

            if is_free:
                slots.append(current_slot_start.strftime("%H:%M")) # Format as HH:MM

            current_slot_start += slot_duration

        logger.info(f"Found {len(slots)} available slots for doctor {doctor_id} on {target_day}")
        return slots

    except Exception as e:
        logger.error(f"Error fetching available slots for doctor {doctor_id} on {target_day}: {e}", exc_info=True)
        return [] # Return empty list on error

async def create_appointment(db: AsyncSession, patient_id: int, doctor_id: int, starts_at: datetime, ends_at: datetime, location: str, notes: Optional[str]) -> Dict[str, Any] | None:
    """Creates a new appointment in the database."""
    logger.info(f"Attempting to book appointment for patient {patient_id} with doctor {doctor_id} at {starts_at}")
    try:
        # Optional: Add extra validation (e.g., check if doctor/patient exist)
        # stmt_doc = select(UserModel).where(UserModel.id == doctor_id, UserModel.role == 'doctor')
        # ...

        # Check for conflicts before inserting
        conflict_stmt = select(AppointmentModel.id).where(
            AppointmentModel.doctor_id == doctor_id,
            or_(
                # Existing appointment overlaps new appointment
                and_(AppointmentModel.starts_at < ends_at, AppointmentModel.ends_at > starts_at),
            )
        ).limit(1)
        result = await db.execute(conflict_stmt)
        if result.scalar_one_or_none() is not None:
            logger.warning(f"Booking conflict detected for doctor {doctor_id} at {starts_at}")
            return {"status": "conflict", "message": "The selected time slot is no longer available."}


        appointment = AppointmentModel(
            patient_id=patient_id,
            doctor_id=doctor_id,
            starts_at=starts_at,
            ends_at=ends_at,
            location=location,
            notes=notes
        )
        db.add(appointment)
        await db.commit()
        await db.refresh(appointment)
        logger.info(f"Successfully booked appointment ID {appointment.id}")
        return {
            "status": "confirmed",
            "id": appointment.id,
            "starts_at": appointment.starts_at.isoformat(),
            "ends_at": appointment.ends_at.isoformat(),
            "location": appointment.location,
        }
    except IntegrityError as e: # Catches unique constraint violation specifically
        await db.rollback()
        logger.warning(f"Database integrity error during booking (likely conflict): {e}")
        return {"status": "conflict", "message": "This time slot has just been booked. Please try another time."}
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating appointment: {e}", exc_info=True)
        return {"status": "error", "message": "An unexpected error occurred while booking."}


async def delete_appointment(db: AsyncSession, appointment_id: int, patient_id: int) -> Dict[str, Any]:
    """Deletes an appointment, ensuring it belongs to the patient."""
    logger.info(f"Attempting to cancel appointment {appointment_id} for patient {patient_id}")
    try:
        stmt = delete(AppointmentModel).where(
            AppointmentModel.id == appointment_id,
            AppointmentModel.patient_id == patient_id # Ensure patient owns the appointment
        ).returning(AppointmentModel.id) # Optional: check if a row was actually deleted

        result = await db.execute(stmt)
        deleted_id = result.scalar_one_or_none()
        await db.commit()

        if deleted_id:
            logger.info(f"Successfully cancelled appointment {appointment_id}")
            return {"status": "cancelled", "message": f"Appointment {appointment_id} has been successfully cancelled."}
        else:
            # Check if appointment exists but belongs to another patient or doesn't exist
            exists_stmt = select(AppointmentModel.id).where(AppointmentModel.id == appointment_id).limit(1)
            exists_result = await db.execute(exists_stmt)
            if exists_result.scalar_one_or_none():
                 logger.warning(f"Patient {patient_id} attempted to cancel appointment {appointment_id} not belonging to them.")
                 return {"status": "forbidden", "message": "You do not have permission to cancel this appointment."}
            else:
                 logger.warning(f"Attempted to cancel non-existent appointment {appointment_id}")
                 return {"status": "not_found", "message": "Appointment not found."}

    except Exception as e:
        await db.rollback()
        logger.error(f"Error cancelling appointment {appointment_id}: {e}", exc_info=True)
        return {"status": "error", "message": "An unexpected error occurred during cancellation."}
