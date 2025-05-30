from fastapi import APIRouter, Depends, HTTPException, Query, Request
from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.core.middleware import get_db
from app.db.models.appointment import AppointmentModel
from app.db.crud.appointment import (
    create_appointment,
    get_appointments,
    get_appointment,
    update_appointment,
    delete_appointment,
    get_doctor_availability
)
from app.core.middleware import get_current_user, require_roles

# Create schemas for appointment
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class AppointmentDoctorProfile(BaseModel):
    first_name: str
    last_name: str
    specialty: str

    class Config:
        from_attributes = True

class AppointmentPatientProfile(BaseModel): # New model for patient details
    first_name: str
    last_name: str
    # email: Optional[str] = None # Example: if you want to show patient email

    class Config:
        from_attributes = True

class AppointmentBase(BaseModel):
    doctor_id: int
    starts_at: datetime
    ends_at: datetime
    location: str
    notes: Optional[str] = None

class AppointmentCreate(AppointmentBase):
    pass

class AppointmentUpdate(BaseModel):
    doctor_id: Optional[int] = None
    doctor_profile: Optional[AppointmentDoctorProfile] = None
    starts_at: Optional[datetime] = None
    ends_at: Optional[datetime] = None
    location: Optional[str] = None
    notes: Optional[str] = None

class Appointment(AppointmentBase):
    id: int
    patient_id: int
    created_at: datetime
    doctor_profile: Optional[AppointmentDoctorProfile] = None
    patient_profile: Optional[AppointmentPatientProfile] = None # Added for doctor's view

    class Config:
        from_attributes = True

router = APIRouter(prefix="/appointments", tags=["appointments"])

@router.post("/", response_model=Appointment, status_code=201)
async def create_appointment_route(
    appointment: AppointmentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new appointment for the current user"""
    # Use the CRUD function
    result = await create_appointment(
        db=db,
        patient_id=int(current_user["user_id"]), # Cast to int
        doctor_id=appointment.doctor_id,
        starts_at=appointment.starts_at,
        ends_at=appointment.ends_at,
        location=appointment.location,
        notes=appointment.notes
    )

    # Check if result is a dict (error happened)
    if isinstance(result, dict) and "status" in result:
        status_code = 409 if result["status"] == "conflict" else 500
        raise HTTPException(status_code=status_code, detail=result["message"])

    return result

@router.get("/", response_model=List[Appointment])
async def get_appointments_route(
    request: Request,
    skip: int = 0,
    limit: int = 100,
    doctor_id: Optional[int] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get appointments based on filters"""
    return await get_appointments(
        db=db,
        user_id=int(current_user["user_id"]), # Cast to int
        role=current_user["role"],
        skip=skip,
        limit=limit,
        doctor_id=doctor_id,
        date_from=date_from,
        date_to=date_to
    )

@router.get("/{appointment_id}", response_model=Appointment)
async def get_appointment_route(
    appointment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get a specific appointment by ID"""
    return await get_appointment(
        db=db,
        appointment_id=appointment_id,
        user_id=int(current_user["user_id"]), # Cast to int
        role=current_user["role"]
    )

@router.put("/{appointment_id}", response_model=Appointment)
async def update_appointment_route(
    appointment_id: int,
    appointment_update: AppointmentUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update an existing appointment"""
    update_data = appointment_update.model_dump(exclude_unset=True)
    return await update_appointment(
        db=db,
        appointment_id=appointment_id,
        user_id=int(current_user["user_id"]), # Cast to int
        role=current_user["role"],
        update_data=update_data
    )

@router.delete("/{appointment_id}", status_code=204)
async def delete_appointment_route(
    appointment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete an appointment"""
    logger.info(f"Attempting to delete appointment {appointment_id} by user: {current_user}")
    await delete_appointment(
        db=db,
        appointment_id=appointment_id,
        user_id=int(current_user["user_id"]), # Cast to int
        role=current_user["role"]
    )
    return None

@router.get("/availability/{doctor_id}", response_model=List[datetime])
async def get_doctor_availability_route(
    doctor_id: int,
    date: datetime = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get available time slots for a doctor on a specific date"""
    return await get_doctor_availability(
        db=db,
        doctor_id=doctor_id,
        date=date
    )
