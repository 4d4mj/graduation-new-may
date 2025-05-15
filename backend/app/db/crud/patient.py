import logging
from typing import List, Optional
from sqlalchemy import select, and_, or_, distinct
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import UserModel, PatientModel, AppointmentModel, DoctorModel

logger = logging.getLogger(__name__)


async def find_patients_by_name_and_verify_doctor_link(
    db: AsyncSession, full_name: str, requesting_doctor_id: int
) -> List[PatientModel]:
    """
    Finds patients by their full names and verifies they have an appointment link with the requesting doctor

    Args:
        db (AsyncSession): the database session
        full_name (str): the full name of the patient to search for
        requesting_doctor_id (int): the User_id of the doctor making the request

    Returns:
        List[PatientModel]: a list of PatientModel objects matching the criteria
    """

    logger.debug(
        f"CRUD: Searching for patient '{full_name}' for doctor_id '{requesting_doctor_id}'"
    )

    if not full_name or not full_name.strip():
        logger.warning(
            "CRUD: find_patients_by_name - full_name was empty or whitespace."
        )
        return []

    name_parts = full_name.strip().split(" ", 1)
    first_name_query = name_parts[0]
    last_name_query = name_parts[1] if len(name_parts) > 1 else None

    # Step 1: Find patients matching the name
    # We join with UserModel to ensure we are only looking at users with 'patient' role,
    # although PatientModel.user_id implies this via FK. Explicit check is safer.
    patient_query = (
        select(PatientModel)
        .join(UserModel, PatientModel.user_id == UserModel.id)
        .where(UserModel.role == "patient")
    )

    if last_name_query:
        # Search by first name and last name (case-insensitive)
        patient_query = patient_query.where(
            and_(
                PatientModel.first_name.ilike(f"%{first_name_query}%"),
                PatientModel.last_name.ilike(f"%{last_name_query}%"),
            )
        )
    else:
        # Search the single provided name part in both first_name and last_name
        patient_query = patient_query.where(
            or_(
                PatientModel.first_name.ilike(f"%{first_name_query}%"),
                PatientModel.last_name.ilike(f"%{first_name_query}%"),
            )
        )

    # Eager load the user relationship for PatientModel if you often need user.email etc.
    # patient_query = patient_query.options(selectinload(PatientModel.user)) # Optional

    possible_patients_result = await db.execute(patient_query)
    possible_patients = possible_patients_result.scalars().all()

    if not possible_patients:
        logger.info(f"CRUD: No patients found matching name '{full_name}'")
        return []

    logger.debug(
        f"CRUD: Found {len(possible_patients)} potential patients by name '{full_name}'. Verifying doctor link..."
    )

    accessible_patients: List[PatientModel] = []
    for patient in possible_patients:
        # Step 2: Verify doctor-patient link via the appointment table
        link_exists_query = (
            select(AppointmentModel.id)
            .where(
                and_(
                    AppointmentModel.patient_id == patient.user_id,
                    AppointmentModel.doctor_id == requesting_doctor_id,
                )
            )
            .limit(1)
        )
        link_exists_result = await db.execute(link_exists_query)
        if link_exists_result.scalar_one_or_none() is not None:
            accessible_patients.append(patient)
            logger.debug(
                f"CRUD: Patient user_id {patient.user_id} ({patient.first_name} {patient.last_name}) is linked to doctor {requesting_doctor_id}"
            )
        else:
            logger.debug(
                f"CRUD: Patient user_id {patient.user_id} ({patient.frsit_name} {patient.last_name}) is NOT linked to doctor {requesting_doctor_id}"
            )

    logger.info(
        f"CRUD: Found {len(accessible_patients)} accessible patients for doctor {requesting_doctor_id} matching name '{full_name}'"
    )
    return accessible_patients


async def get_patients_for_doctor(
    db: AsyncSession, requesting_doctor_id: int, limit: int = 100, offset: int = 0
) -> List[PatientModel]:
    """
    Retrieves all patients who have had an appointment with the specified doctor

    Args:
        db (AsyncSession): the database session
        requesting_doctor_id (int): the user_id of the doctor
        limit (int, optional): maximum number of patients to return
        offset (int, optional): number of patients to skip

    Returns:
        List[PatientModel]: a list of PatientModel objects
    """

    logger.debug(
        f"CRUD: Fetching patients for doctor_id {requesting_doctor_id} with limit {limit}, offset {offset}"
    )

    # Step 1: Find all unique patient_ids from appointments for this doctor
    # Using distinct to avoid fetching the same patient multiple times if they had multiple appointments.
    distinct_patient_ids_query = select(distinct(AppointmentModel.patient_id)).where(
        AppointmentModel.doctor_id == requesting_doctor_id
    )

    patient_ids_result = await db.execute(distinct_patient_ids_query)
    patient_user_ids = patient_ids_result.scalars().all()

    if not patient_user_ids:
        logger.info(
            f"CRUD: No patients found with appointments for doctor_id {requesting_doctor_id}"
        )
        return []

    # Step 2: Fetch patient details for these user_ids
    # We also join with UserModel to ensure role is 'patient', although PatientModel implies it.
    patients_query = (
        select(PatientModel)
        .join(UserModel, PatientModel.user_id == UserModel.id)
        .where(UserModel.role == "patient", PatientModel.user_id.in_(patient_user_ids))
        .order_by(PatientModel.last_name, PatientModel.first_name)
        .limit(limit)
        .offset(offset)
        .options(selectinload(PatientModel.user))
    )

    patients_result = await db.execute(patients_query)
    patients = patients_result.scalars().all()

    logger.info(
        f"CRUD: Found {len(patients)} patients for doctor_id '{requesting_doctor_id}'"
    )
    return patients
