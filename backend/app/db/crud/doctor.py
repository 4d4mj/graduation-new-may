import logging
from typing import Optional, List

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.db.models import DoctorModel, UserModel

logger = logging.getLogger(__name__)

async def get_doctor_details_by_user_id(db: AsyncSession, doctor_user_id: int) -> Optional[DoctorModel]:
    """Fetches doctor profile details using the user_id."""
    logger.debug(f"Fetching doctor details for user_id: {doctor_user_id}")
    try:
        # Ensure we are selecting a user who is actually a doctor
        stmt = (
            select(DoctorModel)
            .join(DoctorModel.user) # Join with UserModel to potentially check role if needed
            .where(DoctorModel.user_id == doctor_user_id)
            .where(UserModel.role == 'doctor') # Explicitly check the role on the user model
            .options(joinedload(DoctorModel.user)) # Eager load user if needed elsewhere
            .limit(1)
        )
        result = await db.execute(stmt)
        doctor = result.scalar_one_or_none()

        if doctor:
            logger.info(f"Found doctor details for user_id {doctor_user_id}: Dr. {doctor.first_name} {doctor.last_name}")
            return doctor
        else:
            logger.warning(f"No doctor found with user_id {doctor_user_id} or user is not assigned 'doctor' role.")
            return None
    except Exception as e:
        logger.error(f"Error fetching doctor details for user_id {doctor_user_id}: {e}", exc_info=True)
        return None

# Optional: Add a function to list all doctors if needed by the agent later
async def list_all_doctors(db: AsyncSession) -> List[DoctorModel]:
    """Fetches all doctor profiles."""
    logger.debug("Fetching list of all doctors.")
    try:
        stmt = select(DoctorModel).options(joinedload(DoctorModel.user))
        result = await db.execute(stmt)
        doctors = result.scalars().all()
        logger.info(f"Found {len(doctors)} doctors.")
        return list(doctors)
    except Exception as e:
        logger.error(f"Error fetching list of doctors: {e}", exc_info=True)
        return []

async def get_doctor_by_name(db: AsyncSession, name: str) -> Optional[DoctorModel]:
    """
    Fetches a doctor by their first or last name (partial match).

    Args:
        db: Database session
        name: First name, last name, or partial name to search for

    Returns:
        DoctorModel if found, None otherwise
    """
    logger.debug(f"Searching for doctor with name: {name}")
    try:
        # Clean up the name - remove extra spaces and make case-insensitive
        search_name = name.strip().lower()

        # Create a query with name filter
        stmt = (
            select(DoctorModel)
            .join(DoctorModel.user)  # Join with UserModel
            .where(UserModel.role == 'doctor')  # Ensure it's a doctor
            .where(
                or_(
                    DoctorModel.first_name.ilike(f"%{search_name}%"),
                    DoctorModel.last_name.ilike(f"%{search_name}%")
                )
            )
            .options(joinedload(DoctorModel.user))
            .limit(1)  # Get the first match
        )

        result = await db.execute(stmt)
        doctor = result.scalar_one_or_none()

        if doctor:
            logger.info(f"Found doctor with name '{name}': Dr. {doctor.first_name} {doctor.last_name} (ID: {doctor.user_id})")
            return doctor
        else:
            logger.warning(f"No doctor found with name '{name}'")
            return None

    except Exception as e:
        logger.error(f"Error searching for doctor with name '{name}': {e}", exc_info=True)
        return None


async def find_doctors_by_name(db: AsyncSession, name: str) -> List[DoctorModel]:
    """
    Searches for doctors by their first or last name (partial match).
    Returns a list of matching doctors.

    Args:
        db: Database session
        name: First name, last name, or partial name to search for

    Returns:
        List of DoctorModel objects that match the search criteria
    """
    logger.debug(f"Searching for doctors with name: {name}")
    try:
        # Clean up the name - remove extra spaces and make case-insensitive
        search_name = name.strip().lower()

        # Create a query with name filter
        stmt = (
            select(DoctorModel)
            .join(DoctorModel.user)  # Join with UserModel
            .where(UserModel.role == 'doctor')  # Ensure it's a doctor
            .where(
                or_(
                    DoctorModel.first_name.ilike(f"%{search_name}%"),
                    DoctorModel.last_name.ilike(f"%{search_name}%")
                )
            )
            .options(joinedload(DoctorModel.user))
            .limit(5)  # Limit to top 5 matches to avoid overwhelming results
        )

        result = await db.execute(stmt)
        doctors = result.scalars().all()

        logger.info(f"Found {len(doctors)} doctors matching '{name}'")
        return list(doctors)

    except Exception as e:
        logger.error(f"Error searching for doctors with name '{name}': {e}", exc_info=True)
        return []
