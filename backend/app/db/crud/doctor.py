import logging
from typing import Optional, List, Dict, Any, Union

from sqlalchemy import select, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.db.models import DoctorModel, UserModel

logger = logging.getLogger(__name__)

async def find_doctors(
    db: AsyncSession,
    doctor_id: Optional[int] = None,
    name: Optional[str] = None,
    specialty: Optional[str] = None,
    limit: int = 5,
    return_single: bool = False
) -> Union[Optional[DoctorModel], List[DoctorModel]]:
    """
    Unified function to find doctors by various criteria.

    Args:
        db: Database session
        doctor_id: Optional ID of the doctor to find
        name: Optional name (first, last, or full) to search for
        specialty: Optional specialty to filter by
        limit: Maximum number of doctors to return (default 5)
        return_single: If True, returns a single doctor or None, otherwise returns a list

    Returns:
        If return_single is True: A single DoctorModel if found, None otherwise
        If return_single is False: A list of DoctorModel objects that match the criteria
    """
    logger.debug(f"Searching for doctors with criteria: id={doctor_id}, name='{name}', specialty='{specialty}'")

    try:
        # Start building the query
        query = select(DoctorModel).join(DoctorModel.user).where(UserModel.role == 'doctor')

        # If doctor_id is provided, it takes precedence over other search criteria
        if doctor_id is not None:
            query = query.where(DoctorModel.user_id == doctor_id)
        elif name:
            # Clean up the name - remove extra spaces and make case-insensitive
            search_name = name.strip().lower()

            # Split the name to handle full name searches
            name_parts = search_name.split()

            if len(name_parts) >= 2:
                # This might be a full name, try to match first and last name together
                first_part = name_parts[0]
                last_part = name_parts[-1]  # Use the last word as last name

                # Try exact match on first+last name first with higher priority
                query = query.where(
                    or_(
                        # Exact match on first and last name (highest priority)
                        and_(
                            DoctorModel.first_name.ilike(first_part),
                            DoctorModel.last_name.ilike(last_part)
                        ),
                        # Partial match on both first and last name (medium priority)
                        and_(
                            DoctorModel.first_name.ilike(f"%{first_part}%"),
                            DoctorModel.last_name.ilike(f"%{last_part}%")
                        ),
                        # Fallback to matching either first or last name (lowest priority)
                        or_(
                            DoctorModel.first_name.ilike(f"%{search_name}%"),
                            DoctorModel.last_name.ilike(f"%{search_name}%")
                        )
                    )
                )
            else:
                # Single word search - match against first or last name
                query = query.where(
                    or_(
                        DoctorModel.first_name.ilike(f"%{search_name}%"),
                        DoctorModel.last_name.ilike(f"%{search_name}%")
                    )
                )

        # Apply specialty filter if provided
        if specialty:
            query = query.where(DoctorModel.specialty.ilike(f"%{specialty}%"))

        # Always load the related user model
        query = query.options(joinedload(DoctorModel.user))

        # Apply limit
        query = query.limit(limit)

        # Execute the query
        result = await db.execute(query)

        if return_single:
            doctor = result.scalar_one_or_none()
            if doctor:
                logger.info(f"Found doctor: {doctor.first_name} {doctor.last_name} (ID: {doctor.user_id})")
            else:
                logger.warning(f"No doctor found matching criteria: id={doctor_id}, name='{name}', specialty='{specialty}'")
            return doctor
        else:
            doctors = list(result.scalars().all())
            logger.info(f"Found {len(doctors)} doctors matching criteria")
            return doctors

    except Exception as e:
        logger.error(f"Error searching for doctors: {e}", exc_info=True)
        return None if return_single else []

# Backward compatibility functions
async def get_doctor_details_by_user_id(db: AsyncSession, doctor_user_id: int) -> Optional[DoctorModel]:
    """Fetches doctor profile details using the user_id."""
    return await find_doctors(db, doctor_id=doctor_user_id, return_single=True)

async def get_doctor_by_name(db: AsyncSession, name: str) -> Optional[DoctorModel]:
    """Fetches a doctor by their name (full name or partial match)."""
    return await find_doctors(db, name=name, return_single=True)

async def find_doctors_by_name(db: AsyncSession, name: str) -> List[DoctorModel]:
    """Searches for doctors by their name (partial match)."""
    return await find_doctors(db, name=name, return_single=False)

async def list_all_doctors(db: AsyncSession) -> List[DoctorModel]:
    """Fetches all doctor profiles."""
    # Use a higher limit to get all doctors
    return await find_doctors(db, limit=100, return_single=False)
