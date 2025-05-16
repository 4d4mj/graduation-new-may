import logging 
from typing import List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import AllergyModel

logger = logging.getLogger(__name__)

async def get_allergies_for_patient(db: AsyncSession, patient_user_id: int) -> List[AllergyModel]:
    """
    Retrieves all recorded allergies for a specific patient

    Args:
        db (AsyncSession): the database session
        patient_user_id (int): the user_id of the patient (corresponds to the patients.user_id)

    Returns:
        List[AllergyModel]: list of AllergyModel objects
    """
    
    logger.debug(f"CRUD: fetching allergies for patient_user_id '{patient_user_id}'")
    
    stmt = select(AllergyModel).where(AllergyModel.patient_id == patient_user_id)
    
    result = await db.execute(stmt)
    allergies = result.scalars().all()
    
    logger.info(f"CRUD: found {len(allergies)} allergies for patient_user_id '{patient_user_id}'")
    return allergies