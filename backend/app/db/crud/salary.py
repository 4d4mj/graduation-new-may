import logging
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload, joinedload  # Add joinedload

from app.db.models.doctor_salary import DoctorSalaryModel
from app.db.models.user import UserModel  # To fetch doctor's name
from app.db.models.doctor import DoctorModel  # To fetch doctor's name

logger = logging.getLogger(__name__)


async def get_doctor_financial_summary_by_user_id(
    db: AsyncSession, doctor_user_id: int
) -> Optional[DoctorSalaryModel]:
    """
    Retrieves the financial summary for a specific doctor by their user ID.
    Eagerly loads related user and doctor profile information for context if needed by the tool.
    """
    logger.debug(
        f"CRUD: Fetching financial summary for doctor_user_id: {doctor_user_id}"
    )
    try:
        stmt = (
            select(DoctorSalaryModel)
            .where(DoctorSalaryModel.doctor_user_id == doctor_user_id)
            .options(
                # Eager load the user associated with this salary entry,
                # then their doctor_profile to get the name easily.
                selectinload(
                    DoctorSalaryModel.user
                ).selectinload(  # User related to salary entry
                    UserModel.doctor_profile
                )  # Doctor profile of that user
            )
        )
        result = await db.execute(stmt)
        financial_summary = result.scalar_one_or_none()

        if financial_summary:
            logger.info(
                f"CRUD: Found financial summary for doctor_user_id: {doctor_user_id}"
            )
            # Example of accessing related data (the tool will do more complex formatting)
            if financial_summary.user and financial_summary.user.doctor_profile:
                logger.debug(
                    f"  Doctor Name: {financial_summary.user.doctor_profile.first_name} {financial_summary.user.doctor_profile.last_name}"
                )
        else:
            logger.info(
                f"CRUD: No financial summary found for doctor_user_id: {doctor_user_id}"
            )

        return financial_summary
    except Exception as e:
        logger.error(
            f"CRUD: Error fetching financial summary for doctor_user_id {doctor_user_id}: {e}",
            exc_info=True,
        )
        return None
