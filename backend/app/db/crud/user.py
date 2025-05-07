# app/db/crud/user.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from sqlalchemy.orm import selectinload
from typing import List, Optional, Dict, Any

from app.db.models.user import UserModel
from app.db.models.patient import PatientModel
from app.db.models.doctor import DoctorModel


async def get_users(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    role: Optional[str] = None
) -> List[UserModel]:
    """
    Get a list of users with optional filtering by role.

    Args:
        db: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return
        role: Filter by user role (optional)

    Returns:
        List of UserModel objects
    """
    query = select(UserModel).options(
        selectinload(UserModel.patient_profile),
        selectinload(UserModel.doctor_profile)
    )

    if role:
        query = query.where(UserModel.role == role)

    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()


async def get_user(db: AsyncSession, user_id: int) -> Optional[UserModel]:
    """
    Get a user by ID with their profiles loaded.

    Args:
        db: Database session
        user_id: User ID

    Returns:
        UserModel or None if not found
    """
    query = select(UserModel).options(
        selectinload(UserModel.patient_profile),
        selectinload(UserModel.doctor_profile)
    ).where(UserModel.id == user_id)

    result = await db.execute(query)
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[UserModel]:
    """
    Get a user by email with their profiles loaded.

    Args:
        db: Database session
        email: User's email address

    Returns:
        UserModel or None if not found
    """
    query = select(UserModel).options(
        selectinload(UserModel.patient_profile),
        selectinload(UserModel.doctor_profile)
    ).where(UserModel.email == email)

    result = await db.execute(query)
    return result.scalar_one_or_none()


async def search_users(
    db: AsyncSession,
    search_term: str,
    role: Optional[str] = None,
    limit: int = 10
) -> List[UserModel]:
    """
    Search for users by email or name (in patient/doctor profiles).

    Args:
        db: Database session
        search_term: Search term to look for in email or names
        role: Filter by user role (optional)
        limit: Maximum number of results

    Returns:
        List of matching UserModel objects
    """
    # Create search pattern for LIKE queries
    pattern = f"%{search_term}%"

    # Build the query with joins to profile tables
    query = select(UserModel).distinct().options(
        selectinload(UserModel.patient_profile),
        selectinload(UserModel.doctor_profile)
    )

    # Join with patient and doctor tables for name searching
    # This is equivalent to a LEFT JOIN in SQL
    query = query.outerjoin(PatientModel).outerjoin(DoctorModel)

    # Search conditions
    conditions = [
        UserModel.email.ilike(pattern),
        PatientModel.first_name.ilike(pattern),
        PatientModel.last_name.ilike(pattern),
        DoctorModel.first_name.ilike(pattern),
        DoctorModel.last_name.ilike(pattern)
    ]

    # Apply search conditions
    query = query.where(or_(*conditions))

    # Apply role filter if provided
    if role:
        query = query.where(UserModel.role == role)

    # Apply limit
    query = query.limit(limit)

    # Execute query
    result = await db.execute(query)
    return result.scalars().all()


async def get_user_count(db: AsyncSession, role: Optional[str] = None) -> int:
    """
    Get the total count of users, optionally filtered by role.

    Args:
        db: Database session
        role: Filter by user role (optional)

    Returns:
        Total count of users
    """
    query = select(func.count(UserModel.id))

    if role:
        query = query.where(UserModel.role == role)

    result = await db.execute(query)
    return result.scalar_one()


async def delete_user(db: AsyncSession, user_id: int) -> bool:
    """
    Delete a user by ID.

    Args:
        db: Database session
        user_id: User ID

    Returns:
        True if user was deleted, False if user wasn't found
    """
    user = await get_user(db, user_id)
    if not user:
        return False

    await db.delete(user)
    await db.commit()
    return True
