from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status

from app.db.models.user import UserModel
from app.schemas.register_request import RegisterRequest
from app.schemas.login_request import LoginRequest
from app.schemas.shared import UserOut as User
from app.schemas.auth_response import AuthResponse
from app.core.auth import get_password_hash, verify_password, create_access_token
from app.config.settings import settings
from datetime import timedelta

# app/routes/auth/services.py
from app.db.models.patient import PatientModel
from app.db.models.doctor  import DoctorModel

async def create_user(db: AsyncSession, data: RegisterRequest) -> User:
    """Insert user and its profile in one transaction."""
    hashed = get_password_hash(data.password)
    role = data.role or ("doctor" if data.doctor_profile else "patient")
    user = UserModel(email=data.email, password_hash=hashed, role=role)
    db.add(user)

    if data.patient_profile:
        db.add(PatientModel(user=user, **data.patient_profile.model_dump()))
    if data.doctor_profile:
        db.add(DoctorModel(user=user, **data.doctor_profile.model_dump()))

    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Email already registered")

    await db.refresh(user)       # refresh brings patient/doctor relationship too
    return User.model_validate(user, from_attributes=True)


async def authenticate_user(db: AsyncSession, login_data: LoginRequest) -> UserModel | None:
    result = await db.execute(select(UserModel).where(UserModel.email == login_data.email))
    user = result.scalar_one_or_none()
    if not user:
        return None
    if not verify_password(login_data.password, user.password_hash):
        return None
    return user


def create_tokens_for_user(user: UserModel) -> AuthResponse:
    access_token = create_access_token(
        {"sub": str(user.id), "role": user.role},
        timedelta(minutes=settings.access_token_expire_minutes)
    )
    refresh_token = create_access_token(
        {"sub": str(user.id)},
        timedelta(days=settings.refresh_token_expire_days),
    )
    return AuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
    )
