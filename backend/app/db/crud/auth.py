from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload
from fastapi import HTTPException, status
from jose import JWTError

from app.db.models.user import UserModel
from app.schemas.register_request import RegisterRequest
from app.schemas.login_request import LoginRequest
from app.schemas.shared import UserOut as User
from app.schemas.auth_response import AuthResponse
from app.core.auth import get_password_hash, verify_password, create_access_token, decode_access_token, create_tokens_for_user
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


async def get_user_by_id(db: AsyncSession, user_id: int) -> UserModel | None:
    """Get user by ID with eager loading of profiles."""
    user = await db.scalar(
        select(UserModel)
        .options(
            selectinload(UserModel.patient_profile),
            selectinload(UserModel.doctor_profile),
        )
        .where(UserModel.id == user_id)
    )
    return user


async def get_user_from_token(db: AsyncSession, token: str) -> User | None:
    """Validates token and returns user if valid."""
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = decode_access_token(token)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user = await get_user_by_id(db, int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return User.model_validate(user, from_attributes=True)


async def refresh_user_token(db: AsyncSession, refresh_token: str) -> AuthResponse:
    """Refreshes user tokens using a refresh token."""
    if not refresh_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing refresh token")
    try:
        payload = decode_access_token(refresh_token)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    user = await db.get(UserModel, int(payload.get("sub")))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    # Issue new tokens
    access_token = create_access_token(
        data={"sub": str(user.id), "role": user.role},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
    )
    new_refresh = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(days=settings.refresh_token_expire_days)
    )

    return AuthResponse(
        access_token=access_token,
        refresh_token=new_refresh,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60
    )
