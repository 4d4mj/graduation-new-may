from datetime import datetime, timedelta, timezone
from typing import Optional

from passlib.context import CryptContext
from jose import JWTError, jwt
from app.schemas.auth_response import AuthResponse
from app.db.models.user import UserModel

from app.config.settings import settings

# Update the CryptContext initialization to explicitly set bcrypt backend options
# This will suppress the warning about '__about__' attribute
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__ident="2b",  # Explicitly set the bcrypt identifier
)
oauth2_scheme = None  # placeholder for Dependency injection


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        return payload
    except JWTError as e:
        raise e


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
