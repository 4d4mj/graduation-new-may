from fastapi import APIRouter, Depends, HTTPException, status, Response, Cookie
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select
from app.core.auth import decode_access_token, create_access_token
from app.db.models.user import UserModel
from app.config.settings import env, settings
from app.routes.auth.services import create_user, authenticate_user, create_tokens_for_user
from app.schemas.register_request import RegisterRequest
from app.schemas.login_request import LoginRequest
from app.schemas.auth_response import AuthResponse
from app.main import get_db
from jose import JWTError
from datetime import timedelta
from app.schemas.shared import UserOut as User

router = APIRouter(
    prefix="/auth",
    tags=["auth"],
)

# determine secure flag
secure_cookie = env == "production"

@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: RegisterRequest,
    response: Response,
    db: AsyncSession = Depends(get_db)
):
    new_user = await create_user(db, user_data)
    tokens = create_tokens_for_user(new_user)
    # Set HttpOnly cookie
    response.set_cookie(
        key="session",
        value=tokens.access_token,
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        max_age=settings.access_token_expire_minutes * 60
    )
    response.set_cookie(
        key="refresh",
        value=tokens.refresh_token,
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        max_age=settings.refresh_token_expire_days * 86400
    )
    return tokens

@router.post("/login", response_model=AuthResponse)
async def login(
    login_data: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db)
):
    user = await authenticate_user(db, login_data)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    tokens = create_tokens_for_user(user)
    # Set HttpOnly cookie
    response.set_cookie(
        key="session",
        value=tokens.access_token,
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        max_age=settings.access_token_expire_minutes * 60
    )
    response.set_cookie(
        key="refresh",
        value=tokens.refresh_token,
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        max_age=settings.refresh_token_expire_days * 86400
    )
    return tokens

@router.post("/refresh", response_model=AuthResponse)
async def refresh(
    response: Response,
    refresh_token: str = Cookie(None),
    db: AsyncSession = Depends(get_db)
):
    if not refresh_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing refresh token")
    try:
        payload = decode_access_token(refresh_token)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
    user = await db.get(UserModel, int(payload.get("sub")))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    # issue new tokens
    access_token = create_access_token(
        data={"sub": str(user.id), "role": user.role},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
    )
    new_refresh = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(days=settings.refresh_token_expire_days)
    )
    response.set_cookie(
        key="session",
        value=access_token,
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        max_age=settings.access_token_expire_minutes * 60
    )
    response.set_cookie(
        key="refresh",
        value=new_refresh,
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        max_age=settings.refresh_token_expire_days * 86400
    )
    return AuthResponse(
        access_token=access_token,
        refresh_token=new_refresh,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60
    )

@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response):
    response.delete_cookie(key="session")
    response.delete_cookie(key="refresh")
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.get("/me", response_model=User)
async def me(
    session: str = Cookie(None),  # bind the "session" cookie
    db: AsyncSession = Depends(get_db)
):
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = decode_access_token(session)  # decode the JWT token to get user info
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = await db.scalar(                          # ② scalar() → single row
        select(UserModel)
        .options(
            selectinload(UserModel.patient_profile), # eager load relations
            selectinload(UserModel.doctor_profile),
        )
        .where(UserModel.id == int(payload["sub"]))
    )
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return User.model_validate(user, from_attributes=True)
