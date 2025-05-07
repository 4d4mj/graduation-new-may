from fastapi import APIRouter, Depends, HTTPException, status, Response, Cookie
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.auth import create_tokens_for_user
from app.db.models.user import UserModel
from app.config.settings import env, settings
from app.db.crud.auth import create_user, authenticate_user, get_user_from_token, refresh_user_token
from app.schemas.register_request import RegisterRequest
from app.schemas.login_request import LoginRequest
from app.schemas.auth_response import AuthResponse
from app.core.middleware import get_db
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
    tokens = await refresh_user_token(db, refresh_token)

    # Set HttpOnly cookies
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
    return await get_user_from_token(db, session)
