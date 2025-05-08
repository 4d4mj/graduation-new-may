from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyCookie
from typing import Optional, Dict, Any

from .auth import decode_access_token
from app.db.session import get_db_session
from app.db.crud.user import get_user

# Create a cookie-based security scheme
cookie_scheme = APIKeyCookie(name="session")

# List of paths that should be excluded from authentication checks
PUBLIC_PATHS = [
    "/auth/login",
    "/auth/register",
    "/auth/refresh",
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc"
]

async def verify_token_middleware(request: Request, call_next):
    """
    Middleware to check the session cookie and add the authenticated user to request state.
    This doesn't block unauthenticated requests, but just adds user info if authenticated.
    """
    # Skip authentication for public paths
    if any(request.url.path.startswith(public_path) for public_path in PUBLIC_PATHS):
        return await call_next(request)

    session_cookie = request.cookies.get("session")
    request.state.user = None

    if session_cookie:
        try:
            token_data = decode_access_token(session_cookie)
            request.state.user = {
                "user_id": token_data.get("sub"),
                "role": token_data.get("role")
            }
        except Exception:
            # Just continue without user info if token is invalid
            pass
    # Check for Authorization header if session cookie is not present
    if not session_cookie:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                token_data = decode_access_token(token)
                request.state.user = {
                    "user_id": token_data.get("sub"),
                    "role": token_data.get("role")
                }
            except Exception:
                # Continue without user info if token is invalid
                pass

    response = await call_next(request)
    return response

# FastAPI dependency for protected routes
def get_current_user(request: Request):
    """
    Dependency to use in FastAPI route functions that require authentication.
    This will raise an HTTPException if the user is not authenticated.
    """
    if not request.state.user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return request.state.user

def get_optional_user(request: Request):
    """
    Dependency to use in FastAPI route functions where authentication is optional.
    This will not raise an exception if user is not authenticated.
    """
    return request.state.user

def require_roles(roles: list):
    """
    Factory function to create a dependency that requires specific roles.
    Usage: @app.get("/admin", dependencies=[Depends(require_roles(["admin"]))])
    """
    def _require_roles(user: dict = Depends(get_current_user)):
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail="Not enough permissions")
        return user

    return _require_roles

# Get a database session dependency
async def get_db(request: Request):
    """Yield an async SQLAlchemy session (dependency)."""
    async for session in get_db_session(request):
        yield session

def db_user_dependency(db_session = Depends(get_db), current_user = Depends(get_current_user)):
    """
    Combined dependency that provides both the current user and the database user model.
    Useful when you need the full user database object, not just the JWT claims.
    """
    user_id = current_user["user_id"]
    db_user = get_user(db_session, user_id)

    if not db_user:
        raise HTTPException(status_code=404, detail="User not found in database")

    return db_user
