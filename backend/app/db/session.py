from sqlalchemy.ext.asyncio import AsyncSession
from app.db.base import get_engine, get_session_factory
from typing import Callable, AsyncGenerator
from fastapi import Request

# Session dependency
async def get_db_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """
    Create and yield a database session using the shared engine
    This will be used as a FastAPI dependency
    """
    async_session = request.app.state.session_factory

    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
