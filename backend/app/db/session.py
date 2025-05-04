from sqlalchemy.ext.asyncio import AsyncSession
from typing import Callable, AsyncGenerator, Optional
from fastapi import Request
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

# Global variable to hold the session factory
_global_session_factory: Optional[AsyncSession] = None

def set_global_session_factory(factory):
    """Sets the globally accessible session factory. Called once at startup."""
    global _global_session_factory
    _global_session_factory = factory
    logger.info("Global SQLAlchemy session factory has been set.")

# Session dependency for FastAPI routes
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

# Context manager for tools that need database access
@asynccontextmanager
async def tool_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Provides a DB session for tools/background tasks using the globally set factory.
    """
    global _global_session_factory
    if _global_session_factory is None:
        logger.error("Global session factory accessed by tool before being set.")
        raise RuntimeError("Database session factory not initialized globally.")

    session_factory = _global_session_factory
    async with session_factory() as session:
        try:
            yield session
        except Exception:
            logger.exception("Error occurred within tool_db_session context")
            raise
