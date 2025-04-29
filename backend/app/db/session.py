from sqlalchemy.ext.asyncio import AsyncSession
from app.db.base import get_engine, get_session_factory
from typing import Callable, AsyncGenerator

# Session dependency
async def get_db_session(database_url: str) -> AsyncGenerator[AsyncSession, None]:
    """
    Create and yield a database session
    This will be used as a FastAPI dependency
    """
    engine = await get_engine(database_url)
    session_factory = await get_session_factory(engine)

    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
