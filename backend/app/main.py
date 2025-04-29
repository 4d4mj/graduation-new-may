from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from .config.settings import settings
from fastapi.middleware.cors import CORSMiddleware
from .db.base import get_engine, Base
from .db.session import get_db_session

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL not set.")

    # Use asyncpg driver as specified in DATABASE_URL
    db_url = str(settings.database_url)
    engine = await get_engine(db_url)
    app.state.engine = engine

    yield

    # Clean up
    await engine.dispose()

app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(o) for o in settings.cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database dependency
async def get_db():
    async for session in get_db_session(str(settings.database_url)):
        yield session

# Include routers
from .routes.auth.router import router as auth_router
app.include_router(auth_router)
from .routes.chat.router import router as chat_router
app.include_router(chat_router)


