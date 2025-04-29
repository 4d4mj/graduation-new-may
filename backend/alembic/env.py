# alembic/env.py
from logging.config import fileConfig
import asyncio

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.base import Base          # import your MetaData
import app.db.models
from app.config.settings import settings

#####################################################################
# 1.  URLs
#####################################################################

ASYNC_URL = str(settings.database_url)                  # postgresql+asyncpg://...
SYNC_URL  = ASYNC_URL.replace("+asyncpg", "")           # postgresql://...

config = context.config
config.set_main_option("sqlalchemy.url", SYNC_URL)      # needed for *offline* mode

#####################################################################
# 2.  Logging
#####################################################################

fileConfig(config.config_file_name)

#####################################################################
# 3.  Metadata for 'autogenerate'
#####################################################################

target_metadata = Base.metadata

#####################################################################
# 4.  Offline migrations (no DB connection)
#####################################################################

def run_migrations_offline() -> None:
    context.configure(
        url=SYNC_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

#####################################################################
# 5.  Online migrations (async connection)
#####################################################################

def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()

async def run_migrations_online() -> None:
    engine = create_async_engine(
        ASYNC_URL,
        poolclass=pool.NullPool,
        future=True,
    )
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()

#####################################################################
# 6.  Entrypoint
#####################################################################

if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
