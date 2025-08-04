from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from app.core.config import get_settings

settings = get_settings()

# ==============================
# Async engine/session (FastAPI)
# ==============================
# Preserve exported name `engine` for backward compatibility (imports in app.main etc.)
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=3600,
)

AsyncSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# Dependency for FastAPI (async)
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# ===================================
# Sync engine/session (Celery worker)
# ===================================
# Derive a sync DSN by replacing asyncpg with psycopg2 if needed
def _to_sync_dsn(dsn: str) -> str:
    # Typical async DSN: postgresql+asyncpg://user:pass@host:5432/db
    # Sync DSN:          postgresql+psycopg2://user:pass@host:5432/db
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
    return dsn


SYNC_DATABASE_URL = _to_sync_dsn(settings.DATABASE_URL)

sync_engine = create_engine(
    SYNC_DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=5,
    pool_timeout=30,
    pool_recycle=3600,
    future=True,
)

SyncSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=sync_engine,
    expire_on_commit=False,
)
