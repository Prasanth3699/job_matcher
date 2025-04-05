from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from app.core.config import get_settings

settings = get_settings()

# Create the database engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,  # Check connections before using them
    pool_size=20,  # Maintain 20 connections by default
    max_overflow=10,  # Allow 10 overflow connections under load
    pool_timeout=30,  # Wait 30 seconds before giving up on getting a connection
    pool_recycle=3600,  # Recycle connections after 1 hour
)

# Session factory - use scoped_session for thread safety
SessionLocal = scoped_session(
    sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False,  # Better for async operations
    )
)


def get_db():
    """
    Dependency that yields a DB session.
    Ensures the session is closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
