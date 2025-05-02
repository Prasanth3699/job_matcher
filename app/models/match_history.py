from datetime import datetime
import pytz
from sqlalchemy import (
    Column,
    Index,
    Integer,
    String,
    DateTime,
    JSON,
)

# Use JSONB if using PostgreSQL for better performance/indexing, otherwise use JSON
from sqlalchemy.dialects.postgresql import JSONB

# Import func for server-side default timestamps
from sqlalchemy.sql import func

# Ensure the Base import path is correct for your project structure
from app.db.base import Base  # Assuming your Base is here

# Define timezone if using Python-level defaults (though server default is often preferred)
IST = pytz.timezone("Asia/Kolkata")


class MatchHistory(Base):
    """
    SQLAlchemy model representing the 'match_history' table in the database.
    Stores records of matching attempts made by users.
    """

    __tablename__ = "match_history"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign Key to the User table (adjust 'users.id' if your table/column names differ)
    user_id = Column(Integer, index=True, nullable=False)

    # --- Match Input Information ---
    # Consider limiting string length for database efficiency
    resume_filename = Column(String(255), nullable=True)
    # Store job IDs as a JSON array
    job_ids = Column(JSON, nullable=False, default=[])
    # Store user preferences as a JSON object (JSONB recommended for PostgreSQL)
    preferences = Column(JSONB, nullable=False, default={})

    # --- Link to Parsed Resume ---
    parsed_resume_id = Column(Integer, nullable=True, index=True)

    # --- Match Output ---
    # Store the list of match result dictionaries as a JSON object/array (JSONB recommended for PostgreSQL)
    results = Column(JSONB, nullable=False, default=[])

    # --- Timestamp ---
    # Recommended: Database-level UTC timestamp default
    # created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    # Alternative: Python-level default using specific timezone (as you had)
    # Be aware of how your DB driver and DB itself handle timezone info from Python.
    created_at = Column(DateTime, default=lambda: datetime.now(IST), nullable=False)

    # --- Table Arguments (e.g., Indexes) ---
    __table_args__ = (
        # Example composite index for common queries
        Index("ix_match_history_user_id_created_at", "user_id", "created_at"),
        # Add other indexes or constraints here if needed
    )

    # Note: __init__ is typically not needed for SQLAlchemy declarative models.
    # SQLAlchemy handles attribute assignment automatically.
