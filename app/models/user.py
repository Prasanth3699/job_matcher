from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Boolean,
    JSON,
    ForeignKey,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.base import Base
import uuid


class User(Base):
    """User accounts with optimized indexes"""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(
        String(255), unique=True, nullable=False, index=True
    )  # Index for login lookups
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), index=True)  # Index for name searches
    created_at = Column(
        DateTime, default=datetime.utcnow, index=True
    )  # Index for time-based queries
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    resumes = relationship("Resume", back_populates="user")
    feedback = relationship("Feedback", back_populates="user")

    # Composite index for common query patterns
    __table_args__ = (
        Index(
            "idx_user_email_status", "email", "hashed_password"
        ),  # For authentication
    )
