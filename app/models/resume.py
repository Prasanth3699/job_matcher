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


class Resume(Base):
    """User resumes with optimized indexes"""

    __tablename__ = "resumes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)  # FK index
    original_filename = Column(String(255), index=True)  # For filename searches
    file_path = Column(String(512))
    file_size = Column(Integer)
    parsed_data = Column(JSON)
    created_at = Column(
        DateTime, default=datetime.utcnow, index=True
    )  # Time-based queries

    user = relationship("User", back_populates="resumes")
    feedback = relationship("Feedback", back_populates="resume")

    # Composite index for common access patterns
    __table_args__ = (
        Index(
            "idx_resume_user_created", "user_id", "created_at"
        ),  # User's resume history
    )
