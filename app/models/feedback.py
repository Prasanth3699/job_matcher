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


class Feedback(Base):
    """User feedback with optimized indexes"""

    __tablename__ = "feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)  # FK index
    job_id = Column(
        UUID(as_uuid=True), ForeignKey("job_postings.id"), index=True
    )  # FK index
    resume_id = Column(
        UUID(as_uuid=True), ForeignKey("resumes.id"), index=True
    )  # FK index
    feedback_type = Column(
        String(20), nullable=False, index=True
    )  # Feedback type filter
    match_score = Column(Float, index=True)  # Score analysis
    feedback_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)  # Time analysis

    user = relationship("User", back_populates="feedback")
    job = relationship("JobPosting", back_populates="feedback")
    resume = relationship("Resume", back_populates="feedback")

    # Composite indexes for analytics
    __table_args__ = (
        Index("idx_feedback_user_job", "user_id", "job_id"),  # User-job interactions
        Index(
            "idx_feedback_type_score", "feedback_type", "match_score"
        ),  # Feedback analysis
        Index("idx_feedback_job_type", "job_id", "feedback_type"),  # Job feedback stats
        Index(
            "idx_feedback_user_date", "user_id", "created_at"
        ),  # User activity over time
    )
