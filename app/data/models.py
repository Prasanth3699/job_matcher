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
from .database import Base
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


class JobPosting(Base):
    """Job postings with optimized indexes"""

    __tablename__ = "job_postings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(String(100), index=True)  # External system reference
    job_title = Column(String(255), nullable=False, index=True)  # Title searches
    company_name = Column(String(255), nullable=False, index=True)  # Company searches
    job_type = Column(String(50), index=True)  # Filter by job type
    salary_min = Column(Float, index=True)  # Salary range queries
    salary_max = Column(Float, index=True)
    salary_currency = Column(String(3))
    experience_min = Column(Integer, index=True)  # Experience filters
    experience_max = Column(Integer)
    location = Column(String(255), index=True)  # Location-based searches
    description = Column(String(10000))
    apply_link = Column(String(512))
    posting_date = Column(DateTime, index=True)  # Time-based queries
    is_active = Column(Boolean, default=True, index=True)  # Active/inactive filter
    processed_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    feedback = relationship("Feedback", back_populates="job")

    # Composite indexes for common search patterns
    __table_args__ = (
        Index("idx_job_title_company", "job_title", "company_name"),  # Combined search
        Index("idx_job_location_type", "location", "job_type"),  # Location + type
        Index("idx_job_salary_range", "salary_min", "salary_max"),  # Salary queries
        Index(
            "idx_job_active_date", "is_active", "posting_date"
        ),  # Active jobs by date
    )


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


class ModelVersion(Base):
    """Model versions with optimized indexes"""

    __tablename__ = "model_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_name = Column(String(100), unique=True, index=True)  # Version lookups
    model_type = Column(String(50), index=True)  # Filter by model type
    storage_path = Column(String(512))
    metrics = Column(JSON)
    is_active = Column(Boolean, default=False, index=True)  # Active model filter
    description = Column(String(500))
    created_at = Column(
        DateTime, default=datetime.utcnow, index=True
    )  # Version timeline
    activated_at = Column(DateTime, index=True)  # Activation tracking

    # Composite index for deployment patterns
    __table_args__ = (
        Index(
            "idx_model_active_type", "is_active", "model_type"
        ),  # Active models by type
    )


class BackupLog(Base):
    """Backup logs with optimized indexes"""

    __tablename__ = "backup_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    backup_type = Column(String(50), index=True)  # Filter by backup type
    file_path = Column(String(512))
    file_size = Column(Integer, index=True)  # Size analysis
    status = Column(String(20), index=True)  # Status monitoring
    created_at = Column(DateTime, default=datetime.utcnow, index=True)  # Timeline
    completed_at = Column(DateTime, index=True)  # Completion tracking

    # Composite indexes for backup monitoring
    __table_args__ = (
        Index("idx_backup_status_date", "status", "created_at"),  # Status over time
        Index(
            "idx_backup_type_status", "backup_type", "status"
        ),  # Type-specific status
    )
