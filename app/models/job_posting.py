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
