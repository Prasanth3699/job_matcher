from sqlalchemy import Column, String, Integer, DateTime, JSON, Text, Float
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
from app.db.base import Base
import uuid


class MatchJob(Base):
    """Async match job tracking"""
    
    __tablename__ = "match_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Integer, nullable=False, index=True)  # Changed from UUID to Integer
    
    # Job details
    status = Column(String(50), nullable=False, default="pending", index=True)  # pending, processing, completed, failed
    task_id = Column(String(255), nullable=True, index=True)  # Celery task ID
    
    # Input data
    resume_filename = Column(String(255), nullable=False)
    job_ids = Column(JSON, nullable=False)  # List of job IDs to match against
    preferences = Column(JSON, nullable=False)  # User preferences
    
    # Results
    match_results = Column(JSON, nullable=True)  # Match results when completed
    parsed_resume_id = Column(Integer, nullable=True)  # ID from profile service
    error_message = Column(Text, nullable=True)  # Error details if failed
    
    # Progress tracking
    progress_percentage = Column(Float, default=0.0)
    current_step = Column(String(100), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)