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
