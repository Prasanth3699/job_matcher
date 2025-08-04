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
from datetime import datetime
from app.db.base import Base
import uuid


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
