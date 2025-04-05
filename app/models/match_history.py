from datetime import datetime
from sqlalchemy import Column, Index, Integer, String, DateTime, JSON
from sqlalchemy.dialects.postgresql import JSONB
from ..db.base import Base
from pydantic import BaseModel, Field
from typing import List, Dict, Any


class MatchHistoryBase(BaseModel):
    user_id: int
    resume_filename: str
    job_ids: List[int]
    preferences: Dict[str, Any]
    results: List[Dict[str, Any]] = Field(default_factory=list)


class MatchHistoryCreate(MatchHistoryBase):
    user_id: int


class MatchHistoryUpdate(MatchHistoryBase):
    pass


class MatchHistoryBase(MatchHistoryBase):
    id: int
    results: List[Dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True

    @classmethod
    def model_validate(cls, obj):
        return cls(
            id=obj.id,
            user_id=obj.user_id,
            resume_filename=obj.resume_filename,
            job_ids=obj.job_ids,
            preferences=obj.preferences,
            results=obj.results,
            created_at=obj.created_at,
        )


class MatchHistory(Base):
    __tablename__ = "match_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    resume_filename = Column(String)
    job_ids = Column(JSON)  # Stores list of job IDs
    preferences = Column(JSONB)  # Stores the preferences JSON
    results = Column(JSONB)  # Stores the full results JSON
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_match_history_user_id_created_at", "user_id", "created_at"),
    )
