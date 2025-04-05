from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class MatchHistory(BaseModel):
    id: int
    user_id: int
    resume_filename: Optional[str] = None
    job_ids: List[int] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    class Config:
        from_attributes = True  # This enables ORM mode

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
