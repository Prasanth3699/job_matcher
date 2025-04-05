# resume_matcher/core/learning/models.py
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel
from dataclasses import dataclass


class FeedbackType(str, Enum):
    """Types of user feedback"""

    IMPRESSION = "impression"  # Job was shown to user
    CLICK = "click"  # User clicked on job
    APPLICATION = "apply"  # User applied to job
    REJECTION = "reject"  # User rejected the recommendation
    HIRED = "hired"  # User was hired for this job


class UserFeedback(BaseModel):
    """Schema for storing user feedback"""

    feedback_id: str
    user_id: str
    job_id: str
    resume_id: str
    feedback_type: FeedbackType
    timestamp: datetime
    match_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelVersion:
    """Track model versions and performance"""

    version_id: str
    creation_date: datetime
    metrics: Dict[str, float]
    is_active: bool
    description: Optional[str] = None
