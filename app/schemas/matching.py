from pydantic import BaseModel
from typing import Dict, List, Any, Optional


class JobMatchResult(BaseModel):
    job_id: str
    overall_score: float
    score_breakdown: Dict[str, float]
    missing_skills: List[str]
    matching_skills: List[str]
    explanation: str
    job_details: Dict[str, Any]


class MatchErrorResponse(BaseModel):
    detail: str
    error_type: str
    context: Optional[str] = None
    solution: Optional[str] = None


class JobPostingRequest(BaseModel):
    # Define job posting request fields
    job_title: str
    company_name: str
    location: str
    job_type: str
    apply_link: str
    skills: List[str]
    description: str
