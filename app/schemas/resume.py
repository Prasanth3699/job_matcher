from datetime import date
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Any, Optional


class Job(BaseModel):
    id: int
    job_title: str
    company_name: Optional[str] = None
    job_type: Optional[str] = None
    salary: Optional[str] = None
    experience: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    detail_url: HttpUrl
    apply_link: HttpUrl
    posting_date: date


class ResumeJobsResponse(BaseModel):
    resume_content: str
    jobs: List[Job]
    user_id: Optional[int] = None


class MatchingSkills(BaseModel):
    """
    Model for representing matching skills
    """

    required: List[str] = Field(default_factory=list)
    preferred: List[str] = Field(default_factory=list)


class MatchScores(BaseModel):
    """
    Model for job match scores
    """

    skill_score: float
    experience_score: float
    domain_score: float
    final_score: float
    matching_skills: MatchingSkills


class JobMatch(BaseModel):
    """
    Model for job with match scores
    """

    job: Dict[str, Any]
    scores: MatchScores


class ResumeRecommendationResponse(BaseModel):
    """
    Response model for resume recommendations
    """

    resume_content: str
    recommendations: List[JobMatch]
    user_id: Optional[int] = None
