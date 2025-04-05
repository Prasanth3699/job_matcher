from typing import List, Dict, Optional
from pydantic import BaseModel, HttpUrl, Field
from datetime import datetime
from enum import Enum


class JobType(str, Enum):
    FULL_TIME = "Full Time"
    PART_TIME = "Part Time"
    INTERNSHIP = "Internship"
    REMOTE = "Remote"
    HYBRID = "Hybrid"
    OTHER = "Other"


class ResumeFormat(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class JobPostingRequest(BaseModel):
    """Schema for incoming job postings"""

    job_title: str
    company_name: str
    job_type: JobType
    salary: Optional[str] = None
    experience: Optional[str] = None
    location: str
    description: str
    apply_link: HttpUrl
    posting_date: Optional[datetime] = None


class ResumeUploadRequest(BaseModel):
    """Schema for resume upload"""

    file_format: ResumeFormat
    file_content: str = Field(..., description="Base64 encoded file content")
    preferred_job_types: List[JobType] = []
    preferred_locations: List[str] = []
    salary_expectation: Optional[str] = None
    target_title: Optional[str] = None
    preferred_companies: List[str] = []


class MatchResultResponse(BaseModel):
    """Schema for match results"""

    job_id: str
    job_title: str
    company_name: str
    overall_score: float
    score_breakdown: Dict[str, float]
    missing_skills: List[str]
    matching_skills: List[str]
    explanation: str
    apply_link: HttpUrl
    location: str
    job_type: JobType


class APIErrorResponse(BaseModel):
    """Schema for error responses"""

    detail: str
    error_type: str
    request_id: Optional[str] = None
