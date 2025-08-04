"""
Matching domain entities representing core business objects.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
from uuid import UUID, uuid4

from .value_objects import Skills, Experience, Score, MatchConfidence


class MatchStatus(Enum):
    """Status of a match job"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Match:
    """
    Core matching entity representing a resume-to-job match operation.
    
    This entity encapsulates the business logic for tracking match jobs
    and their lifecycle through the matching process.
    """
    
    user_id: int
    resume_filename: str
    job_ids: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    status: MatchStatus = field(default=MatchStatus.PENDING)
    
    # Processing state
    task_id: Optional[str] = field(default=None)
    progress_percentage: float = field(default=0.0)
    current_step: Optional[str] = field(default=None)
    
    # Results
    match_results: Optional[List['MatchResult']] = field(default=None)
    parsed_resume_id: Optional[int] = field(default=None)
    error_message: Optional[str] = field(default=None)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = field(default=None)
    completed_at: Optional[datetime] = field(default=None)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def start_processing(self, task_id: str) -> None:
        """Mark the match as started with processing task ID."""
        if self.status != MatchStatus.PENDING:
            raise ValueError(f"Cannot start processing from status: {self.status}")
        
        self.status = MatchStatus.PROCESSING
        self.task_id = task_id
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def update_progress(self, percentage: float, step: str) -> None:
        """Update the processing progress."""
        if self.status != MatchStatus.PROCESSING:
            raise ValueError(f"Cannot update progress for status: {self.status}")
        
        if not 0 <= percentage <= 100:
            raise ValueError("Progress percentage must be between 0 and 100")
        
        self.progress_percentage = percentage
        self.current_step = step
        self.updated_at = datetime.utcnow()
    
    def complete_successfully(self, results: List['MatchResult'], parsed_resume_id: int) -> None:
        """Mark the match as completed with results."""
        if self.status != MatchStatus.PROCESSING:
            raise ValueError(f"Cannot complete from status: {self.status}")
        
        self.status = MatchStatus.COMPLETED
        self.match_results = results
        self.parsed_resume_id = parsed_resume_id
        self.progress_percentage = 100.0
        self.current_step = "Completed"
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def fail_with_error(self, error_message: str) -> None:
        """Mark the match as failed with error details."""
        if self.status not in [MatchStatus.PENDING, MatchStatus.PROCESSING]:
            raise ValueError(f"Cannot fail from status: {self.status}")
        
        self.status = MatchStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def is_processing(self) -> bool:
        """Check if the match is currently being processed."""
        return self.status == MatchStatus.PROCESSING
    
    def is_completed(self) -> bool:
        """Check if the match has completed successfully."""
        return self.status == MatchStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if the match has failed."""
        return self.status == MatchStatus.FAILED
    
    def get_processing_duration(self) -> Optional[float]:
        """Get the processing duration in seconds, if applicable."""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()


@dataclass
class MatchResult:
    """
    Individual job match result containing scoring and analysis.
    
    This entity represents the outcome of matching a resume against
    a specific job posting with detailed scoring and explanations.
    """
    
    job_id: str
    overall_score: Score
    confidence: MatchConfidence
    original_job_id: Optional[int] = field(default=None)
    
    # Skill analysis
    matching_skills: Skills = field(default_factory=lambda: Skills([]))
    missing_skills: Skills = field(default_factory=lambda: Skills([]))
    
    # Detailed scoring
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    explanation: str = field(default="")
    
    # Job details
    job_title: str = field(default="")
    company_name: str = field(default="")
    location: str = field(default="")
    job_type: str = field(default="")
    apply_link: Optional[str] = field(default=None)
    
    # Ranking information
    rank_position: Optional[int] = field(default=None)
    diversity_score: Optional[float] = field(default=None)
    
    def __post_init__(self):
        """Validate the match result after initialization."""
        if not 0 <= self.overall_score.value <= 1:
            raise ValueError("Overall score must be between 0 and 1")
        
        if not 0 <= self.confidence.value <= 1:
            raise ValueError("Confidence must be between 0 and 1")
    
    def is_high_quality_match(self, threshold: float = 0.7) -> bool:
        """Determine if this is a high-quality match based on score."""
        return self.overall_score.value >= threshold
    
    def is_confident_match(self, threshold: float = 0.8) -> bool:
        """Determine if this is a confident match based on confidence score."""
        return self.confidence.value >= threshold
    
    def get_skill_match_ratio(self) -> float:
        """Calculate the ratio of matching skills to total required skills."""
        total_skills = len(self.matching_skills.skills) + len(self.missing_skills.skills)
        if total_skills == 0:
            return 0.0
        return len(self.matching_skills.skills) / total_skills
    
    def add_score_component(self, component: str, score: float) -> None:
        """Add a component to the score breakdown."""
        if not 0 <= score <= 1:
            raise ValueError(f"Score component {component} must be between 0 and 1")
        
        self.score_breakdown[component] = score
    
    def get_weighted_score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted score based on component weights."""
        if not self.score_breakdown:
            return self.overall_score.value
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for component, score in self.score_breakdown.items():
            weight = weights.get(component, 1.0)
            total_weighted_score += score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert match result to dictionary representation."""
        return {
            "job_id": self.job_id,
            "original_job_id": self.original_job_id,
            "overall_score": self.overall_score.value,
            "confidence": self.confidence.value,
            "matching_skills": self.matching_skills.skills,
            "missing_skills": self.missing_skills.skills,
            "score_breakdown": self.score_breakdown,
            "explanation": self.explanation,
            "job_details": {
                "job_title": self.job_title,
                "company_name": self.company_name,
                "location": self.location,
                "job_type": self.job_type,
                "apply_link": self.apply_link,
            },
            "rank_position": self.rank_position,
            "diversity_score": self.diversity_score,
        }