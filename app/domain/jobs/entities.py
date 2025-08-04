"""
Jobs domain entities representing job-related business objects.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from uuid import UUID, uuid4

from ..matching.value_objects import Skills, Experience, LocationPreference, SalaryExpectation


@dataclass
class JobRequirements:
    """
    Job requirements encapsulating skills, experience, and other criteria.
    """
    
    required_skills: Skills = field(default_factory=lambda: Skills([]))
    preferred_skills: Skills = field(default_factory=lambda: Skills([]))
    experience: Optional[Experience] = field(default=None)
    education_level: Optional[str] = field(default=None)
    certifications: List[str] = field(default_factory=list)
    
    # Salary information
    salary_min: Optional[float] = field(default=None)
    salary_max: Optional[float] = field(default=None)
    salary_currency: str = field(default="USD")
    
    # Additional requirements
    remote_allowed: bool = field(default=False)
    relocation_assistance: bool = field(default=False)
    travel_required: bool = field(default=False)
    security_clearance: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """Validate job requirements after initialization."""
        if self.salary_min and self.salary_min < 0:
            raise ValueError("Minimum salary cannot be negative")
        
        if self.salary_max and self.salary_min and self.salary_max < self.salary_min:
            raise ValueError("Maximum salary cannot be less than minimum salary")
        
        # Normalize currency
        if self.salary_currency:
            object.__setattr__(self, 'salary_currency', self.salary_currency.upper())
    
    def has_salary_info(self) -> bool:
        """Check if salary information is available."""
        return self.salary_min is not None or self.salary_max is not None
    
    def matches_salary_expectation(self, expectation: SalaryExpectation) -> bool:
        """Check if job salary matches candidate's expectation."""
        return expectation.matches_salary_range(self.salary_min, self.salary_max)
    
    def get_all_skills(self) -> Skills:
        """Get combined required and preferred skills."""
        all_skills = list(self.required_skills.skills) + list(self.preferred_skills.skills)
        return Skills(all_skills)
    
    def skill_priority_score(self, skill_name: str) -> float:
        """Get priority score for a skill (1.0 for required, 0.5 for preferred)."""
        if self.required_skills.has_skill(skill_name):
            return 1.0
        elif self.preferred_skills.has_skill(skill_name):
            return 0.5
        else:
            return 0.0


@dataclass
class JobDetails:
    """
    Job posting details and metadata.
    """
    
    job_title: str = field()
    company_name: str = field()
    department: Optional[str] = field(default=None)
    job_type: str = field(default="full-time")  # full-time, part-time, contract, internship
    
    # Location information
    location: str = field(default="")
    remote_policy: str = field(default="on-site")  # on-site, remote, hybrid
    
    # Job description and details
    description: str = field(default="")
    responsibilities: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    
    # Application details
    apply_link: Optional[str] = field(default=None)
    application_deadline: Optional[datetime] = field(default=None)
    contact_email: Optional[str] = field(default=None)
    
    # Company information
    company_size: Optional[str] = field(default=None)
    industry: Optional[str] = field(default=None)
    company_description: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """Validate job details after initialization."""
        if not self.job_title or not self.job_title.strip():
            raise ValueError("Job title cannot be empty")
        
        if not self.company_name or not self.company_name.strip():
            raise ValueError("Company name cannot be empty")
        
        # Normalize job type
        valid_job_types = ["full-time", "part-time", "contract", "internship", "temporary"]
        if self.job_type.lower() not in valid_job_types:
            object.__setattr__(self, 'job_type', "full-time")  # Default fallback
        
        # Normalize remote policy
        valid_remote_policies = ["on-site", "remote", "hybrid"]
        if self.remote_policy.lower() not in valid_remote_policies:
            object.__setattr__(self, 'remote_policy', "on-site")  # Default fallback
    
    def is_remote_friendly(self) -> bool:
        """Check if job allows remote work."""
        return self.remote_policy in ["remote", "hybrid"]
    
    def is_deadline_approaching(self, days_threshold: int = 7) -> bool:
        """Check if application deadline is approaching."""
        if not self.application_deadline:
            return False
        
        days_until_deadline = (self.application_deadline - datetime.utcnow()).days
        return 0 <= days_until_deadline <= days_threshold
    
    def get_summary(self) -> str:
        """Get a summary string of the job."""
        return f"{self.job_title} at {self.company_name} ({self.location})"


@dataclass
class Job:
    """
    Core job entity representing a job posting with all associated information.
    
    This entity encapsulates all job-related data and business logic
    for job postings within the matching system.
    """
    
    details: JobDetails
    requirements: JobRequirements
    id: UUID = field(default_factory=uuid4)
    source_id: Optional[str] = field(default=None)  # External system reference
    
    # Metadata
    posting_date: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = field(default=True)
    
    # Processing metadata
    processed_data: Optional[Dict[str, Any]] = field(default=None)
    embeddings: Optional[Dict[str, List[float]]] = field(default=None)
    search_keywords: List[str] = field(default_factory=list)
    
    # Analytics
    view_count: int = field(default=0)
    application_count: int = field(default=0)
    match_count: int = field(default=0)
    
    def __post_init__(self):
        """Validate job entity after initialization."""
        if self.posting_date > datetime.utcnow():
            raise ValueError("Posting date cannot be in the future")
    
    def deactivate(self) -> None:
        """Deactivate the job posting."""
        self.is_active = False
        self.last_updated = datetime.utcnow()
    
    def reactivate(self) -> None:
        """Reactivate the job posting."""
        self.is_active = True
        self.last_updated = datetime.utcnow()
    
    def update_details(self, new_details: JobDetails) -> None:
        """Update job details."""
        self.details = new_details
        self.last_updated = datetime.utcnow()
    
    def update_requirements(self, new_requirements: JobRequirements) -> None:
        """Update job requirements."""
        self.requirements = new_requirements
        self.last_updated = datetime.utcnow()
    
    def increment_view_count(self) -> None:
        """Increment the view count."""
        self.view_count += 1
    
    def increment_application_count(self) -> None:
        """Increment the application count."""
        self.application_count += 1
    
    def increment_match_count(self) -> None:
        """Increment the match count."""
        self.match_count += 1
    
    def is_recently_posted(self, days_threshold: int = 7) -> bool:
        """Check if job was posted recently."""
        days_since_posting = (datetime.utcnow() - self.posting_date).days
        return days_since_posting <= days_threshold
    
    def is_popular(self, view_threshold: int = 100) -> bool:
        """Check if job is popular based on view count."""
        return self.view_count >= view_threshold
    
    def get_application_rate(self) -> float:
        """Calculate application rate (applications per view)."""
        if self.view_count == 0:
            return 0.0
        return self.application_count / self.view_count
    
    def get_match_rate(self) -> float:
        """Calculate match rate (matches per view)."""
        if self.view_count == 0:
            return 0.0
        return self.match_count / self.view_count
    
    def matches_location_preference(self, preference: LocationPreference) -> bool:
        """Check if job matches location preference."""
        return preference.matches_location(self.details.location)
    
    def matches_salary_expectation(self, expectation: SalaryExpectation) -> bool:
        """Check if job matches salary expectation."""
        return self.requirements.matches_salary_expectation(expectation)
    
    def get_skill_overlap_score(self, candidate_skills: Skills) -> float:
        """Calculate skill overlap score with candidate."""
        job_skills = self.requirements.get_all_skills()
        if not job_skills:
            return 0.0
        
        matching_skills = candidate_skills.intersection(job_skills)
        return len(matching_skills) / len(job_skills)
    
    def add_search_keywords(self, keywords: List[str]) -> None:
        """Add search keywords for better discoverability."""
        # Normalize and deduplicate keywords
        normalized_keywords = [kw.lower().strip() for kw in keywords if kw.strip()]
        
        existing_keywords = set(self.search_keywords)
        new_keywords = [kw for kw in normalized_keywords if kw not in existing_keywords]
        
        self.search_keywords.extend(new_keywords)
        self.last_updated = datetime.utcnow()
    
    def set_embeddings(self, embeddings: Dict[str, List[float]]) -> None:
        """Set embeddings for semantic search."""
        self.embeddings = embeddings
        self.last_updated = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary representation."""
        return {
            "id": str(self.id),
            "source_id": self.source_id,
            "job_title": self.details.job_title,
            "company_name": self.details.company_name,
            "location": self.details.location,
            "job_type": self.details.job_type,
            "remote_policy": self.details.remote_policy,
            "description": self.details.description,
            "apply_link": self.details.apply_link,
            "salary_min": self.requirements.salary_min,
            "salary_max": self.requirements.salary_max,
            "required_skills": [skill.name for skill in self.requirements.required_skills.skills],
            "preferred_skills": [skill.name for skill in self.requirements.preferred_skills.skills],
            "posting_date": self.posting_date.isoformat(),
            "is_active": self.is_active,
            "view_count": self.view_count,
            "application_count": self.application_count,
            "match_count": self.match_count
        }
    
    def get_freshness_score(self) -> float:
        """Calculate freshness score based on posting date (1.0 = today, decreases over time)."""
        days_since_posting = (datetime.utcnow() - self.posting_date).days
        
        if days_since_posting <= 1:
            return 1.0
        elif days_since_posting <= 7:
            return 0.9
        elif days_since_posting <= 30:
            return 0.7
        elif days_since_posting <= 90:
            return 0.5
        else:
            return 0.3