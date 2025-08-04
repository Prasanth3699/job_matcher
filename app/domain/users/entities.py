"""
Users domain entities representing user and resume-related business objects.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from uuid import UUID, uuid4

from ..matching.value_objects import Skills, Experience, LocationPreference, SalaryExpectation


@dataclass
class UserProfile:
    """
    User profile containing career preferences and goals.
    """
    
    # Career preferences
    desired_job_titles: List[str] = field(default_factory=list)
    desired_industries: List[str] = field(default_factory=list)
    career_level: Optional[str] = field(default=None)
    
    # Location and work preferences
    location_preference: Optional[LocationPreference] = field(default=None)
    remote_work_preference: str = field(default="flexible")  # flexible, required, not_preferred
    relocation_willingness: bool = field(default=False)
    
    # Salary expectations
    salary_expectation: Optional[SalaryExpectation] = field(default=None)
    
    # Availability
    availability: str = field(default="active")  # active, passive, not_looking
    start_date: Optional[datetime] = field(default=None)
    
    # Additional preferences
    company_size_preference: Optional[str] = field(default=None)  # startup, medium, large, enterprise
    work_culture_preferences: List[str] = field(default_factory=list)
    benefits_priorities: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate user profile after initialization."""
        valid_remote_preferences = ["flexible", "required", "not_preferred"]
        if self.remote_work_preference not in valid_remote_preferences:
            object.__setattr__(self, 'remote_work_preference', "flexible")
        
        valid_availability = ["active", "passive", "not_looking"]
        if self.availability not in valid_availability:
            object.__setattr__(self, 'availability', "active")
    
    def is_actively_looking(self) -> bool:
        """Check if user is actively looking for jobs."""
        return self.availability == "active"
    
    def is_open_to_opportunities(self) -> bool:
        """Check if user is open to job opportunities."""
        return self.availability in ["active", "passive"]
    
    def matches_job_title(self, job_title: str) -> bool:
        """Check if job title matches user's desired titles."""
        if not self.desired_job_titles:
            return True  # No specific preference
        
        job_title_lower = job_title.lower()
        return any(title.lower() in job_title_lower for title in self.desired_job_titles)
    
    def matches_industry(self, industry: str) -> bool:
        """Check if industry matches user's desired industries."""
        if not self.desired_industries:
            return True  # No specific preference
        
        industry_lower = industry.lower()
        return any(ind.lower() in industry_lower for ind in self.desired_industries)
    
    def get_preference_score(self, job_title: str, industry: str, location: str) -> float:
        """Calculate preference match score for a job."""
        score = 0.0
        total_factors = 0
        
        # Job title preference
        if self.matches_job_title(job_title):
            score += 1.0
        total_factors += 1
        
        # Industry preference
        if self.matches_industry(industry):
            score += 1.0
        total_factors += 1
        
        # Location preference
        if self.location_preference and self.location_preference.matches_location(location):
            score += 1.0
        total_factors += 1
        
        return score / total_factors if total_factors > 0 else 0.0


@dataclass
class Resume:
    """
    Resume entity containing parsed resume data and metadata.
    """
    
    user_id: UUID
    original_filename: str
    id: UUID = field(default_factory=uuid4)
    file_path: Optional[str] = field(default=None)
    file_size: Optional[int] = field(default=None)
    personal_info: Dict[str, Any] = field(default_factory=dict)
    skills: Skills = field(default_factory=lambda: Skills([]))
    experience: Optional[Experience] = field(default=None)
    education: List[Dict[str, Any]] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    work_history: List[Dict[str, Any]] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)
    parsed_data: Optional[Dict[str, Any]] = field(default=None)
    parsing_status: str = field(default="pending")
    parsing_error: Optional[str] = field(default=None)
    embeddings: Optional[Dict[str, List[float]]] = field(default=None)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate resume after initialization."""
        if not self.original_filename.strip():
            raise ValueError("Original filename cannot be empty")
        
        if self.file_size and self.file_size <= 0:
            raise ValueError("File size must be positive")
    
    def mark_parsing_completed(self, parsed_data: Dict[str, Any]) -> None:
        """Mark resume parsing as completed with results."""
        self.parsing_status = "completed"
        self.parsed_data = parsed_data
        self.parsing_error = None
        self.updated_at = datetime.utcnow()
    
    def mark_parsing_failed(self, error_message: str) -> None:
        """Mark resume parsing as failed with error."""
        self.parsing_status = "failed"
        self.parsing_error = error_message
        self.updated_at = datetime.utcnow()
    
    def update_skills(self, new_skills: Skills) -> None:
        """Update resume skills."""
        self.skills = new_skills
        self.updated_at = datetime.utcnow()
    
    def update_experience(self, new_experience: Experience) -> None:
        """Update work experience."""
        self.experience = new_experience
        self.updated_at = datetime.utcnow()
    
    def add_work_entry(self, work_entry: Dict[str, Any]) -> None:
        """Add a work history entry."""
        required_fields = ["company", "position", "start_date"]
        if not all(field in work_entry for field in required_fields):
            raise ValueError("Work entry must include company, position, and start_date")
        
        self.work_history.append(work_entry)
        self.updated_at = datetime.utcnow()
    
    def add_education_entry(self, education_entry: Dict[str, Any]) -> None:
        """Add an education entry."""
        required_fields = ["institution", "degree"]
        if not all(field in education_entry for field in required_fields):
            raise ValueError("Education entry must include institution and degree")
        
        self.education.append(education_entry)
        self.updated_at = datetime.utcnow()
    
    def set_embeddings(self, embeddings: Dict[str, List[float]]) -> None:
        """Set embeddings for semantic search."""
        self.embeddings = embeddings
        self.updated_at = datetime.utcnow()
    
    def is_parsed(self) -> bool:
        """Check if resume has been successfully parsed."""
        return self.parsing_status == "completed"
    
    def is_parsing_failed(self) -> bool:
        """Check if resume parsing failed."""
        return self.parsing_status == "failed"
    
    def get_total_experience_years(self) -> float:
        """Calculate total years of experience from work history."""
        if self.experience:
            return self.experience.total_years
        
        # Calculate from work history if experience not set
        total_years = 0.0
        for work_entry in self.work_history:
            start_date = work_entry.get("start_date")
            end_date = work_entry.get("end_date", datetime.utcnow())
            
            if start_date:
                if isinstance(start_date, str):
                    try:
                        start_date = datetime.fromisoformat(start_date)
                    except ValueError:
                        continue
                
                if isinstance(end_date, str):
                    try:
                        end_date = datetime.fromisoformat(end_date)
                    except ValueError:
                        end_date = datetime.utcnow()
                
                duration = (end_date - start_date).days / 365.25
                total_years += max(0, duration)
        
        return total_years
    
    def get_latest_position(self) -> Optional[Dict[str, Any]]:
        """Get the most recent work position."""
        if not self.work_history:
            return None
        
        # Sort by start date descending
        sorted_history = sorted(
            self.work_history,
            key=lambda x: x.get("start_date", datetime.min),
            reverse=True
        )
        
        return sorted_history[0] if sorted_history else None
    
    def get_companies_worked_for(self) -> List[str]:
        """Get list of companies from work history."""
        companies = []
        for work_entry in self.work_history:
            company = work_entry.get("company")
            if company and company not in companies:
                companies.append(company)
        
        return companies
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert resume to dictionary representation."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "original_filename": self.original_filename,
            "file_size": self.file_size,
            "personal_info": self.personal_info,
            "skills": [skill.name for skill in self.skills.skills],
            "experience_years": self.get_total_experience_years(),
            "education": self.education,
            "certifications": self.certifications,
            "work_history": self.work_history,
            "achievements": self.achievements,
            "parsing_status": self.parsing_status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class User:
    """
    Core user entity representing a user account with profile and resumes.
    
    This entity encapsulates user account information, career profile,
    and associated resumes within the matching system.
    """
    
    email: str
    full_name: str
    id: UUID = field(default_factory=uuid4)
    
    # Account status
    is_active: bool = field(default=True)
    email_verified: bool = field(default=False)
    
    # Profile information
    profile: Optional[UserProfile] = field(default=None)
    
    # Associated resumes
    resumes: List[Resume] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = field(default=None)
    
    def __post_init__(self):
        """Validate user entity after initialization."""
        if not self.email or "@" not in self.email:
            raise ValueError("Valid email address is required")
        
        if not self.full_name.strip():
            raise ValueError("Full name cannot be empty")
        
        # Normalize email
        object.__setattr__(self, 'email', self.email.lower().strip())
    
    def deactivate(self) -> None:
        """Deactivate the user account."""
        self.is_active = False
        self.updated_at = datetime.utcnow()
    
    def reactivate(self) -> None:
        """Reactivate the user account."""
        self.is_active = True
        self.updated_at = datetime.utcnow()
    
    def verify_email(self) -> None:
        """Mark email as verified."""
        self.email_verified = True
        self.updated_at = datetime.utcnow()
    
    def update_profile(self, new_profile: UserProfile) -> None:
        """Update user profile."""
        self.profile = new_profile
        self.updated_at = datetime.utcnow()
    
    def record_login(self) -> None:
        """Record user login timestamp."""
        self.last_login = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def add_resume(self, resume: Resume) -> None:
        """Add a resume to the user's account."""
        if resume.user_id != self.id:
            raise ValueError("Resume user_id must match user ID")
        
        # Check for duplicate filenames
        existing_names = {r.original_filename for r in self.resumes}
        if resume.original_filename in existing_names:
            raise ValueError(f"Resume with filename '{resume.original_filename}' already exists")
        
        self.resumes.append(resume)
        self.updated_at = datetime.utcnow()
    
    def remove_resume(self, resume_id: UUID) -> bool:
        """Remove a resume from the user's account."""
        original_count = len(self.resumes)
        self.resumes = [r for r in self.resumes if r.id != resume_id]
        
        if len(self.resumes) < original_count:
            self.updated_at = datetime.utcnow()
            return True
        
        return False
    
    def get_resume_by_id(self, resume_id: UUID) -> Optional[Resume]:
        """Get a specific resume by ID."""
        return next((r for r in self.resumes if r.id == resume_id), None)
    
    def get_latest_resume(self) -> Optional[Resume]:
        """Get the most recently updated resume."""
        if not self.resumes:
            return None
        
        return max(self.resumes, key=lambda r: r.updated_at)
    
    def get_parsed_resumes(self) -> List[Resume]:
        """Get all successfully parsed resumes."""
        return [r for r in self.resumes if r.is_parsed()]
    
    def has_active_resume(self) -> bool:
        """Check if user has at least one successfully parsed resume."""
        return any(r.is_parsed() for r in self.resumes)
    
    def is_eligible_for_matching(self) -> bool:
        """Check if user is eligible for job matching."""
        return (self.is_active and 
                self.email_verified and 
                self.has_active_resume() and
                self.profile and 
                self.profile.is_open_to_opportunities())
    
    def get_combined_skills(self) -> Skills:
        """Get combined skills from all parsed resumes."""
        all_skills = []
        
        for resume in self.get_parsed_resumes():
            all_skills.extend(resume.skills.skills)
        
        return Skills(all_skills)
    
    def get_primary_experience(self) -> Optional[Experience]:
        """Get experience from the latest parsed resume."""
        latest_resume = self.get_latest_resume()
        if latest_resume and latest_resume.is_parsed():
            return latest_resume.experience
        
        return None
    
    def get_activity_status(self) -> str:
        """Get user activity status."""
        if not self.last_login:
            return "never_logged_in"
        
        days_since_login = (datetime.utcnow() - self.last_login).days
        
        if days_since_login <= 7:
            return "active"
        elif days_since_login <= 30:
            return "recent"
        elif days_since_login <= 90:
            return "inactive"
        else:
            return "dormant"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary representation."""
        return {
            "id": str(self.id),
            "email": self.email,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "email_verified": self.email_verified,
            "profile": {
                "desired_job_titles": self.profile.desired_job_titles if self.profile else [],
                "desired_industries": self.profile.desired_industries if self.profile else [],
                "availability": self.profile.availability if self.profile else "active",
                "remote_work_preference": self.profile.remote_work_preference if self.profile else "flexible"
            } if self.profile else None,
            "resume_count": len(self.resumes),
            "parsed_resume_count": len(self.get_parsed_resumes()),
            "activity_status": self.get_activity_status(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None
        }