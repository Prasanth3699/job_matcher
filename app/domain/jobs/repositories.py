"""
Jobs domain repositories providing data access interfaces and implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from .entities import Job, JobDetails, JobRequirements
from ..matching.value_objects import Skills


class JobRepository(ABC):
    """
    Abstract repository interface for job operations.
    
    This interface defines the contract for data access operations
    related to jobs, following the Repository pattern.
    """
    
    @abstractmethod
    async def save_job(self, job: Job) -> Job:
        """Save a job entity to persistent storage."""
        pass
    
    @abstractmethod
    async def get_job_by_id(self, job_id: UUID) -> Optional[Job]:
        """Retrieve a job by its ID."""
        pass
    
    @abstractmethod
    async def get_job_by_source_id(self, source_id: str) -> Optional[Job]:
        """Retrieve a job by its external source ID."""
        pass
    
    @abstractmethod
    async def get_jobs_by_ids(self, job_ids: List[UUID]) -> List[Job]:
        """Retrieve multiple jobs by their IDs."""
        pass
    
    @abstractmethod
    async def get_active_jobs(self, limit: int = 100, offset: int = 0) -> List[Job]:
        """Get active job postings with pagination."""
        pass
    
    @abstractmethod
    async def search_jobs_by_keywords(self, keywords: List[str], limit: int = 50) -> List[Job]:
        """Search jobs by keywords."""
        pass
    
    @abstractmethod
    async def search_jobs_by_skills(self, skills: Skills, limit: int = 50) -> List[Job]:
        """Search jobs by required skills."""
        pass
    
    @abstractmethod
    async def get_jobs_by_company(self, company_name: str, limit: int = 50) -> List[Job]:
        """Get jobs from a specific company."""
        pass
    
    @abstractmethod
    async def get_jobs_by_location(self, location: str, limit: int = 50) -> List[Job]:
        """Get jobs in a specific location."""
        pass
    
    @abstractmethod
    async def get_recent_jobs(self, days: int = 7, limit: int = 100) -> List[Job]:
        """Get recently posted jobs."""
        pass
    
    @abstractmethod
    async def update_job_analytics(self, job_id: UUID, view_count: int, application_count: int, match_count: int) -> bool:
        """Update job analytics counters."""
        pass
    
    @abstractmethod
    async def deactivate_job(self, job_id: UUID) -> bool:
        """Deactivate a job posting."""
        pass
    
    @abstractmethod
    async def delete_job(self, job_id: UUID) -> bool:
        """Delete a job from storage."""
        pass
    
    @abstractmethod
    async def get_jobs_count(self) -> int:
        """Get total count of jobs in the system."""
        pass


class SQLAlchemyJobRepository(JobRepository):
    """
    SQLAlchemy implementation of the job repository.
    
    This class provides concrete implementation of data access operations
    using SQLAlchemy ORM and the existing JobPosting model.
    """
    
    def __init__(self, db_session):
        """Initialize with database session."""
        self.db_session = db_session
    
    async def save_job(self, job: Job) -> Job:
        """Save a job entity to the database."""
        from app.data.models import JobPosting
        
        # Convert domain entity to ORM model
        job_posting = JobPosting(
            id=job.id,
            source_id=job.source_id,
            job_title=job.details.job_title,
            company_name=job.details.company_name,
            job_type=job.details.job_type,
            salary_min=job.requirements.salary_min,
            salary_max=job.requirements.salary_max,
            salary_currency=job.requirements.salary_currency,
            experience_min=job.requirements.experience.total_years if job.requirements.experience else None,
            experience_max=job.requirements.experience.total_years if job.requirements.experience else None,
            location=job.details.location,
            description=job.details.description,
            apply_link=job.details.apply_link,
            posting_date=job.posting_date,
            is_active=job.is_active,
            processed_data=self._serialize_job_data(job),
            created_at=job.posting_date,
            updated_at=job.last_updated
        )
        
        self.db_session.add(job_posting)
        await self.db_session.commit()
        await self.db_session.refresh(job_posting)
        
        return self._convert_to_domain_entity(job_posting)
    
    async def get_job_by_id(self, job_id: UUID) -> Optional[Job]:
        """Retrieve a job by its ID."""
        from app.data.models import JobPosting
        
        job_posting = await self.db_session.get(JobPosting, job_id)
        if not job_posting:
            return None
        
        return self._convert_to_domain_entity(job_posting)
    
    async def get_job_by_source_id(self, source_id: str) -> Optional[Job]:
        """Retrieve a job by its external source ID."""
        from app.data.models import JobPosting
        from sqlalchemy import select
        
        stmt = select(JobPosting).where(JobPosting.source_id == source_id)
        result = await self.db_session.execute(stmt)
        job_posting = result.scalar_one_or_none()
        
        if not job_posting:
            return None
        
        return self._convert_to_domain_entity(job_posting)
    
    async def get_jobs_by_ids(self, job_ids: List[UUID]) -> List[Job]:
        """Retrieve multiple jobs by their IDs."""
        from app.data.models import JobPosting
        from sqlalchemy import select
        
        stmt = select(JobPosting).where(JobPosting.id.in_(job_ids))
        result = await self.db_session.execute(stmt)
        job_postings = result.scalars().all()
        
        return [self._convert_to_domain_entity(jp) for jp in job_postings]
    
    async def get_active_jobs(self, limit: int = 100, offset: int = 0) -> List[Job]:
        """Get active job postings with pagination."""
        from app.data.models import JobPosting
        from sqlalchemy import select
        
        stmt = (
            select(JobPosting)
            .where(JobPosting.is_active == True)
            .order_by(JobPosting.posting_date.desc())
            .limit(limit)
            .offset(offset)
        )
        
        result = await self.db_session.execute(stmt)
        job_postings = result.scalars().all()
        
        return [self._convert_to_domain_entity(jp) for jp in job_postings]
    
    async def search_jobs_by_keywords(self, keywords: List[str], limit: int = 50) -> List[Job]:
        """Search jobs by keywords."""
        from app.data.models import JobPosting
        from sqlalchemy import select, or_, and_
        
        # Create search conditions for job title and description
        search_conditions = []
        for keyword in keywords:
            keyword_pattern = f"%{keyword}%"
            search_conditions.append(
                or_(
                    JobPosting.job_title.ilike(keyword_pattern),
                    JobPosting.description.ilike(keyword_pattern)
                )
            )
        
        stmt = (
            select(JobPosting)
            .where(
                and_(
                    JobPosting.is_active == True,
                    or_(*search_conditions)
                )
            )
            .order_by(JobPosting.posting_date.desc())
            .limit(limit)
        )
        
        result = await self.db_session.execute(stmt)
        job_postings = result.scalars().all()
        
        return [self._convert_to_domain_entity(jp) for jp in job_postings]
    
    async def search_jobs_by_skills(self, skills: Skills, limit: int = 50) -> List[Job]:
        """Search jobs by required skills."""
        from app.data.models import JobPosting
        from sqlalchemy import select, or_, and_
        
        # Create search conditions for skills in description
        skill_conditions = []
        for skill in skills.skills:
            skill_pattern = f"%{skill.name}%"
            skill_conditions.append(JobPosting.description.ilike(skill_pattern))
        
        stmt = (
            select(JobPosting)
            .where(
                and_(
                    JobPosting.is_active == True,
                    or_(*skill_conditions)
                )
            )
            .order_by(JobPosting.posting_date.desc())
            .limit(limit)
        )
        
        result = await self.db_session.execute(stmt)
        job_postings = result.scalars().all()
        
        return [self._convert_to_domain_entity(jp) for jp in job_postings]
    
    async def get_jobs_by_company(self, company_name: str, limit: int = 50) -> List[Job]:
        """Get jobs from a specific company."""
        from app.data.models import JobPosting
        from sqlalchemy import select
        
        stmt = (
            select(JobPosting)
            .where(
                JobPosting.company_name.ilike(f"%{company_name}%"),
                JobPosting.is_active == True
            )
            .order_by(JobPosting.posting_date.desc())
            .limit(limit)
        )
        
        result = await self.db_session.execute(stmt)
        job_postings = result.scalars().all()
        
        return [self._convert_to_domain_entity(jp) for jp in job_postings]
    
    async def get_jobs_by_location(self, location: str, limit: int = 50) -> List[Job]:
        """Get jobs in a specific location."""
        from app.data.models import JobPosting
        from sqlalchemy import select
        
        stmt = (
            select(JobPosting)
            .where(
                JobPosting.location.ilike(f"%{location}%"),
                JobPosting.is_active == True
            )
            .order_by(JobPosting.posting_date.desc())
            .limit(limit)
        )
        
        result = await self.db_session.execute(stmt)
        job_postings = result.scalars().all()
        
        return [self._convert_to_domain_entity(jp) for jp in job_postings]
    
    async def get_recent_jobs(self, days: int = 7, limit: int = 100) -> List[Job]:
        """Get recently posted jobs."""
        from app.data.models import JobPosting
        from sqlalchemy import select
        from datetime import timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        stmt = (
            select(JobPosting)
            .where(
                JobPosting.posting_date >= cutoff_date,
                JobPosting.is_active == True
            )
            .order_by(JobPosting.posting_date.desc())
            .limit(limit)
        )
        
        result = await self.db_session.execute(stmt)
        job_postings = result.scalars().all()
        
        return [self._convert_to_domain_entity(jp) for jp in job_postings]
    
    async def update_job_analytics(self, job_id: UUID, view_count: int, application_count: int, match_count: int) -> bool:
        """Update job analytics counters."""
        from app.data.models import JobPosting
        from sqlalchemy import update
        
        stmt = (
            update(JobPosting)
            .where(JobPosting.id == job_id)
            .values(
                updated_at=datetime.utcnow()
            )
        )
        
        result = await self.db_session.execute(stmt)
        await self.db_session.commit()
        
        return result.rowcount > 0
    
    async def deactivate_job(self, job_id: UUID) -> bool:
        """Deactivate a job posting."""
        from app.data.models import JobPosting
        from sqlalchemy import update
        
        stmt = (
            update(JobPosting)
            .where(JobPosting.id == job_id)
            .values(
                is_active=False,
                updated_at=datetime.utcnow()
            )
        )
        
        result = await self.db_session.execute(stmt)
        await self.db_session.commit()
        
        return result.rowcount > 0
    
    async def delete_job(self, job_id: UUID) -> bool:
        """Delete a job from storage."""
        from app.data.models import JobPosting
        
        job_posting = await self.db_session.get(JobPosting, job_id)
        if not job_posting:
            return False
        
        await self.db_session.delete(job_posting)
        await self.db_session.commit()
        
        return True
    
    async def get_jobs_count(self) -> int:
        """Get total count of jobs in the system."""
        from app.data.models import JobPosting
        from sqlalchemy import select, func
        
        stmt = select(func.count(JobPosting.id)).where(JobPosting.is_active == True)
        result = await self.db_session.execute(stmt)
        
        return result.scalar() or 0
    
    def _convert_to_domain_entity(self, job_posting) -> Job:
        """Convert ORM model to domain entity."""
        # Deserialize processed data
        processed_data = job_posting.processed_data or {}
        
        # Create job details
        details = JobDetails(
            job_title=job_posting.job_title,
            company_name=job_posting.company_name,
            job_type=job_posting.job_type or "full-time",
            location=job_posting.location or "",
            description=job_posting.description or "",
            apply_link=job_posting.apply_link,
            remote_policy=processed_data.get("remote_policy", "on-site"),
            responsibilities=processed_data.get("responsibilities", []),
            benefits=processed_data.get("benefits", []),
            department=processed_data.get("department"),
            contact_email=processed_data.get("contact_email"),
            company_size=processed_data.get("company_size"),
            industry=processed_data.get("industry"),
            company_description=processed_data.get("company_description")
        )
        
        # Create job requirements
        from ..matching.value_objects import Experience, ExperienceLevel, Skills, Skill, SkillLevel
        
        experience = None
        if job_posting.experience_min:
            # Determine experience level based on years
            if job_posting.experience_min <= 1:
                level = ExperienceLevel.ENTRY
            elif job_posting.experience_min <= 3:
                level = ExperienceLevel.JUNIOR
            elif job_posting.experience_min <= 5:
                level = ExperienceLevel.MID
            elif job_posting.experience_min <= 8:
                level = ExperienceLevel.SENIOR
            else:
                level = ExperienceLevel.LEAD
            
            experience = Experience(
                total_years=job_posting.experience_min,
                level=level
            )
        
        # Extract skills from processed data
        required_skills_data = processed_data.get("required_skills", [])
        preferred_skills_data = processed_data.get("preferred_skills", [])
        
        required_skills = Skills([
            Skill(name=skill_name, level=SkillLevel.INTERMEDIATE)
            for skill_name in required_skills_data
        ])
        
        preferred_skills = Skills([
            Skill(name=skill_name, level=SkillLevel.INTERMEDIATE)
            for skill_name in preferred_skills_data
        ])
        
        requirements = JobRequirements(
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            experience=experience,
            salary_min=job_posting.salary_min,
            salary_max=job_posting.salary_max,
            salary_currency=job_posting.salary_currency or "USD",
            education_level=processed_data.get("education_level"),
            certifications=processed_data.get("certifications", []),
            remote_allowed=processed_data.get("remote_allowed", False),
            relocation_assistance=processed_data.get("relocation_assistance", False),
            travel_required=processed_data.get("travel_required", False),
            security_clearance=processed_data.get("security_clearance")
        )
        
        return Job(
            id=job_posting.id,
            source_id=job_posting.source_id,
            details=details,
            requirements=requirements,
            posting_date=job_posting.posting_date or job_posting.created_at,
            last_updated=job_posting.updated_at,
            is_active=job_posting.is_active,
            processed_data=processed_data,
            embeddings=processed_data.get("embeddings"),
            search_keywords=processed_data.get("search_keywords", []),
            view_count=processed_data.get("view_count", 0),
            application_count=processed_data.get("application_count", 0),
            match_count=processed_data.get("match_count", 0)
        )
    
    def _serialize_job_data(self, job: Job) -> Dict[str, Any]:
        """Serialize job data for database storage."""
        return {
            "remote_policy": job.details.remote_policy,
            "responsibilities": job.details.responsibilities,
            "benefits": job.details.benefits,
            "department": job.details.department,
            "contact_email": job.details.contact_email,
            "company_size": job.details.company_size,
            "industry": job.details.industry,
            "company_description": job.details.company_description,
            "required_skills": [skill.name for skill in job.requirements.required_skills.skills],
            "preferred_skills": [skill.name for skill in job.requirements.preferred_skills.skills],
            "education_level": job.requirements.education_level,
            "certifications": job.requirements.certifications,
            "remote_allowed": job.requirements.remote_allowed,
            "relocation_assistance": job.requirements.relocation_assistance,
            "travel_required": job.requirements.travel_required,
            "security_clearance": job.requirements.security_clearance,
            "embeddings": job.embeddings,
            "search_keywords": job.search_keywords,
            "view_count": job.view_count,
            "application_count": job.application_count,
            "match_count": job.match_count
        }


class InMemoryJobRepository(JobRepository):
    """
    In-memory implementation of the job repository for testing.
    """
    
    def __init__(self):
        """Initialize with empty storage."""
        self._jobs: Dict[UUID, Job] = {}
        self._source_id_map: Dict[str, UUID] = {}
    
    async def save_job(self, job: Job) -> Job:
        """Save a job to memory storage."""
        self._jobs[job.id] = job
        if job.source_id:
            self._source_id_map[job.source_id] = job.id
        return job
    
    async def get_job_by_id(self, job_id: UUID) -> Optional[Job]:
        """Retrieve a job by its ID."""
        return self._jobs.get(job_id)
    
    async def get_job_by_source_id(self, source_id: str) -> Optional[Job]:
        """Retrieve a job by its external source ID."""
        job_id = self._source_id_map.get(source_id)
        if job_id:
            return self._jobs.get(job_id)
        return None
    
    async def get_jobs_by_ids(self, job_ids: List[UUID]) -> List[Job]:
        """Retrieve multiple jobs by their IDs."""
        return [self._jobs[job_id] for job_id in job_ids if job_id in self._jobs]
    
    async def get_active_jobs(self, limit: int = 100, offset: int = 0) -> List[Job]:
        """Get active job postings with pagination."""
        active_jobs = [job for job in self._jobs.values() if job.is_active]
        active_jobs.sort(key=lambda j: j.posting_date, reverse=True)
        return active_jobs[offset:offset + limit]
    
    async def search_jobs_by_keywords(self, keywords: List[str], limit: int = 50) -> List[Job]:
        """Search jobs by keywords."""
        matching_jobs = []
        for job in self._jobs.values():
            if not job.is_active:
                continue
            
            job_text = f"{job.details.job_title} {job.details.description}".lower()
            if any(keyword.lower() in job_text for keyword in keywords):
                matching_jobs.append(job)
        
        matching_jobs.sort(key=lambda j: j.posting_date, reverse=True)
        return matching_jobs[:limit]
    
    async def search_jobs_by_skills(self, skills: Skills, limit: int = 50) -> List[Job]:
        """Search jobs by required skills."""
        matching_jobs = []
        skill_names = {skill.name.lower() for skill in skills.skills}
        
        for job in self._jobs.values():
            if not job.is_active:
                continue
            
            job_skills = job.requirements.get_all_skills()
            job_skill_names = {skill.name.lower() for skill in job_skills.skills}
            
            if skill_names.intersection(job_skill_names):
                matching_jobs.append(job)
        
        matching_jobs.sort(key=lambda j: j.posting_date, reverse=True)
        return matching_jobs[:limit]
    
    async def get_jobs_by_company(self, company_name: str, limit: int = 50) -> List[Job]:
        """Get jobs from a specific company."""
        company_jobs = [
            job for job in self._jobs.values()
            if job.is_active and company_name.lower() in job.details.company_name.lower()
        ]
        company_jobs.sort(key=lambda j: j.posting_date, reverse=True)
        return company_jobs[:limit]
    
    async def get_jobs_by_location(self, location: str, limit: int = 50) -> List[Job]:
        """Get jobs in a specific location."""
        location_jobs = [
            job for job in self._jobs.values()
            if job.is_active and location.lower() in job.details.location.lower()
        ]
        location_jobs.sort(key=lambda j: j.posting_date, reverse=True)
        return location_jobs[:limit]
    
    async def get_recent_jobs(self, days: int = 7, limit: int = 100) -> List[Job]:
        """Get recently posted jobs."""
        from datetime import timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_jobs = [
            job for job in self._jobs.values()
            if job.is_active and job.posting_date >= cutoff_date
        ]
        recent_jobs.sort(key=lambda j: j.posting_date, reverse=True)
        return recent_jobs[:limit]
    
    async def update_job_analytics(self, job_id: UUID, view_count: int, application_count: int, match_count: int) -> bool:
        """Update job analytics counters."""
        if job_id not in self._jobs:
            return False
        
        job = self._jobs[job_id]
        job.view_count = view_count
        job.application_count = application_count
        job.match_count = match_count
        job.last_updated = datetime.utcnow()
        
        return True
    
    async def deactivate_job(self, job_id: UUID) -> bool:
        """Deactivate a job posting."""
        if job_id not in self._jobs:
            return False
        
        job = self._jobs[job_id]
        job.deactivate()
        
        return True
    
    async def delete_job(self, job_id: UUID) -> bool:
        """Delete a job from storage."""
        if job_id in self._jobs:
            job = self._jobs[job_id]
            if job.source_id and job.source_id in self._source_id_map:
                del self._source_id_map[job.source_id]
            del self._jobs[job_id]
            return True
        return False
    
    async def get_jobs_count(self) -> int:
        """Get total count of jobs in the system."""
        return sum(1 for job in self._jobs.values() if job.is_active)