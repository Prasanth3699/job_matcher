"""
Jobs domain service containing core business logic for job management.
"""

from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from datetime import datetime, timedelta
import logging

from .entities import Job, JobDetails, JobRequirements
from .repositories import JobRepository
from ..matching.value_objects import Skills, LocationPreference, SalaryExpectation

logger = logging.getLogger(__name__)


class JobDomainService:
    """
    Core domain service for job-related business logic.
    
    This service encapsulates the business rules and operations for
    managing job postings within the matching system.
    """
    
    def __init__(self, repository: JobRepository):
        """Initialize with repository dependency."""
        self.repository = repository
        self._popularity_threshold = 100  # View count threshold for popular jobs
        self._recent_days_threshold = 7   # Days threshold for recent jobs
    
    async def create_job(
        self,
        details: JobDetails,
        requirements: JobRequirements,
        source_id: Optional[str] = None
    ) -> Job:
        """
        Create a new job posting.
        
        Args:
            details: Job details and metadata
            requirements: Job requirements and criteria
            source_id: External system reference ID
            
        Returns:
            Created job entity
        """
        # Validate business rules
        self._validate_job_creation(details, requirements)
        
        job = Job(
            source_id=source_id,
            details=details,
            requirements=requirements,
            posting_date=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            is_active=True
        )
        
        # Generate search keywords for better discoverability
        keywords = self._extract_search_keywords(details, requirements)
        job.add_search_keywords(keywords)
        
        return await self.repository.save_job(job)
    
    async def update_job_details(self, job_id: UUID, new_details: JobDetails) -> bool:
        """
        Update job details.
        
        Args:
            job_id: ID of the job to update
            new_details: New job details
            
        Returns:
            True if successfully updated, False otherwise
        """
        job = await self.repository.get_job_by_id(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found")
            return False
        
        try:
            job.update_details(new_details)
            
            # Update search keywords
            keywords = self._extract_search_keywords(new_details, job.requirements)
            job.add_search_keywords(keywords)
            
            await self.repository.save_job(job)
            return True
        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {e}")
            return False
    
    async def update_job_requirements(self, job_id: UUID, new_requirements: JobRequirements) -> bool:
        """
        Update job requirements.
        
        Args:
            job_id: ID of the job to update
            new_requirements: New job requirements
            
        Returns:
            True if successfully updated, False otherwise
        """
        job = await self.repository.get_job_by_id(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found")
            return False
        
        try:
            job.update_requirements(new_requirements)
            
            # Update search keywords
            keywords = self._extract_search_keywords(job.details, new_requirements)
            job.add_search_keywords(keywords)
            
            await self.repository.save_job(job)
            return True
        except Exception as e:
            logger.error(f"Failed to update job requirements {job_id}: {e}")
            return False
    
    async def deactivate_job(self, job_id: UUID) -> bool:
        """
        Deactivate a job posting.
        
        Args:
            job_id: ID of the job to deactivate
            
        Returns:
            True if successfully deactivated, False otherwise
        """
        return await self.repository.deactivate_job(job_id)
    
    async def get_job_by_id(self, job_id: UUID) -> Optional[Job]:
        """
        Get a specific job by ID.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Job entity if found, None otherwise
        """
        return await self.repository.get_job_by_id(job_id)
    
    async def get_jobs_for_matching(self, job_ids: List[str]) -> List[Job]:
        """
        Get jobs for matching process, handling both UUID and string IDs.
        
        Args:
            job_ids: List of job IDs (can be UUIDs or source IDs)
            
        Returns:
            List of job entities
        """
        jobs = []
        
        for job_id in job_ids:
            try:
                # Try as UUID first
                uuid_id = UUID(job_id)
                job = await self.repository.get_job_by_id(uuid_id)
                if job:
                    jobs.append(job)
            except ValueError:
                # Try as source ID
                job = await self.repository.get_job_by_source_id(job_id)
                if job:
                    jobs.append(job)
        
        return jobs
    
    async def search_jobs(
        self,
        keywords: Optional[List[str]] = None,
        skills: Optional[Skills] = None,
        location: Optional[str] = None,
        company: Optional[str] = None,
        limit: int = 50
    ) -> List[Job]:
        """
        Search jobs with multiple criteria.
        
        Args:
            keywords: Keywords to search for
            skills: Skills to match
            location: Location filter
            company: Company filter
            limit: Maximum number of results
            
        Returns:
            List of matching jobs
        """
        if keywords:
            return await self.repository.search_jobs_by_keywords(keywords, limit)
        elif skills:
            return await self.repository.search_jobs_by_skills(skills, limit)
        elif location:
            return await self.repository.get_jobs_by_location(location, limit)
        elif company:
            return await self.repository.get_jobs_by_company(company, limit)
        else:
            return await self.repository.get_active_jobs(limit)
    
    async def get_recent_jobs(self, days: int = None) -> List[Job]:
        """
        Get recently posted jobs.
        
        Args:
            days: Number of days to look back (default: use service threshold)
            
        Returns:
            List of recent jobs
        """
        days = days or self._recent_days_threshold
        return await self.repository.get_recent_jobs(days)
    
    async def get_popular_jobs(self, limit: int = 50) -> List[Job]:
        """
        Get popular jobs based on view count.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of popular jobs
        """
        all_jobs = await self.repository.get_active_jobs(limit * 2)  # Get more to filter
        popular_jobs = [job for job in all_jobs if job.is_popular(self._popularity_threshold)]
        popular_jobs.sort(key=lambda j: j.view_count, reverse=True)
        return popular_jobs[:limit]
    
    async def track_job_view(self, job_id: UUID) -> bool:
        """
        Track a job view for analytics.
        
        Args:
            job_id: ID of the viewed job
            
        Returns:
            True if successfully tracked, False otherwise
        """
        job = await self.repository.get_job_by_id(job_id)
        if not job:
            return False
        
        job.increment_view_count()
        await self.repository.save_job(job)
        return True
    
    async def track_job_application(self, job_id: UUID) -> bool:
        """
        Track a job application for analytics.
        
        Args:
            job_id: ID of the job applied to
            
        Returns:
            True if successfully tracked, False otherwise
        """
        job = await self.repository.get_job_by_id(job_id)
        if not job:
            return False
        
        job.increment_application_count()
        await self.repository.save_job(job)
        return True
    
    async def track_job_match(self, job_id: UUID) -> bool:
        """
        Track a job match for analytics.
        
        Args:
            job_id: ID of the matched job
            
        Returns:
            True if successfully tracked, False otherwise
        """
        job = await self.repository.get_job_by_id(job_id)
        if not job:
            return False
        
        job.increment_match_count()
        await self.repository.save_job(job)
        return True
    
    async def get_job_analytics(self, job_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get analytics data for a specific job.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Analytics data if job found, None otherwise
        """
        job = await self.repository.get_job_by_id(job_id)
        if not job:
            return None
        
        return {
            "job_id": str(job.id),
            "view_count": job.view_count,
            "application_count": job.application_count,
            "match_count": job.match_count,
            "application_rate": job.get_application_rate(),
            "match_rate": job.get_match_rate(),
            "freshness_score": job.get_freshness_score(),
            "is_popular": job.is_popular(self._popularity_threshold),
            "is_recent": job.is_recently_posted(self._recent_days_threshold),
            "posting_date": job.posting_date.isoformat(),
            "last_updated": job.last_updated.isoformat()
        }
    
    async def cleanup_inactive_jobs(self, days_threshold: int = 90) -> int:
        """
        Clean up old inactive jobs.
        
        Args:
            days_threshold: Days after which to consider jobs for cleanup
            
        Returns:
            Number of jobs cleaned up
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
        
        # Get all jobs (would need repository method to get by date range)
        all_jobs = await self.repository.get_active_jobs(limit=10000)  # Large limit
        
        cleanup_count = 0
        for job in all_jobs:
            if (not job.is_active and 
                job.last_updated < cutoff_date and
                job.view_count == 0 and 
                job.application_count == 0):
                
                try:
                    await self.repository.delete_job(job.id)
                    cleanup_count += 1
                    logger.info(f"Deleted inactive job {job.id}")
                except Exception as e:
                    logger.error(f"Failed to delete job {job.id}: {e}")
        
        return cleanup_count
    
    def calculate_job_match_score(
        self,
        job: Job,
        candidate_skills: Skills,
        candidate_location_pref: Optional[LocationPreference] = None,
        candidate_salary_exp: Optional[SalaryExpectation] = None
    ) -> float:
        """
        Calculate how well a job matches a candidate's profile.
        
        Args:
            job: Job to evaluate
            candidate_skills: Candidate's skills
            candidate_location_pref: Candidate's location preference
            candidate_salary_exp: Candidate's salary expectation
            
        Returns:
            Match score between 0 and 1
        """
        total_score = 0.0
        total_weight = 0.0
        
        # Skills matching (weight: 0.5)
        skills_score = job.get_skill_overlap_score(candidate_skills)
        total_score += skills_score * 0.5
        total_weight += 0.5
        
        # Location matching (weight: 0.2)
        if candidate_location_pref:
            location_score = 1.0 if job.matches_location_preference(candidate_location_pref) else 0.3
            total_score += location_score * 0.2
            total_weight += 0.2
        
        # Salary matching (weight: 0.2)
        if candidate_salary_exp:
            salary_score = 1.0 if job.matches_salary_expectation(candidate_salary_exp) else 0.5
            total_score += salary_score * 0.2
            total_weight += 0.2
        
        # Freshness bonus (weight: 0.1)
        freshness_score = job.get_freshness_score()
        total_score += freshness_score * 0.1
        total_weight += 0.1
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def rank_jobs_for_candidate(
        self,
        jobs: List[Job],
        candidate_skills: Skills,
        candidate_location_pref: Optional[LocationPreference] = None,
        candidate_salary_exp: Optional[SalaryExpectation] = None
    ) -> List[Tuple[Job, float]]:
        """
        Rank jobs for a candidate based on matching criteria.
        
        Args:
            jobs: List of jobs to rank
            candidate_skills: Candidate's skills
            candidate_location_pref: Candidate's location preference
            candidate_salary_exp: Candidate's salary expectation
            
        Returns:
            List of (job, score) tuples sorted by score descending
        """
        job_scores = []
        
        for job in jobs:
            if not job.is_active:
                continue
            
            score = self.calculate_job_match_score(
                job, candidate_skills, candidate_location_pref, candidate_salary_exp
            )
            job_scores.append((job, score))
        
        # Sort by score descending
        job_scores.sort(key=lambda x: x[1], reverse=True)
        
        return job_scores
    
    def _validate_job_creation(self, details: JobDetails, requirements: JobRequirements) -> None:
        """
        Validate job creation business rules.
        
        Args:
            details: Job details to validate
            requirements: Job requirements to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Basic validation
        if not details.job_title.strip():
            raise ValueError("Job title cannot be empty")
        
        if not details.company_name.strip():
            raise ValueError("Company name cannot be empty")
        
        # Business rules validation
        if (requirements.salary_min and requirements.salary_max and 
            requirements.salary_max < requirements.salary_min):
            raise ValueError("Maximum salary cannot be less than minimum salary")
        
        # Salary range validation
        if requirements.salary_min and requirements.salary_min < 0:
            raise ValueError("Salary cannot be negative")
        
        # Skills validation
        if (len(requirements.required_skills.skills) + 
            len(requirements.preferred_skills.skills)) > 50:
            raise ValueError("Total number of skills cannot exceed 50")
    
    def _extract_search_keywords(self, details: JobDetails, requirements: JobRequirements) -> List[str]:
        """
        Extract search keywords from job details and requirements.
        
        Args:
            details: Job details
            requirements: Job requirements
            
        Returns:
            List of search keywords
        """
        keywords = []
        
        # Add job title words
        keywords.extend(details.job_title.lower().split())
        
        # Add company name
        keywords.append(details.company_name.lower())
        
        # Add location
        if details.location:
            keywords.extend(details.location.lower().split())
        
        # Add skills
        for skill in requirements.required_skills.skills:
            keywords.append(skill.name.lower())
        
        for skill in requirements.preferred_skills.skills:
            keywords.append(skill.name.lower())
        
        # Add job type
        keywords.append(details.job_type.lower())
        
        # Add industry if available
        if details.industry:
            keywords.append(details.industry.lower())
        
        # Remove duplicates and empty strings
        keywords = list(set(kw.strip() for kw in keywords if kw.strip()))
        
        return keywords