"""
Users domain service containing core business logic for user management.
"""

from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from datetime import datetime, timedelta
import logging

from .entities import User, Resume, UserProfile
from .repositories import UserRepository
from ..matching.value_objects import Skills, Experience, LocationPreference, SalaryExpectation

logger = logging.getLogger(__name__)


class UserDomainService:
    """
    Core domain service for user-related business logic.
    
    This service encapsulates the business rules and operations for
    managing users and their profiles within the matching system.
    """
    
    def __init__(self, repository: UserRepository):
        """Initialize with repository dependency."""
        self.repository = repository
        self._max_resumes_per_user = 5
        self._max_file_size_mb = 10
        self._supported_file_types = [".pdf", ".doc", ".docx", ".txt"]
    
    async def create_user(
        self,
        email: str,
        full_name: str,
        profile: Optional[UserProfile] = None
    ) -> User:
        """
        Create a new user account.
        
        Args:
            email: User's email address
            full_name: User's full name
            profile: Optional user profile
            
        Returns:
            Created user entity
            
        Raises:
            ValueError: If validation fails
        """
        # Validate business rules
        await self._validate_user_creation(email, full_name)
        
        user = User(
            email=email,
            full_name=full_name,
            profile=profile,
            is_active=True,
            email_verified=False
        )
        
        return await self.repository.save_user(user)
    
    async def update_user_profile(self, user_id: UUID, profile: UserProfile) -> bool:
        """
        Update user profile.
        
        Args:
            user_id: ID of the user to update
            profile: New user profile
            
        Returns:
            True if successfully updated, False otherwise
        """
        user = await self.repository.get_user_by_id(user_id)
        if not user:
            logger.warning(f"User {user_id} not found")
            return False
        
        try:
            user.update_profile(profile)
            await self.repository.save_user(user)
            return True
        except Exception as e:
            logger.error(f"Failed to update user profile {user_id}: {e}")
            return False
    
    async def verify_user_email(self, user_id: UUID) -> bool:
        """
        Mark user email as verified.
        
        Args:
            user_id: ID of the user
            
        Returns:
            True if successfully verified, False otherwise
        """
        user = await self.repository.get_user_by_id(user_id)
        if not user:
            logger.warning(f"User {user_id} not found")
            return False
        
        try:
            user.verify_email()
            await self.repository.save_user(user)
            return True
        except Exception as e:
            logger.error(f"Failed to verify email for user {user_id}: {e}")
            return False
    
    async def record_user_login(self, user_id: UUID) -> bool:
        """
        Record user login timestamp.
        
        Args:
            user_id: ID of the user
            
        Returns:
            True if successfully recorded, False otherwise
        """
        user = await self.repository.get_user_by_id(user_id)
        if not user:
            logger.warning(f"User {user_id} not found")
            return False
        
        try:
            user.record_login()
            await self.repository.save_user(user)
            return True
        except Exception as e:
            logger.error(f"Failed to record login for user {user_id}: {e}")
            return False
    
    async def deactivate_user(self, user_id: UUID) -> bool:
        """
        Deactivate a user account.
        
        Args:
            user_id: ID of the user to deactivate
            
        Returns:
            True if successfully deactivated, False otherwise
        """
        return await self.repository.deactivate_user(user_id)
    
    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """
        Get a specific user by ID.
        
        Args:
            user_id: ID of the user
            
        Returns:
            User entity if found, None otherwise
        """
        return await self.repository.get_user_by_id(user_id)
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get a user by email address.
        
        Args:
            email: Email address to search for
            
        Returns:
            User entity if found, None otherwise
        """
        return await self.repository.get_user_by_email(email)
    
    async def add_resume_to_user(
        self,
        user_id: UUID,
        filename: str,
        file_path: str,
        file_size: int
    ) -> Optional[Resume]:
        """
        Add a new resume to a user's account.
        
        Args:
            user_id: ID of the user
            filename: Original filename of the resume
            file_path: Path where the file is stored
            file_size: Size of the file in bytes
            
        Returns:
            Created resume entity if successful, None otherwise
        """
        user = await self.repository.get_user_by_id(user_id)
        if not user:
            logger.warning(f"User {user_id} not found")
            return None
        
        try:
            # Validate resume creation
            self._validate_resume_creation(filename, file_size, len(user.resumes))
            
            resume = Resume(
                user_id=user_id,
                original_filename=filename,
                file_path=file_path,
                file_size=file_size,
                parsing_status="pending"
            )
            
            # Save resume and add to user
            saved_resume = await self.repository.save_resume(resume)
            user.add_resume(saved_resume)
            await self.repository.save_user(user)
            
            return saved_resume
        except Exception as e:
            logger.error(f"Failed to add resume to user {user_id}: {e}")
            return None
    
    async def update_resume_parsing_results(
        self,
        resume_id: UUID,
        parsed_data: Dict[str, Any],
        skills: Skills,
        experience: Optional[Experience] = None
    ) -> bool:
        """
        Update resume with parsing results.
        
        Args:
            resume_id: ID of the resume
            parsed_data: Parsed resume data
            skills: Extracted skills
            experience: Extracted experience
            
        Returns:
            True if successfully updated, False otherwise
        """
        resume = await self.repository.get_resume_by_id(resume_id)
        if not resume:
            logger.warning(f"Resume {resume_id} not found")
            return False
        
        try:
            resume.mark_parsing_completed(parsed_data)
            resume.update_skills(skills)
            
            if experience:
                resume.update_experience(experience)
            
            # Extract additional data from parsed_data
            if "personal_info" in parsed_data:
                resume.personal_info = parsed_data["personal_info"]
            
            if "education" in parsed_data:
                resume.education = parsed_data["education"]
            
            if "certifications" in parsed_data:
                resume.certifications = parsed_data["certifications"]
            
            if "work_history" in parsed_data:
                resume.work_history = parsed_data["work_history"]
            
            if "achievements" in parsed_data:
                resume.achievements = parsed_data["achievements"]
            
            await self.repository.save_resume(resume)
            return True
        except Exception as e:
            logger.error(f"Failed to update resume parsing results {resume_id}: {e}")
            return False
    
    async def mark_resume_parsing_failed(self, resume_id: UUID, error_message: str) -> bool:
        """
        Mark resume parsing as failed.
        
        Args:
            resume_id: ID of the resume
            error_message: Error description
            
        Returns:
            True if successfully marked as failed, False otherwise
        """
        resume = await self.repository.get_resume_by_id(resume_id)
        if not resume:
            logger.warning(f"Resume {resume_id} not found")
            return False
        
        try:
            resume.mark_parsing_failed(error_message)
            await self.repository.save_resume(resume)
            return True
        except Exception as e:
            logger.error(f"Failed to mark resume parsing as failed {resume_id}: {e}")
            return False
    
    async def delete_user_resume(self, user_id: UUID, resume_id: UUID) -> bool:
        """
        Delete a resume from a user's account.
        
        Args:
            user_id: ID of the user
            resume_id: ID of the resume to delete
            
        Returns:
            True if successfully deleted, False otherwise
        """
        user = await self.repository.get_user_by_id(user_id)
        if not user:
            logger.warning(f"User {user_id} not found")
            return False
        
        # Check if resume belongs to user
        user_resume = user.get_resume_by_id(resume_id)
        if not user_resume:
            logger.warning(f"Resume {resume_id} not found for user {user_id}")
            return False
        
        try:
            # Remove from user's resumes
            user.remove_resume(resume_id)
            await self.repository.save_user(user)
            
            # Delete from storage
            await self.repository.delete_resume(resume_id)
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete resume {resume_id} for user {user_id}: {e}")
            return False
    
    async def get_eligible_users_for_matching(self, limit: int = 100) -> List[User]:
        """
        Get users eligible for job matching.
        
        Args:
            limit: Maximum number of users to return
            
        Returns:
            List of eligible users
        """
        active_users = await self.repository.get_active_users(limit * 2)  # Get more to filter
        
        eligible_users = [
            user for user in active_users
            if user.is_eligible_for_matching()
        ]
        
        return eligible_users[:limit]
    
    async def get_user_analytics(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get analytics data for a specific user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Analytics data if user found, None otherwise
        """
        user = await self.repository.get_user_by_id(user_id)
        if not user:
            return None
        
        user_resumes = await self.repository.get_resumes_by_user_id(user_id)
        
        return {
            "user_id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "email_verified": user.email_verified,
            "activity_status": user.get_activity_status(),
            "total_resumes": len(user_resumes),
            "parsed_resumes": len([r for r in user_resumes if r.is_parsed()]),
            "failed_resumes": len([r for r in user_resumes if r.is_parsing_failed()]),
            "is_eligible_for_matching": user.is_eligible_for_matching(),
            "profile_completeness": self._calculate_profile_completeness(user),
            "last_resume_update": max(
                [r.updated_at for r in user_resumes],
                default=None
            ),
            "created_at": user.created_at.isoformat(),
            "updated_at": user.updated_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None
        }
    
    async def cleanup_inactive_users(self, days_threshold: int = 365) -> int:
        """
        Clean up inactive user accounts.
        
        Args:
            days_threshold: Days after which to consider users for cleanup
            
        Returns:
            Number of users cleaned up
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
        
        # Get dormant users
        dormant_users = await self.repository.get_users_by_activity_status("dormant", 1000)
        
        cleanup_count = 0
        for user in dormant_users:
            # Only cleanup users who haven't logged in within threshold and have no activity
            if (not user.last_login or 
                user.last_login < cutoff_date and
                not user.email_verified and
                not user.has_active_resume()):
                
                try:
                    await self.repository.delete_user(user.id)
                    cleanup_count += 1
                    logger.info(f"Deleted inactive user {user.id}")
                except Exception as e:
                    logger.error(f"Failed to delete user {user.id}: {e}")
        
        return cleanup_count
    
    def calculate_user_profile_score(self, user: User) -> float:
        """
        Calculate user profile completeness score.
        
        Args:
            user: User to evaluate
            
        Returns:
            Profile score between 0 and 1
        """
        return self._calculate_profile_completeness(user)
    
    def recommend_profile_improvements(self, user: User) -> List[str]:
        """
        Recommend improvements to user profile.
        
        Args:
            user: User to analyze
            
        Returns:
            List of improvement recommendations
        """
        recommendations = []
        
        if not user.email_verified:
            recommendations.append("Verify your email address")
        
        if not user.profile:
            recommendations.append("Complete your career profile")
            return recommendations
        
        if not user.profile.desired_job_titles:
            recommendations.append("Add desired job titles")
        
        if not user.profile.desired_industries:
            recommendations.append("Specify preferred industries")
        
        if not user.profile.location_preference:
            recommendations.append("Set location preferences")
        
        if not user.profile.salary_expectation:
            recommendations.append("Provide salary expectations")
        
        if not user.has_active_resume():
            recommendations.append("Upload and parse at least one resume")
        
        if len(user.get_parsed_resumes()) == 0:
            recommendations.append("Ensure your resume is successfully parsed")
        
        parsed_resumes = user.get_parsed_resumes()
        if parsed_resumes:
            latest_resume = max(parsed_resumes, key=lambda r: r.updated_at)
            if not latest_resume.skills.skills:
                recommendations.append("Add skills to your resume")
            
            if not latest_resume.experience:
                recommendations.append("Include work experience in your resume")
        
        return recommendations
    
    async def _validate_user_creation(self, email: str, full_name: str) -> None:
        """
        Validate user creation business rules.
        
        Args:
            email: Email address to validate
            full_name: Full name to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check if email already exists
        existing_user = await self.repository.get_user_by_email(email)
        if existing_user:
            raise ValueError(f"User with email {email} already exists")
        
        # Validate email format (basic check)
        if "@" not in email or "." not in email.split("@")[1]:
            raise ValueError("Invalid email format")
        
        # Validate full name
        if not full_name.strip() or len(full_name.strip()) < 2:
            raise ValueError("Full name must be at least 2 characters")
    
    def _validate_resume_creation(self, filename: str, file_size: int, current_resume_count: int) -> None:
        """
        Validate resume creation business rules.
        
        Args:
            filename: Resume filename
            file_size: File size in bytes
            current_resume_count: Current number of resumes for user
            
        Raises:
            ValueError: If validation fails
        """
        # Check resume count limit
        if current_resume_count >= self._max_resumes_per_user:
            raise ValueError(f"Maximum {self._max_resumes_per_user} resumes allowed per user")
        
        # Check file size
        max_size_bytes = self._max_file_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            raise ValueError(f"File size cannot exceed {self._max_file_size_mb}MB")
        
        # Check file type
        file_extension = "." + filename.split(".")[-1].lower() if "." in filename else ""
        if file_extension not in self._supported_file_types:
            raise ValueError(f"Unsupported file type. Supported types: {', '.join(self._supported_file_types)}")
    
    def _calculate_profile_completeness(self, user: User) -> float:
        """
        Calculate profile completeness score.
        
        Args:
            user: User to evaluate
            
        Returns:
            Completeness score between 0 and 1
        """
        score = 0.0
        total_factors = 8  # Total number of completeness factors
        
        # Basic account info (weight: 2)
        if user.email_verified:
            score += 1.0
        if user.full_name and len(user.full_name.strip()) > 0:
            score += 1.0
        
        # Profile information (weight: 4)
        if user.profile:
            if user.profile.desired_job_titles:
                score += 1.0
            if user.profile.desired_industries:
                score += 1.0
            if user.profile.location_preference:
                score += 1.0
            if user.profile.salary_expectation:
                score += 1.0
        
        # Resume information (weight: 2)
        if user.has_active_resume():
            score += 1.0
        
        parsed_resumes = user.get_parsed_resumes()
        if parsed_resumes:
            latest_resume = max(parsed_resumes, key=lambda r: r.updated_at)
            if latest_resume.skills.skills and latest_resume.experience:
                score += 1.0
        
        return score / total_factors