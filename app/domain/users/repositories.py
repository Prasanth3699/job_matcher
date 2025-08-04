"""
Users domain repositories providing data access interfaces and implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from .entities import User, Resume, UserProfile


class UserRepository(ABC):
    """
    Abstract repository interface for user operations.
    
    This interface defines the contract for data access operations
    related to users, following the Repository pattern.
    """
    
    @abstractmethod
    async def save_user(self, user: User) -> User:
        """Save a user entity to persistent storage."""
        pass
    
    @abstractmethod
    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Retrieve a user by their ID."""
        pass
    
    @abstractmethod
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Retrieve a user by their email address."""
        pass
    
    @abstractmethod
    async def get_users_by_activity_status(self, status: str, limit: int = 100) -> List[User]:
        """Get users filtered by activity status."""
        pass
    
    @abstractmethod
    async def get_active_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """Get active users with pagination."""
        pass
    
    @abstractmethod
    async def update_user_profile(self, user_id: UUID, profile: UserProfile) -> bool:
        """Update user profile."""
        pass
    
    @abstractmethod
    async def deactivate_user(self, user_id: UUID) -> bool:
        """Deactivate a user account."""
        pass
    
    @abstractmethod
    async def delete_user(self, user_id: UUID) -> bool:
        """Delete a user from storage."""
        pass
    
    @abstractmethod
    async def save_resume(self, resume: Resume) -> Resume:
        """Save a resume entity to persistent storage."""
        pass
    
    @abstractmethod
    async def get_resume_by_id(self, resume_id: UUID) -> Optional[Resume]:
        """Retrieve a resume by its ID."""
        pass
    
    @abstractmethod
    async def get_resumes_by_user_id(self, user_id: UUID) -> List[Resume]:
        """Get all resumes for a specific user."""
        pass
    
    @abstractmethod
    async def delete_resume(self, resume_id: UUID) -> bool:
        """Delete a resume from storage."""
        pass
    
    @abstractmethod
    async def get_users_count(self) -> int:
        """Get total count of users in the system."""
        pass


class SQLAlchemyUserRepository(UserRepository):
    """
    SQLAlchemy implementation of the user repository.
    
    This class provides concrete implementation of data access operations
    using SQLAlchemy ORM and the existing User and Resume models.
    """
    
    def __init__(self, db_session):
        """Initialize with database session."""
        self.db_session = db_session
    
    async def save_user(self, user: User) -> User:
        """Save a user entity to the database."""
        from app.data.models import User as UserModel
        
        # Convert domain entity to ORM model
        user_model = UserModel(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            hashed_password="",  # Would be set during registration
            created_at=user.created_at,
            updated_at=user.updated_at
        )
        
        self.db_session.add(user_model)
        await self.db_session.commit()
        await self.db_session.refresh(user_model)
        
        return self._convert_user_to_domain_entity(user_model)
    
    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Retrieve a user by their ID."""
        from app.data.models import User as UserModel
        
        user_model = await self.db_session.get(UserModel, user_id)
        if not user_model:
            return None
        
        return self._convert_user_to_domain_entity(user_model)
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Retrieve a user by their email address."""
        from app.data.models import User as UserModel
        from sqlalchemy import select
        
        stmt = select(UserModel).where(UserModel.email == email.lower())
        result = await self.db_session.execute(stmt)
        user_model = result.scalar_one_or_none()
        
        if not user_model:
            return None
        
        return self._convert_user_to_domain_entity(user_model)
    
    async def get_users_by_activity_status(self, status: str, limit: int = 100) -> List[User]:
        """Get users filtered by activity status."""
        from app.data.models import User as UserModel
        from sqlalchemy import select
        from datetime import timedelta
        
        # Define activity status date ranges
        now = datetime.utcnow()
        
        if status == "active":
            cutoff_date = now - timedelta(days=7)
            # This would require a last_login field in the model
            stmt = select(UserModel).limit(limit)
        elif status == "recent":
            cutoff_date = now - timedelta(days=30)
            stmt = select(UserModel).limit(limit)
        else:
            stmt = select(UserModel).limit(limit)
        
        result = await self.db_session.execute(stmt)
        user_models = result.scalars().all()
        
        return [self._convert_user_to_domain_entity(um) for um in user_models]
    
    async def get_active_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """Get active users with pagination."""
        from app.data.models import User as UserModel
        from sqlalchemy import select
        
        stmt = (
            select(UserModel)
            .order_by(UserModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        
        result = await self.db_session.execute(stmt)
        user_models = result.scalars().all()
        
        return [self._convert_user_to_domain_entity(um) for um in user_models]
    
    async def update_user_profile(self, user_id: UUID, profile: UserProfile) -> bool:
        """Update user profile."""
        from app.data.models import User as UserModel
        from sqlalchemy import update
        
        # Serialize profile data
        profile_data = self._serialize_user_profile(profile)
        
        stmt = (
            update(UserModel)
            .where(UserModel.id == user_id)
            .values(updated_at=datetime.utcnow())
        )
        
        result = await self.db_session.execute(stmt)
        await self.db_session.commit()
        
        return result.rowcount > 0
    
    async def deactivate_user(self, user_id: UUID) -> bool:
        """Deactivate a user account."""
        from app.data.models import User as UserModel
        from sqlalchemy import update
        
        stmt = (
            update(UserModel)
            .where(UserModel.id == user_id)
            .values(updated_at=datetime.utcnow())
        )
        
        result = await self.db_session.execute(stmt)
        await self.db_session.commit()
        
        return result.rowcount > 0
    
    async def delete_user(self, user_id: UUID) -> bool:
        """Delete a user from storage."""
        from app.data.models import User as UserModel
        
        user_model = await self.db_session.get(UserModel, user_id)
        if not user_model:
            return False
        
        await self.db_session.delete(user_model)
        await self.db_session.commit()
        
        return True
    
    async def save_resume(self, resume: Resume) -> Resume:
        """Save a resume entity to persistent storage."""
        from app.data.models import Resume as ResumeModel
        
        # Convert domain entity to ORM model
        resume_model = ResumeModel(
            id=resume.id,
            user_id=resume.user_id,
            original_filename=resume.original_filename,
            file_path=resume.file_path,
            file_size=resume.file_size,
            parsed_data=self._serialize_resume_data(resume),
            created_at=resume.created_at
        )
        
        self.db_session.add(resume_model)
        await self.db_session.commit()
        await self.db_session.refresh(resume_model)
        
        return self._convert_resume_to_domain_entity(resume_model)
    
    async def get_resume_by_id(self, resume_id: UUID) -> Optional[Resume]:
        """Retrieve a resume by its ID."""
        from app.data.models import Resume as ResumeModel
        
        resume_model = await self.db_session.get(ResumeModel, resume_id)
        if not resume_model:
            return None
        
        return self._convert_resume_to_domain_entity(resume_model)
    
    async def get_resumes_by_user_id(self, user_id: UUID) -> List[Resume]:
        """Get all resumes for a specific user."""
        from app.data.models import Resume as ResumeModel
        from sqlalchemy import select
        
        stmt = (
            select(ResumeModel)
            .where(ResumeModel.user_id == user_id)
            .order_by(ResumeModel.created_at.desc())
        )
        
        result = await self.db_session.execute(stmt)
        resume_models = result.scalars().all()
        
        return [self._convert_resume_to_domain_entity(rm) for rm in resume_models]
    
    async def delete_resume(self, resume_id: UUID) -> bool:
        """Delete a resume from storage."""
        from app.data.models import Resume as ResumeModel
        
        resume_model = await self.db_session.get(ResumeModel, resume_id)
        if not resume_model:
            return False
        
        await self.db_session.delete(resume_model)
        await self.db_session.commit()
        
        return True
    
    async def get_users_count(self) -> int:
        """Get total count of users in the system."""
        from app.data.models import User as UserModel
        from sqlalchemy import select, func
        
        stmt = select(func.count(UserModel.id))
        result = await self.db_session.execute(stmt)
        
        return result.scalar() or 0
    
    def _convert_user_to_domain_entity(self, user_model) -> User:
        """Convert ORM model to user domain entity."""
        # Create user profile (would typically be stored in a separate profile table)
        profile = None  # Would deserialize from database if stored
        
        user = User(
            id=user_model.id,
            email=user_model.email,
            full_name=user_model.full_name,
            is_active=True,  # Would come from model
            email_verified=True,  # Would come from model
            profile=profile,
            created_at=user_model.created_at,
            updated_at=user_model.updated_at,
            last_login=None  # Would come from model
        )
        
        return user
    
    def _convert_resume_to_domain_entity(self, resume_model) -> Resume:
        """Convert ORM model to resume domain entity."""
        from ..matching.value_objects import Skills, Skill, SkillLevel, Experience, ExperienceLevel
        
        parsed_data = resume_model.parsed_data or {}
        
        # Extract skills
        skills_data = parsed_data.get("skills", [])
        skills = Skills([
            Skill(name=skill_name, level=SkillLevel.INTERMEDIATE)
            for skill_name in skills_data
        ])
        
        # Extract experience
        experience = None
        experience_data = parsed_data.get("experience")
        if experience_data:
            experience = Experience(
                total_years=experience_data.get("total_years", 0),
                level=ExperienceLevel(experience_data.get("level", "entry"))
            )
        
        resume = Resume(
            id=resume_model.id,
            user_id=resume_model.user_id,
            original_filename=resume_model.original_filename,
            file_path=resume_model.file_path,
            file_size=resume_model.file_size,
            personal_info=parsed_data.get("personal_info", {}),
            skills=skills,
            experience=experience,
            education=parsed_data.get("education", []),
            certifications=parsed_data.get("certifications", []),
            work_history=parsed_data.get("work_history", []),
            achievements=parsed_data.get("achievements", []),
            parsed_data=parsed_data,
            parsing_status=parsed_data.get("parsing_status", "completed"),
            parsing_error=parsed_data.get("parsing_error"),
            embeddings=parsed_data.get("embeddings"),
            created_at=resume_model.created_at,
            updated_at=datetime.utcnow()
        )
        
        return resume
    
    def _serialize_user_profile(self, profile: UserProfile) -> Dict[str, Any]:
        """Serialize user profile for database storage."""
        return {
            "desired_job_titles": profile.desired_job_titles,
            "desired_industries": profile.desired_industries,
            "career_level": profile.career_level,
            "location_preference": {
                "preferred_location": profile.location_preference.preferred_location,
                "remote_acceptable": profile.location_preference.remote_acceptable,
                "max_commute_distance": profile.location_preference.max_commute_distance,
                "relocation_acceptable": profile.location_preference.relocation_acceptable
            } if profile.location_preference else None,
            "remote_work_preference": profile.remote_work_preference,
            "relocation_willingness": profile.relocation_willingness,
            "salary_expectation": {
                "min_salary": profile.salary_expectation.min_salary,
                "max_salary": profile.salary_expectation.max_salary,
                "currency": profile.salary_expectation.currency,
                "negotiable": profile.salary_expectation.negotiable
            } if profile.salary_expectation else None,
            "availability": profile.availability,
            "start_date": profile.start_date.isoformat() if profile.start_date else None,
            "company_size_preference": profile.company_size_preference,
            "work_culture_preferences": profile.work_culture_preferences,
            "benefits_priorities": profile.benefits_priorities
        }
    
    def _serialize_resume_data(self, resume: Resume) -> Dict[str, Any]:
        """Serialize resume data for database storage."""
        return {
            "personal_info": resume.personal_info,
            "skills": [skill.name for skill in resume.skills.skills],
            "experience": {
                "total_years": resume.experience.total_years,
                "level": resume.experience.level.value
            } if resume.experience else None,
            "education": resume.education,
            "certifications": resume.certifications,
            "work_history": resume.work_history,
            "achievements": resume.achievements,
            "parsing_status": resume.parsing_status,
            "parsing_error": resume.parsing_error,
            "embeddings": resume.embeddings
        }


class InMemoryUserRepository(UserRepository):
    """
    In-memory implementation of the user repository for testing.
    """
    
    def __init__(self):
        """Initialize with empty storage."""
        self._users: Dict[UUID, User] = {}
        self._email_map: Dict[str, UUID] = {}
        self._resumes: Dict[UUID, Resume] = {}
    
    async def save_user(self, user: User) -> User:
        """Save a user to memory storage."""
        self._users[user.id] = user
        self._email_map[user.email.lower()] = user.id
        return user
    
    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Retrieve a user by their ID."""
        return self._users.get(user_id)
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Retrieve a user by their email address."""
        user_id = self._email_map.get(email.lower())
        if user_id:
            return self._users.get(user_id)
        return None
    
    async def get_users_by_activity_status(self, status: str, limit: int = 100) -> List[User]:
        """Get users filtered by activity status."""
        filtered_users = []
        for user in self._users.values():
            if user.get_activity_status() == status:
                filtered_users.append(user)
        
        filtered_users.sort(key=lambda u: u.updated_at, reverse=True)
        return filtered_users[:limit]
    
    async def get_active_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """Get active users with pagination."""
        active_users = [user for user in self._users.values() if user.is_active]
        active_users.sort(key=lambda u: u.created_at, reverse=True)
        return active_users[offset:offset + limit]
    
    async def update_user_profile(self, user_id: UUID, profile: UserProfile) -> bool:
        """Update user profile."""
        user = self._users.get(user_id)
        if not user:
            return False
        
        user.update_profile(profile)
        return True
    
    async def deactivate_user(self, user_id: UUID) -> bool:
        """Deactivate a user account."""
        user = self._users.get(user_id)
        if not user:
            return False
        
        user.deactivate()
        return True
    
    async def delete_user(self, user_id: UUID) -> bool:
        """Delete a user from storage."""
        if user_id not in self._users:
            return False
        
        user = self._users[user_id]
        
        # Remove from email map
        if user.email.lower() in self._email_map:
            del self._email_map[user.email.lower()]
        
        # Remove user's resumes
        user_resume_ids = [r.id for r in user.resumes]
        for resume_id in user_resume_ids:
            if resume_id in self._resumes:
                del self._resumes[resume_id]
        
        # Remove user
        del self._users[user_id]
        
        return True
    
    async def save_resume(self, resume: Resume) -> Resume:
        """Save a resume entity to persistent storage."""
        self._resumes[resume.id] = resume
        
        # Add to user's resumes if user exists
        user = self._users.get(resume.user_id)
        if user:
            # Check if resume already in user's list
            if not any(r.id == resume.id for r in user.resumes):
                user.add_resume(resume)
        
        return resume
    
    async def get_resume_by_id(self, resume_id: UUID) -> Optional[Resume]:
        """Retrieve a resume by its ID."""
        return self._resumes.get(resume_id)
    
    async def get_resumes_by_user_id(self, user_id: UUID) -> List[Resume]:
        """Get all resumes for a specific user."""
        user_resumes = [
            resume for resume in self._resumes.values()
            if resume.user_id == user_id
        ]
        
        user_resumes.sort(key=lambda r: r.created_at, reverse=True)
        return user_resumes
    
    async def delete_resume(self, resume_id: UUID) -> bool:
        """Delete a resume from storage."""
        if resume_id not in self._resumes:
            return False
        
        resume = self._resumes[resume_id]
        
        # Remove from user's resumes
        user = self._users.get(resume.user_id)
        if user:
            user.remove_resume(resume_id)
        
        # Remove from storage
        del self._resumes[resume_id]
        
        return True
    
    async def get_users_count(self) -> int:
        """Get total count of users in the system."""
        return len([user for user in self._users.values() if user.is_active])