# resume_matcher/data/repositories.py
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
import logging
from .models import User, Resume, JobPosting, Feedback, ModelVersion, BackupLog
from ..utils.logger import logger


class UserRepository:
    """User data access"""

    def __init__(self, db_session: Session):
        self.db = db_session

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.db.query(User).filter(User.email == email).first()

    def create_user(
        self, email: str, hashed_password: str, full_name: str = None
    ) -> User:
        """Create new user"""
        user = User(email=email, hashed_password=hashed_password, full_name=full_name)
        self.db.add(user)
        self.db.flush()
        return user


class ResumeRepository:
    """Resume data access"""

    def __init__(self, db_session: Session):
        self.db = db_session

    def create_resume(
        self,
        user_id: UUID,
        original_filename: str,
        file_path: str,
        file_size: int,
        parsed_data: Dict[str, Any] = None,
    ) -> Resume:
        """Create new resume record"""
        resume = Resume(
            user_id=user_id,
            original_filename=original_filename,
            file_path=file_path,
            file_size=file_size,
            parsed_data=parsed_data,
        )
        self.db.add(resume)
        self.db.flush()
        return resume

    def get_user_resumes(self, user_id: UUID, limit: int = 10) -> List[Resume]:
        """Get user's resumes"""
        return (
            self.db.query(Resume)
            .filter(Resume.user_id == user_id)
            .order_by(Resume.created_at.desc())
            .limit(limit)
            .all()
        )


class JobRepository:
    """Job posting data access"""

    def __init__(self, db_session: Session):
        self.db = db_session

    def create_job_posting(self, job_data: Dict[str, Any]) -> JobPosting:
        """Create new job posting"""
        job = JobPosting(
            source_id=job_data.get("source_id"),
            job_title=job_data["job_title"],
            company_name=job_data["company_name"],
            job_type=job_data.get("job_type"),
            salary_min=job_data.get("salary_min"),
            salary_max=job_data.get("salary_max"),
            salary_currency=job_data.get("salary_currency"),
            experience_min=job_data.get("experience_min"),
            experience_max=job_data.get("experience_max"),
            location=job_data.get("location"),
            description=job_data["description"],
            apply_link=job_data["apply_link"],
            posting_date=job_data.get("posting_date"),
            processed_data=job_data.get("processed_data"),
        )
        self.db.add(job)
        self.db.flush()
        return job

    def get_active_jobs(self, limit: int = 100) -> List[JobPosting]:
        """Get active job postings"""
        return (
            self.db.query(JobPosting)
            .filter(JobPosting.is_active == True)
            .order_by(JobPosting.posting_date.desc())
            .limit(limit)
            .all()
        )

    def get_jobs_for_matching(
        self, days: int = 30, limit: int = 1000
    ) -> List[JobPosting]:
        """Get recent jobs for matching"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return (
            self.db.query(JobPosting)
            .filter(
                and_(
                    JobPosting.is_active == True, JobPosting.posting_date >= cutoff_date
                )
            )
            .order_by(JobPosting.posting_date.desc())
            .limit(limit)
            .all()
        )


class FeedbackRepository:
    """Feedback data access"""

    def __init__(self, db_session: Session):
        self.db = db_session

    def create_feedback(
        self,
        user_id: UUID,
        job_id: UUID,
        resume_id: UUID,
        feedback_type: str,
        match_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Feedback:
        """Create new feedback record"""
        feedback = Feedback(
            user_id=user_id,
            job_id=job_id,
            resume_id=resume_id,
            feedback_type=feedback_type,
            match_score=match_score,
            metadata=metadata,
        )
        self.db.add(feedback)
        self.db.flush()
        return feedback

    def get_feedback_for_training(
        self, days: int = 30, sample_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get feedback data for model training"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Get positive feedback (clicks, applications, hires)
        positive = (
            self.db.query(Feedback)
            .filter(
                and_(
                    Feedback.created_at >= cutoff_date,
                    Feedback.feedback_type.in_(["click", "apply", "hired"]),
                )
            )
            .order_by(func.random())
            .limit(sample_size // 2)
            .all()
        )

        # Get negative feedback (impressions without clicks)
        negative = (
            self.db.query(Feedback)
            .filter(
                and_(
                    Feedback.created_at >= cutoff_date,
                    Feedback.feedback_type == "impression",
                    ~Feedback.job_id.in_(
                        self.db.query(Feedback.job_id)
                        .filter(Feedback.feedback_type.in_(["click", "apply", "hired"]))
                        .subquery()
                    ),
                )
            )
            .order_by(func.random())
            .limit(sample_size // 2)
            .all()
        )

        return positive + negative

    def get_conversion_rates(self, days: int = 30) -> Dict[str, float]:
        """Calculate conversion rates from feedback"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Count different feedback types
        counts = (
            self.db.query(
                Feedback.feedback_type, func.count(Feedback.id).label("count")
            )
            .filter(Feedback.created_at >= cutoff_date)
            .group_by(Feedback.feedback_type)
            .all()
        )

        counts_dict = {ftype: count for ftype, count in counts}

        # Calculate rates
        rates = {}
        if counts_dict.get("impression", 0) > 0:
            rates["impression_to_click"] = (
                counts_dict.get("click", 0) / counts_dict["impression"]
            )
        if counts_dict.get("click", 0) > 0:
            rates["click_to_apply"] = counts_dict.get("apply", 0) / counts_dict["click"]
        if counts_dict.get("apply", 0) > 0:
            rates["apply_to_hired"] = counts_dict.get("hired", 0) / counts_dict["apply"]

        return rates


class ModelVersionRepository:
    """Model version data access"""

    def __init__(self, db_session: Session):
        self.db = db_session

    def create_version(
        self,
        version_name: str,
        model_type: str,
        storage_path: str,
        metrics: Dict[str, Any],
        description: str = None,
    ) -> ModelVersion:
        """Create new model version"""
        version = ModelVersion(
            version_name=version_name,
            model_type=model_type,
            storage_path=storage_path,
            metrics=metrics,
            description=description,
        )
        self.db.add(version)
        self.db.flush()
        return version

    def activate_version(self, version_id: UUID) -> ModelVersion:
        """Activate a model version"""
        # Deactivate current active version
        self.db.query(ModelVersion).filter(ModelVersion.is_active == True).update(
            {"is_active": False}
        )

        # Activate new version
        version = (
            self.db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
        )
        if version:
            version.is_active = True
            version.activated_at = datetime.utcnow()
            self.db.flush()

        return version

    def get_active_version(self, model_type: str = "hybrid") -> Optional[ModelVersion]:
        """Get currently active model version"""
        return (
            self.db.query(ModelVersion)
            .filter(
                and_(
                    ModelVersion.is_active == True,
                    ModelVersion.model_type == model_type,
                )
            )
            .first()
        )


class BackupRepository:
    """Backup operations"""

    def __init__(self, db_session: Session):
        self.db = db_session

    def log_backup(
        self, backup_type: str, file_path: str, status: str, file_size: int = None
    ) -> BackupLog:
        """Create backup log entry"""
        log = BackupLog(
            backup_type=backup_type,
            file_path=file_path,
            file_size=file_size,
            status=status,
        )
        self.db.add(log)

        if status == "completed":
            log.completed_at = datetime.utcnow()

        self.db.flush()
        return log

    def get_recent_backups(self, limit: int = 5) -> List[BackupLog]:
        """Get recent backup logs"""
        return (
            self.db.query(BackupLog)
            .order_by(BackupLog.created_at.desc())
            .limit(limit)
            .all()
        )
