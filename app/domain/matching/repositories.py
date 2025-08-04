"""
Matching domain repositories providing data access interfaces and implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from .entities import Match, MatchResult, MatchStatus
from .value_objects import Skills, Experience, Score, MatchConfidence


class MatchingRepository(ABC):
    """
    Abstract repository interface for matching operations.

    This interface defines the contract for data access operations
    related to matching, following the Repository pattern.
    """

    @abstractmethod
    async def create_match(self, match: Match) -> Match:
        """Create a new match entity in persistent storage (INSERT only)."""
        pass

    @abstractmethod
    async def update_match(self, match: Match) -> Match:
        """Update an existing match entity in persistent storage (UPDATE only)."""
        pass

    @abstractmethod
    async def get_match_by_id(self, match_id: UUID) -> Optional[Match]:
        """Retrieve a match by its ID."""
        pass

    @abstractmethod
    async def get_matches_by_user_id(
        self, user_id: int, limit: int = 50
    ) -> List[Match]:
        """Get all matches for a specific user."""
        pass

    @abstractmethod
    async def get_matches_by_status(
        self, status: MatchStatus, limit: int = 100
    ) -> List[Match]:
        """Get matches filtered by status."""
        pass

    @abstractmethod
    async def update_match_status(self, match_id: UUID, status: MatchStatus) -> bool:
        """Update the status of a match."""
        pass

    @abstractmethod
    async def update_match_progress(
        self, match_id: UUID, percentage: float, step: str
    ) -> bool:
        """Update the progress of a processing match."""
        pass

    @abstractmethod
    async def delete_match(self, match_id: UUID) -> bool:
        """Delete a match from storage."""
        pass

    @abstractmethod
    async def get_processing_matches_older_than(self, minutes: int) -> List[Match]:
        """Get matches that have been processing for longer than specified minutes."""
        pass


class SQLAlchemyMatchingRepository(MatchingRepository):
    """
    SQLAlchemy implementation of the matching repository.

    This class provides concrete implementation of data access operations
    using SQLAlchemy ORM and the existing MatchJob model.
    """

    def __init__(self, db_session):
        """Initialize with database session."""
        self.db_session = db_session

    async def create_match(self, match: Match) -> Match:
        """Create a new match entity (INSERT only)."""
        from app.models.match_job import MatchJob

        match_job = MatchJob(
            id=match.id,
            user_id=match.user_id,
            status=match.status.value,
            task_id=match.task_id,
            resume_filename=match.resume_filename,
            job_ids=match.job_ids,
            preferences=match.preferences,
            match_results=self._serialize_match_results(match.match_results),
            parsed_resume_id=match.parsed_resume_id,
            error_message=match.error_message,
            progress_percentage=match.progress_percentage,
            current_step=match.current_step,
            created_at=match.created_at,
            started_at=match.started_at,
            completed_at=match.completed_at,
            updated_at=match.updated_at,
        )
        self.db_session.add(match_job)
        await self.db_session.commit()
        await self.db_session.refresh(match_job)
        return self._convert_to_domain_entity(match_job)

    async def update_match(self, match: Match) -> Match:
        """Update existing match entity (UPDATE only)."""
        from app.models.match_job import MatchJob
        from sqlalchemy import update, select

        # Prepare values to update
        values = {
            "status": match.status.value,
            "task_id": match.task_id,
            "match_results": self._serialize_match_results(match.match_results),
            "parsed_resume_id": match.parsed_resume_id,
            "error_message": match.error_message,
            "progress_percentage": match.progress_percentage,
            "current_step": match.current_step,
            "started_at": match.started_at,
            "completed_at": match.completed_at,
            "updated_at": match.updated_at or datetime.utcnow(),
        }
        stmt = update(MatchJob).where(MatchJob.id == match.id).values(**values)
        await self.db_session.execute(stmt)
        await self.db_session.commit()
        # Re-fetch updated row
        stmt_sel = select(MatchJob).where(MatchJob.id == match.id)
        result = await self.db_session.execute(stmt_sel)
        match_job = result.scalars().first()
        return self._convert_to_domain_entity(match_job)

    async def get_match_by_id(self, match_id: UUID) -> Optional[Match]:
        """Retrieve a match by its ID."""
        from app.models.match_job import MatchJob

        match_job = await self.db_session.get(MatchJob, match_id)
        if not match_job:
            return None

        return self._convert_to_domain_entity(match_job)

    async def get_matches_by_user_id(
        self, user_id: int, limit: int = 50
    ) -> List[Match]:
        """Get all matches for a specific user."""
        from app.models.match_job import MatchJob
        from sqlalchemy import select

        stmt = (
            select(MatchJob)
            .where(MatchJob.user_id == user_id)
            .order_by(MatchJob.created_at.desc())
            .limit(limit)
        )

        result = await self.db_session.execute(stmt)
        match_jobs = result.scalars().all()

        return [self._convert_to_domain_entity(mj) for mj in match_jobs]

    async def get_matches_by_status(
        self, status: MatchStatus, limit: int = 100
    ) -> List[Match]:
        """Get matches filtered by status."""
        from app.models.match_job import MatchJob
        from sqlalchemy import select

        stmt = (
            select(MatchJob)
            .where(MatchJob.status == status.value)
            .order_by(MatchJob.updated_at.desc())
            .limit(limit)
        )

        result = await self.db_session.execute(stmt)
        match_jobs = result.scalars().all()

        return [self._convert_to_domain_entity(mj) for mj in match_jobs]

    async def update_match_status(self, match_id: UUID, status: MatchStatus) -> bool:
        """Update the status of a match."""
        from app.models.match_job import MatchJob
        from sqlalchemy import update

        stmt = (
            update(MatchJob)
            .where(MatchJob.id == match_id)
            .values(status=status.value, updated_at=datetime.utcnow())
        )

        result = await self.db_session.execute(stmt)
        await self.db_session.commit()

        return result.rowcount > 0

    async def update_match_progress(
        self, match_id: UUID, percentage: float, step: str
    ) -> bool:
        """Update the progress of a processing match."""
        from app.models.match_job import MatchJob
        from sqlalchemy import update

        stmt = (
            update(MatchJob)
            .where(MatchJob.id == match_id)
            .values(
                progress_percentage=percentage,
                current_step=step,
                updated_at=datetime.utcnow(),
            )
        )

        result = await self.db_session.execute(stmt)
        await self.db_session.commit()

        return result.rowcount > 0

    async def delete_match(self, match_id: UUID) -> bool:
        """Delete a match from storage."""
        from app.models.match_job import MatchJob

        match_job = await self.db_session.get(MatchJob, match_id)
        if not match_job:
            return False

        await self.db_session.delete(match_job)
        await self.db_session.commit()

        return True

    async def get_processing_matches_older_than(self, minutes: int) -> List[Match]:
        """Get matches that have been processing for longer than specified minutes."""
        from app.models.match_job import MatchJob
        from sqlalchemy import select
        from datetime import timedelta

        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        stmt = (
            select(MatchJob)
            .where(
                MatchJob.status == MatchStatus.PROCESSING.value,
                MatchJob.started_at < cutoff_time,
            )
            .order_by(MatchJob.started_at)
        )

        result = await self.db_session.execute(stmt)
        match_jobs = result.scalars().all()

        return [self._convert_to_domain_entity(mj) for mj in match_jobs]

    def _convert_to_domain_entity(self, match_job) -> Match:
        """Convert ORM model to domain entity."""
        match_results = None
        if match_job.match_results:
            match_results = self._deserialize_match_results(match_job.match_results)

        return Match(
            id=match_job.id,
            user_id=match_job.user_id,
            status=MatchStatus(match_job.status),
            resume_filename=match_job.resume_filename,
            job_ids=match_job.job_ids or [],
            preferences=match_job.preferences or {},
            task_id=match_job.task_id,
            progress_percentage=match_job.progress_percentage,
            current_step=match_job.current_step,
            match_results=match_results,
            parsed_resume_id=match_job.parsed_resume_id,
            error_message=match_job.error_message,
            created_at=match_job.created_at,
            started_at=match_job.started_at,
            completed_at=match_job.completed_at,
            updated_at=match_job.updated_at,
        )

    def _serialize_match_results(
        self, match_results: Optional[List[MatchResult]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Serialize match results for database storage."""
        if not match_results:
            return None

        return [result.to_dict() for result in match_results]

    def _deserialize_match_results(
        self, serialized_results: List[Dict[str, Any]]
    ) -> List[MatchResult]:
        """Deserialize match results from database storage."""
        results = []

        for result_data in serialized_results:
            # Extract skills
            matching_skills = Skills(
                [
                    self._create_skill_from_string(skill)
                    for skill in result_data.get("matching_skills", [])
                ]
            )
            missing_skills = Skills(
                [
                    self._create_skill_from_string(skill)
                    for skill in result_data.get("missing_skills", [])
                ]
            )

            # Extract job details
            job_details = result_data.get("job_details", {})

            match_result = MatchResult(
                job_id=result_data["job_id"],
                original_job_id=result_data.get("original_job_id"),
                overall_score=Score(result_data["overall_score"]),
                confidence=MatchConfidence(result_data["confidence"]),
                matching_skills=matching_skills,
                missing_skills=missing_skills,
                score_breakdown=result_data.get("score_breakdown", {}),
                explanation=result_data.get("explanation", ""),
                job_title=job_details.get("job_title", ""),
                company_name=job_details.get("company_name", ""),
                location=job_details.get("location", ""),
                job_type=job_details.get("job_type", ""),
                apply_link=job_details.get("apply_link"),
                rank_position=result_data.get("rank_position"),
                diversity_score=result_data.get("diversity_score"),
            )

            results.append(match_result)

        return results

    def _create_skill_from_string(self, skill_name: str):
        """Create a Skill object from string name."""
        from .value_objects import Skill, SkillLevel

        return Skill(name=skill_name, level=SkillLevel.INTERMEDIATE)


class InMemoryMatchingRepository(MatchingRepository):
    """
    In-memory implementation of the matching repository for testing.

    This implementation stores matches in memory and is useful for
    unit tests and development scenarios.
    """

    def __init__(self):
        """Initialize with empty storage."""
        self._matches: Dict[UUID, Match] = {}

    async def create_match(self, match: Match) -> Match:
        """Create a match to memory storage (inserts new)."""
        if match.id in self._matches:
            raise ValueError(f"Match with id {match.id} already exists")
        self._matches[match.id] = match
        return match

    async def update_match(self, match: Match) -> Match:
        """Update a match in memory storage (updates existing)."""
        if match.id not in self._matches:
            raise ValueError(f"Match with id {match.id} does not exist")
        self._matches[match.id] = match
        return match

    async def get_match_by_id(self, match_id: UUID) -> Optional[Match]:
        """Retrieve a match by its ID."""
        return self._matches.get(match_id)

    async def get_matches_by_user_id(
        self, user_id: int, limit: int = 50
    ) -> List[Match]:
        """Get all matches for a specific user."""
        user_matches = [
            match for match in self._matches.values() if match.user_id == user_id
        ]

        # Sort by creation date descending
        user_matches.sort(key=lambda m: m.created_at, reverse=True)

        return user_matches[:limit]

    async def get_matches_by_status(
        self, status: MatchStatus, limit: int = 100
    ) -> List[Match]:
        """Get matches filtered by status."""
        status_matches = [
            match for match in self._matches.values() if match.status == status
        ]

        # Sort by update date descending
        status_matches.sort(key=lambda m: m.updated_at, reverse=True)

        return status_matches[:limit]

    async def update_match_status(self, match_id: UUID, status: MatchStatus) -> bool:
        """Update the status of a match."""
        if match_id not in self._matches:
            return False

        match = self._matches[match_id]
        # Create updated match (dataclass is immutable)
        updated_match = Match(
            id=match.id,
            user_id=match.user_id,
            status=status,
            resume_filename=match.resume_filename,
            job_ids=match.job_ids,
            preferences=match.preferences,
            task_id=match.task_id,
            progress_percentage=match.progress_percentage,
            current_step=match.current_step,
            match_results=match.match_results,
            parsed_resume_id=match.parsed_resume_id,
            error_message=match.error_message,
            created_at=match.created_at,
            started_at=match.started_at,
            completed_at=match.completed_at,
            updated_at=datetime.utcnow(),
        )

        self._matches[match_id] = updated_match
        return True

    async def update_match_progress(
        self, match_id: UUID, percentage: float, step: str
    ) -> bool:
        """Update the progress of a processing match."""
        if match_id not in self._matches:
            return False

        match = self._matches[match_id]
        # Create updated match
        updated_match = Match(
            id=match.id,
            user_id=match.user_id,
            status=match.status,
            resume_filename=match.resume_filename,
            job_ids=match.job_ids,
            preferences=match.preferences,
            task_id=match.task_id,
            progress_percentage=percentage,
            current_step=step,
            match_results=match.match_results,
            parsed_resume_id=match.parsed_resume_id,
            error_message=match.error_message,
            created_at=match.created_at,
            started_at=match.started_at,
            completed_at=match.completed_at,
            updated_at=datetime.utcnow(),
        )

        self._matches[match_id] = updated_match
        return True

    async def delete_match(self, match_id: UUID) -> bool:
        """Delete a match from storage."""
        if match_id in self._matches:
            del self._matches[match_id]
            return True
        return False

    async def get_processing_matches_older_than(self, minutes: int) -> List[Match]:
        """Get matches that have been processing for longer than specified minutes."""
        from datetime import timedelta

        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        old_processing_matches = [
            match
            for match in self._matches.values()
            if (
                match.status == MatchStatus.PROCESSING
                and match.started_at
                and match.started_at < cutoff_time
            )
        ]

        # Sort by start time
        old_processing_matches.sort(key=lambda m: m.started_at)

        return old_processing_matches


class SQLAlchemyMatchingRepositorySync:
    """
    Synchronous SQLAlchemy implementation used by Celery worker.
    Backed by SyncSessionLocal (psycopg2). Provides sync counterparts
    to the async repository operations.
    """

    def __init__(self, db_session):
        self.db_session = db_session

    def get_match_by_id_sync(self, match_id: UUID) -> Optional[Match]:
        from app.models.match_job import MatchJob

        obj = self.db_session.get(MatchJob, match_id)
        if not obj:
            return None
        return self._convert_to_domain_entity_sync(obj)

    def start_processing_sync(self, match_id: UUID, task_id: str) -> bool:
        from app.models.match_job import MatchJob
        from sqlalchemy import update, select

        # Load current state
        obj = self.db_session.get(MatchJob, match_id)
        if not obj:
            return False

        # Only transition from PENDING to PROCESSING; if already PROCESSING, treat as ok
        if obj.status == MatchStatus.PROCESSING.value:
            return True

        if obj.status != MatchStatus.PENDING.value:
            return False

        stmt = (
            update(MatchJob)
            .where(MatchJob.id == match_id)
            .values(
                status=MatchStatus.PROCESSING.value,
                task_id=task_id,
                started_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
        )
        self.db_session.execute(stmt)
        self.db_session.commit()
        return True

    def update_match_progress_sync(
        self, match_id: UUID, percentage: float, step: str
    ) -> bool:
        from app.models.match_job import MatchJob
        from sqlalchemy import update

        stmt = (
            update(MatchJob)
            .where(MatchJob.id == match_id)
            .values(
                progress_percentage=percentage,
                current_step=step,
                updated_at=datetime.utcnow(),
            )
        )
        result = self.db_session.execute(stmt)
        self.db_session.commit()
        return result.rowcount > 0

    def complete_match_sync(
        self, match_id: UUID, match_results: List[MatchResult], parsed_resume_id: int
    ) -> bool:
        from app.models.match_job import MatchJob
        from sqlalchemy import update, select

        # Fetch current row to compute processing duration and merge results
        obj = self.db_session.get(MatchJob, match_id)
        if not obj:
            return False

        # Serialize results
        serialized_results = self._serialize_match_results_sync(match_results)

        stmt = (
            update(MatchJob)
            .where(MatchJob.id == match_id)
            .values(
                status=MatchStatus.COMPLETED.value,
                match_results=serialized_results,
                parsed_resume_id=parsed_resume_id,
                progress_percentage=100.0,
                current_step="Completed",
                completed_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
        )
        self.db_session.execute(stmt)
        self.db_session.commit()
        return True

    def fail_match_sync(self, match_id: UUID, error_message: str) -> bool:
        from app.models.match_job import MatchJob
        from sqlalchemy import update

        stmt = (
            update(MatchJob)
            .where(MatchJob.id == match_id)
            .values(
                status=MatchStatus.FAILED.value,
                error_message=error_message,
                updated_at=datetime.utcnow(),
            )
        )
        self.db_session.execute(stmt)
        self.db_session.commit()
        return True

    # Helpers (sync variants) -----------------------------------------
    def _serialize_match_results_sync(
        self, match_results: Optional[List[MatchResult]]
    ) -> Optional[List[Dict[str, Any]]]:
        if not match_results:
            return None
        return [mr.to_dict() for mr in match_results]

    def _convert_to_domain_entity_sync(self, match_job) -> Match:
        match_results = None
        if match_job.match_results:
            match_results = self._deserialize_match_results_sync(
                match_job.match_results
            )
        return Match(
            id=match_job.id,
            user_id=match_job.user_id,
            status=MatchStatus(match_job.status),
            resume_filename=match_job.resume_filename,
            job_ids=match_job.job_ids or [],
            preferences=match_job.preferences or {},
            task_id=match_job.task_id,
            progress_percentage=match_job.progress_percentage,
            current_step=match_job.current_step,
            match_results=match_results,
            parsed_resume_id=match_job.parsed_resume_id,
            error_message=match_job.error_message,
            created_at=match_job.created_at,
            started_at=match_job.started_at,
            completed_at=match_job.completed_at,
            updated_at=match_job.updated_at,
        )

    def _deserialize_match_results_sync(
        self, serialized_results: List[Dict[str, Any]]
    ) -> List[MatchResult]:
        # Reuse the async serializer since it's pure; fallback here
        results: List[MatchResult] = []
        for result_data in serialized_results or []:
            matching_skills = Skills(
                [
                    self._create_skill_from_string_sync(skill)
                    for skill in result_data.get("matching_skills", [])
                ]
            )
            missing_skills = Skills(
                [
                    self._create_skill_from_string_sync(skill)
                    for skill in result_data.get("missing_skills", [])
                ]
            )
            job_details = result_data.get("job_details", {})
            results.append(
                MatchResult(
                    job_id=result_data.get("job_id"),
                    original_job_id=result_data.get("original_job_id"),
                    overall_score=Score(result_data.get("overall_score", 0.0)),
                    confidence=MatchConfidence(result_data.get("confidence", 0.0)),
                    matching_skills=matching_skills,
                    missing_skills=missing_skills,
                    score_breakdown=result_data.get("score_breakdown", {}),
                    explanation=result_data.get("explanation", ""),
                    job_title=job_details.get("job_title", ""),
                    company_name=job_details.get("company_name", ""),
                    location=job_details.get("location", ""),
                    job_type=job_details.get("job_type", ""),
                    apply_link=job_details.get("apply_link"),
                    rank_position=result_data.get("rank_position"),
                    diversity_score=result_data.get("diversity_score"),
                )
            )
        return results

    def _create_skill_from_string_sync(self, skill_name: str):
        from .value_objects import Skill, SkillLevel

        return Skill(name=skill_name, level=SkillLevel.INTERMEDIATE)
