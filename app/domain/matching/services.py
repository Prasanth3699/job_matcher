"""
Matching domain service containing core business logic for resume-to-job matching.
"""

from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from datetime import datetime
import logging

from .entities import Match, MatchResult, MatchStatus
from .value_objects import (
    Skills,
    Skill,
    SkillLevel,
    Experience,
    ExperienceLevel,
    Score,
    MatchConfidence,
    LocationPreference,
    SalaryExpectation,
)
from .repositories import MatchingRepository

logger = logging.getLogger(__name__)


class MatchingDomainService:
    """
    Core domain service for resume-to-job matching business logic.

    This service encapsulates the business rules and algorithms for
    matching resumes to job postings, independent of infrastructure concerns.
    """

    def __init__(self, repository: MatchingRepository):
        """Initialize with repository dependency."""
        self.repository = repository
        self._skill_weights = {
            "technical": 0.4,
            "soft": 0.2,
            "domain": 0.3,
            "tools": 0.1,
        }
        self._score_weights = {
            "skills": 0.4,
            "experience": 0.3,
            "location": 0.1,
            "salary": 0.1,
            "education": 0.1,
        }

    async def create_match_job(
        self,
        user_id: int,
        resume_filename: str,
        job_ids: List[str],
        preferences: Dict[str, Any],
    ) -> Match:
        """
        Create a new match job for processing.

        Args:
            user_id: ID of the user requesting the match
            resume_filename: Name of the uploaded resume file
            job_ids: List of job IDs to match against
            preferences: User preferences for matching

        Returns:
            Created match entity
        """
        if not job_ids:
            raise ValueError("At least one job ID must be provided")

        if len(job_ids) > 100:  # Business rule: limit job count
            raise ValueError("Cannot match against more than 100 jobs at once")

        match = Match(
            user_id=user_id,
            resume_filename=resume_filename,
            job_ids=job_ids,
            preferences=preferences,
            status=MatchStatus.PENDING,
        )

        return await self.repository.create_match(match)

    async def start_processing(self, match_id: UUID, task_id: str) -> bool:
        """
        Mark a match job as started with processing task ID.

        This method is idempotent:
        - If match is PENDING: transition to PROCESSING and set task_id.
        - If match is already PROCESSING: treat as started and continue (handles duplicates/retries).
          If an existing task_id differs, we proceed and log a warning to avoid false negatives.
        - If match is COMPLETED/FAILED: do not start.

        Args:
            match_id: ID of the match to start
            task_id: Celery task ID for tracking

        Returns:
            True if successfully started, False otherwise
        """
        match = await self.repository.get_match_by_id(match_id)
        if not match:
            logger.warning(f"Match {match_id} not found")
            return False

        # Idempotent handling to avoid conflicts/races
        if match.status == MatchStatus.PROCESSING:
            # If task_id is missing or different, keep existing state and proceed.
            if match.task_id and match.task_id != task_id:
                logger.warning(
                    f"Match {match_id} already PROCESSING with task_id={match.task_id}, "
                    f"new_task_id={task_id}. Proceeding idempotently."
                )
            else:
                logger.info(
                    f"Match {match_id} already in PROCESSING. Proceeding idempotently."
                )
            return True

        if match.status == MatchStatus.PENDING:
            try:
                match.start_processing(task_id)
                await self.repository.update_match(match)
                return True
            except ValueError as e:
                # In case of a race where another worker just transitioned it,
                # we treat it as idempotent success if now PROCESSING.
                logger.warning(
                    f"Race detected while starting match {match_id}: {e}. "
                    f"Rechecking status for idempotent proceed."
                )
                reloaded = await self.repository.get_match_by_id(match_id)
                if reloaded and reloaded.status == MatchStatus.PROCESSING:
                    return True
                logger.error(f"Failed to start processing match {match_id}: {e}")
                return False

        # Do not start if already terminal
        if match.status in (MatchStatus.COMPLETED, MatchStatus.FAILED):
            logger.error(
                f"Cannot start processing for match {match_id} from terminal status: {match.status}"
            )
            return False

        # Fallback safety
        logger.error(
            f"Cannot start processing for match {match_id} from status: {match.status}"
        )
        return False

    async def update_progress(
        self, match_id: UUID, percentage: float, step: str
    ) -> bool:
        """
        Update the progress of a processing match.

        Args:
            match_id: ID of the match to update
            percentage: Progress percentage (0-100)
            step: Current processing step description

        Returns:
            True if successfully updated, False otherwise
        """
        return await self.repository.update_match_progress(match_id, percentage, step)

    async def complete_match(
        self, match_id: UUID, match_results: List[MatchResult], parsed_resume_id: int
    ) -> bool:
        """
        Complete a match job with results.

        Args:
            match_id: ID of the match to complete
            match_results: List of match results
            parsed_resume_id: ID of the parsed resume

        Returns:
            True if successfully completed, False otherwise
        """
        match = await self.repository.get_match_by_id(match_id)
        if not match:
            logger.warning(f"Match {match_id} not found")
            return False

        try:
            # Validate and rank results
            validated_results = self._validate_and_rank_results(match_results)

            match.complete_successfully(validated_results, parsed_resume_id)
            await self.repository.update_match(match)
            return True
        except ValueError as e:
            logger.error(f"Failed to complete match {match_id}: {e}")
            return False

    async def fail_match(self, match_id: UUID, error_message: str) -> bool:
        """
        Mark a match job as failed with error details.

        Args:
            match_id: ID of the match to fail
            error_message: Error description

        Returns:
            True if successfully marked as failed, False otherwise
        """
        match = await self.repository.get_match_by_id(match_id)
        if not match:
            logger.warning(f"Match {match_id} not found")
            return False

        try:
            match.fail_with_error(error_message)
            await self.repository.update_match(match)
            return True
        except ValueError as e:
            logger.error(f"Failed to mark match {match_id} as failed: {e}")
            return False

    async def get_user_matches(self, user_id: int, limit: int = 50) -> List[Match]:
        """
        Get all matches for a specific user.

        Args:
            user_id: ID of the user
            limit: Maximum number of matches to return

        Returns:
            List of user's matches
        """
        return await self.repository.get_matches_by_user_id(user_id, limit)

    async def get_match_by_id(self, match_id: UUID) -> Optional[Match]:
        """
        Get a specific match by ID.

        Args:
            match_id: ID of the match

        Returns:
            Match entity if found, None otherwise
        """
        return await self.repository.get_match_by_id(match_id)

    async def cleanup_stale_matches(self, timeout_minutes: int = 30) -> int:
        """
        Clean up matches that have been processing for too long.

        Args:
            timeout_minutes: Minutes after which to consider a match stale

        Returns:
            Number of matches cleaned up
        """
        stale_matches = await self.repository.get_processing_matches_older_than(
            timeout_minutes
        )

        cleanup_count = 0
        for match in stale_matches:
            try:
                match.fail_with_error(
                    f"Match timed out after {timeout_minutes} minutes"
                )
                await self.repository.save_match(match)
                cleanup_count += 1
                logger.info(f"Marked stale match {match.id} as failed")
            except Exception as e:
                logger.error(f"Failed to cleanup stale match {match.id}: {e}")

        return cleanup_count

    def calculate_skill_match_score(
        self, resume_skills: Skills, job_requirements: Skills
    ) -> Tuple[Score, Skills, Skills]:
        """
        Calculate skill matching score between resume and job requirements.

        Args:
            resume_skills: Skills from the resume
            job_requirements: Required skills for the job

        Returns:
            Tuple of (overall_score, matching_skills, missing_skills)
        """
        # Treat empty job requirements as neutral rather than perfect to avoid inflated identical scores
        if not job_requirements or len(job_requirements) == 0:
            return Score(0.5), Skills([]), Skills([])

        matching_skills = resume_skills.intersection(job_requirements)
        missing_skills = job_requirements.difference(resume_skills)

        # Calculate weighted score based on skill categories
        total_score = 0.0
        total_weight = 0.0

        for category, weight in self._skill_weights.items():
            job_category_skills = job_requirements.get_skills_by_category(category)
            if not job_category_skills:
                continue

            matching_category_skills = [
                skill for skill in matching_skills.skills if skill.category == category
            ]

            category_score = len(matching_category_skills) / len(job_category_skills)
            total_score += category_score * weight
            total_weight += weight

        # If no categorized skills, use simple ratio
        if total_weight == 0:
            if len(job_requirements) == 0:
                final_score = 1.0
            else:
                final_score = len(matching_skills) / len(job_requirements)
        else:
            final_score = total_score / total_weight

        return Score(final_score), matching_skills, missing_skills

    def calculate_experience_score(
        self, candidate_experience: Experience, required_experience: Experience
    ) -> Score:
        """
        Calculate experience matching score.

        Args:
            candidate_experience: Candidate's experience
            required_experience: Required experience for the job

        Returns:
            Experience matching score
        """
        # Years of experience score
        years_ratio = min(
            1.0,
            candidate_experience.total_years / max(1, required_experience.total_years),
        )
        years_score = min(1.0, years_ratio)

        # Experience level score
        level_order = [
            ExperienceLevel.ENTRY,
            ExperienceLevel.JUNIOR,
            ExperienceLevel.MID,
            ExperienceLevel.SENIOR,
            ExperienceLevel.LEAD,
            ExperienceLevel.PRINCIPAL,
        ]

        candidate_level_index = level_order.index(candidate_experience.level)
        required_level_index = level_order.index(required_experience.level)

        if candidate_level_index >= required_level_index:
            level_score = 1.0
        else:
            # Partial credit for lower levels
            level_score = candidate_level_index / required_level_index

        # Industry experience bonus
        industry_bonus = 0.0
        if (
            candidate_experience.industry_years
            and required_experience.industry_years
            and candidate_experience.industry_years
            >= required_experience.industry_years
        ):
            industry_bonus = 0.1

        # Leadership experience bonus
        leadership_bonus = 0.0
        if (
            candidate_experience.leadership_years
            and required_experience.leadership_years
            and candidate_experience.leadership_years
            >= required_experience.leadership_years
        ):
            leadership_bonus = 0.1

        # Combine scores
        final_score = min(
            1.0,
            (years_score * 0.6 + level_score * 0.4) + industry_bonus + leadership_bonus,
        )

        return Score(final_score)

    def calculate_location_score(
        self, preference: LocationPreference, job_location: str
    ) -> Score:
        """
        Calculate location matching score.

        Args:
            preference: User's location preference
            job_location: Job's location

        Returns:
            Location matching score
        """
        if preference.matches_location(job_location):
            return Score(1.0)

        # Partial scores for flexible preferences
        if preference.remote_acceptable and "remote" in job_location.lower():
            return Score(1.0)

        if preference.relocation_acceptable:
            return Score(0.7)  # Good score for relocation willingness

        # Check for same city/region
        normalized_job = job_location.lower().strip()
        normalized_pref = preference.preferred_location

        # Simple location similarity (could be enhanced with geocoding)
        if any(part in normalized_job for part in normalized_pref.split()):
            return Score(0.5)

        return Score(0.1)  # Minimum score for location mismatch

    def calculate_salary_score(
        self,
        expectation: SalaryExpectation,
        job_salary_min: Optional[float],
        job_salary_max: Optional[float],
    ) -> Score:
        """
        Calculate salary matching score.

        Args:
            expectation: User's salary expectation
            job_salary_min: Job's minimum salary
            job_salary_max: Job's maximum salary

        Returns:
            Salary matching score
        """
        score = expectation.salary_score(job_salary_min, job_salary_max)
        return Score(score)

    def calculate_overall_score(
        self,
        skill_score: Score,
        experience_score: Score,
        location_score: Score,
        salary_score: Score,
        education_score: Optional[Score] = None,
    ) -> Score:
        """
        Calculate overall matching score using weighted combination.

        Args:
            skill_score: Skills matching score
            experience_score: Experience matching score
            location_score: Location matching score
            salary_score: Salary matching score
            education_score: Education matching score (optional)

        Returns:
            Overall matching score
        """
        total_score = 0.0
        total_weight = 0.0

        # Add weighted scores
        total_score += skill_score.value * self._score_weights["skills"]
        total_weight += self._score_weights["skills"]

        total_score += experience_score.value * self._score_weights["experience"]
        total_weight += self._score_weights["experience"]

        total_score += location_score.value * self._score_weights["location"]
        total_weight += self._score_weights["location"]

        total_score += salary_score.value * self._score_weights["salary"]
        total_weight += self._score_weights["salary"]

        if education_score:
            total_score += education_score.value * self._score_weights["education"]
            total_weight += self._score_weights["education"]

        final_score = total_score / total_weight if total_weight > 0 else 0.0
        return Score(final_score)

    def calculate_match_confidence(
        self,
        skill_score: Score,
        experience_score: Score,
        data_completeness: float = 1.0,
    ) -> MatchConfidence:
        """
        Calculate confidence level for the match prediction.

        Args:
            skill_score: Skills matching score
            experience_score: Experience matching score
            data_completeness: Completeness of input data (0-1)

        Returns:
            Match confidence level
        """
        # Base confidence from score consistency
        score_variance = abs(skill_score.value - experience_score.value)
        consistency_factor = 1.0 - (score_variance * 0.5)

        # Data quality factor
        data_quality_factor = min(1.0, data_completeness)

        # Minimum score threshold factor
        min_score = min(skill_score.value, experience_score.value)
        threshold_factor = min_score  # Lower scores reduce confidence

        # Combine factors
        confidence = (
            consistency_factor * 0.4
            + data_quality_factor * 0.3
            + threshold_factor * 0.3
        )

        return MatchConfidence(min(1.0, max(0.1, confidence)))

    def _validate_and_rank_results(
        self, results: List[MatchResult]
    ) -> List[MatchResult]:
        """
        Validate and rank match results by overall score.

        Args:
            results: List of match results to validate and rank

        Returns:
            Validated and ranked results
        """
        # Validate results
        for result in results:
            if not 0 <= result.overall_score.value <= 1:
                raise ValueError(f"Invalid overall score: {result.overall_score.value}")

            if not 0 <= result.confidence.value <= 1:
                raise ValueError(f"Invalid confidence: {result.confidence.value}")

        # Sort by overall score descending
        ranked_results = sorted(
            results, key=lambda r: r.overall_score.value, reverse=True
        )

        # Assign rank positions
        for i, result in enumerate(ranked_results):
            result.rank_position = i + 1

        return ranked_results
