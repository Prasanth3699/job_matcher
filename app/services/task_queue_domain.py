"""
Updated task queue using domain services for Phase 3 implementation.
"""

from datetime import datetime
from typing import Dict, Any
import os
import json
import asyncio
import tempfile
from pathlib import Path
from uuid import UUID

from celery import Celery
from kombu import Queue, Exchange

from app.utils.logger import logger
from app.utils.serialization import make_json_serializable
from app.core.constants import (
    APIConstants,
    QueueSettings,
    TimeoutSettings,
    RetrySettings,
)
from app.db.session import get_db
from app.core.config import get_settings

# Domain services
from app.domain.matching.services import MatchingDomainService
from app.domain.matching.repositories import SQLAlchemyMatchingRepository
from app.domain.matching.repositories import SQLAlchemyMatchingRepositorySync
from app.domain.matching.value_objects import (
    Skills,
    Skill,
    SkillLevel,
    Experience,
    ExperienceLevel,
)
from app.domain.matching.entities import MatchResult, MatchStatus
from app.domain.jobs.services import JobDomainService
from app.domain.jobs.repositories import SQLAlchemyJobRepository
from app.db.session import AsyncSessionLocal
from app.db.session import SyncSessionLocal  # sync session for Celery worker

# Reuse a single ensemble engine per worker to avoid cold starts
_ENSEMBLE_ENGINE_SINGLETON = None

settings = get_settings()

# Configure Celery with environment variables
app = Celery("resume_matcher")
app.conf.update(
    broker_url=settings.CELERY_BROKER_URL,
    result_backend=settings.CELERY_RESULT_BACKEND,
)

# Custom task queue configuration using centralized constants
app.conf.task_queues = (
    Queue(
        QueueSettings.DEFAULT_QUEUE,
        Exchange(QueueSettings.DEFAULT_QUEUE),
        routing_key=QueueSettings.DEFAULT_ROUTING_KEY,
    ),
    Queue(
        QueueSettings.MODEL_TRAINING_QUEUE,
        Exchange(QueueSettings.MODEL_TRAINING_QUEUE),
        routing_key=QueueSettings.MODEL_TRAINING_ROUTING_KEY,
    ),
    Queue(
        QueueSettings.ANALYSIS_QUEUE,
        Exchange(QueueSettings.ANALYSIS_QUEUE),
        routing_key=QueueSettings.ANALYSIS_ROUTING_KEY,
    ),
)

app.conf.task_routes = {
    "app.services.task_queue_domain.process_match_job_domain": {
        "queue": QueueSettings.DEFAULT_QUEUE
    },
    "app.services.task_queue_domain.process_match_job_enhanced_ml": {
        "queue": QueueSettings.DEFAULT_QUEUE
    },
    "app.services.task_queue_domain.analyze_resume": {
        "queue": QueueSettings.ANALYSIS_QUEUE
    },
    "app.services.task_queue_domain.test_simple_task": {
        "queue": QueueSettings.DEFAULT_QUEUE
    },
}

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=QueueSettings.WORKER_PREFETCH_MULTIPLIER,
    task_acks_late=QueueSettings.TASK_ACKS_LATE,
    task_reject_on_worker_lost=QueueSettings.TASK_REJECT_ON_WORKER_LOST,
    task_track_started=QueueSettings.TASK_TRACK_STARTED,
    result_expires=QueueSettings.RESULT_EXPIRES,
    task_time_limit=TimeoutSettings.DEFAULT_TASK_TIMEOUT,
    worker_max_tasks_per_child=QueueSettings.WORKER_MAX_TASKS_PER_CHILD,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": TimeoutSettings.DEFAULT_CACHE_TTL},
    worker_pool_restarts=True,
    task_default_queue=QueueSettings.DEFAULT_QUEUE,
    task_default_exchange=QueueSettings.DEFAULT_QUEUE,
    task_default_routing_key=QueueSettings.DEFAULT_ROUTING_KEY,
)


@app.task(bind=True)
def process_match_job_domain(self, match_job_id, resume_file_data, user_data):
    """
    Synchronous Celery task wrapper to avoid asyncio event loop issues on Windows threads.
    """
    return _process_match_job_domain_sync(
        self, match_job_id, resume_file_data, user_data
    )


def _process_match_job_domain_sync(
    self, match_job_id: str, resume_file_data: Dict, user_data: Dict
):
    """
    Domain-driven Celery task for processing match job using synchronous DB to avoid
    asyncio event loop conflicts on Windows thread pools.
    """
    logger.info(f"Starting domain-based process_match_job for job ID: {match_job_id}")

    try:
        # Use a synchronous DB session in Celery worker
        with SyncSessionLocal() as db:
            logger.info("Database connection established")

            # Initialize domain services with sync repositories
            # Use the synchronous repository in Celery worker
            matching_repo = SQLAlchemyMatchingRepositorySync(db)
            matching_service = MatchingDomainService(matching_repo)

            job_repo = SQLAlchemyJobRepository(db)
            job_service = JobDomainService(job_repo)

            try:
                # Convert match_job_id to UUID
                match_uuid = UUID(match_job_id)

                # Get match job using domain service (synchronous)
                match = matching_repo.get_match_by_id_sync(match_uuid)
                if not match:
                    logger.error(f"Match job {match_job_id} not found")
                    return {"status": "error", "error": "Match job not found"}

                # Start processing using domain service (update existing row only)
                started = matching_repo.start_processing_sync(
                    match_uuid, self.request.id
                )
                if not started:
                    logger.error(f"Failed to start processing for match {match_job_id}")
                    return {"status": "error", "error": "Failed to start processing"}

                # Update progress
                matching_repo.update_match_progress_sync(
                    match_uuid, 10, "Processing resume"
                )

                # Step 1: Process resume file
                logger.info(f"Processing resume for match job {match_job_id}")

                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=f"_{resume_file_data['filename']}"
                    ) as temp_file:
                        temp_file.write(resume_file_data["file_bytes"])
                        temp_path = Path(temp_file.name)

                    try:
                        # Parse resume using legacy services (until migrated to domain)
                        from app.services.dependencies import (
                            get_resume_parser,
                            get_resume_extractor,
                        )

                        resume_parser = get_resume_parser()
                        resume_extractor = get_resume_extractor()

                        raw_text, metadata = resume_parser.parse(temp_path)
                        resume_data_obj = resume_extractor.extract(raw_text, metadata)

                        # Convert to dict
                        if hasattr(resume_data_obj, "__dict__"):
                            parsed_resume_dict = vars(resume_data_obj).copy()
                        elif isinstance(resume_data_obj, dict):
                            parsed_resume_dict = resume_data_obj.copy()
                        else:
                            raise Exception(
                                f"Unexpected resume data type: {type(resume_data_obj)}"
                            )

                        # Extract skills using domain value objects
                        resume_skills_list = parsed_resume_dict.get("skills", [])
                        resume_skills = Skills(
                            [
                                Skill(name=skill_name, level=SkillLevel.INTERMEDIATE)
                                for skill_name in resume_skills_list
                                if isinstance(skill_name, str)
                            ]
                        )

                        # Extract experience using domain value objects
                        experience_years = parsed_resume_dict.get("experience_years", 0)
                        experience_level = ExperienceLevel.ENTRY
                        if experience_years >= 8:
                            experience_level = ExperienceLevel.SENIOR
                        elif experience_years >= 5:
                            experience_level = ExperienceLevel.MID
                        elif experience_years >= 2:
                            experience_level = ExperienceLevel.JUNIOR

                        resume_experience = Experience(
                            total_years=experience_years, level=experience_level
                        )

                        # Update progress
                        matching_repo.update_match_progress_sync(
                            match_uuid, 30, "Resume parsed successfully"
                        )

                    finally:
                        if temp_path.exists():
                            temp_path.unlink()

                except Exception as e:
                    logger.error(f"Resume processing failed: {e}", exc_info=True)
                    matching_repo.fail_match_sync(
                        match_uuid, f"Resume processing failed: {str(e)}"
                    )
                    return {
                        "status": "error",
                        "error": f"Resume processing failed: {str(e)}",
                    }

                # Step 2: Save to profile service (optional)
                parsed_resume_id = None
                try:
                    from app.services.profile_client import ProfileServiceClient

                    # Push parsed resume via synchronous bridge (avoid await in sync task)
                    parsed_resume_response = (
                        ProfileServiceClient.push_parsed_resume_sync(
                            token=user_data["token"],
                            filename=resume_file_data["filename"],
                            file_bytes=resume_file_data["file_bytes"],
                            raw_text=raw_text,
                            parsed_data=parsed_resume_dict,
                            metadata=metadata,
                        )
                    )

                    if parsed_resume_response and isinstance(
                        parsed_resume_response.get("id"), int
                    ):
                        parsed_resume_id = parsed_resume_response["id"]
                        logger.info(
                            f"Resume saved to profile service with ID: {parsed_resume_id}"
                        )

                except Exception as e:
                    logger.warning(f"Failed to save resume to profile service: {e}")

                matching_repo.update_match_progress_sync(
                    match_uuid, 50, "Fetching job details"
                )

                # Step 3: Fetch jobs using domain service
                logger.info(f"Fetching jobs for IDs: {match.job_ids}")
                try:
                    job_ids_for_fetching = []
                    for job_id in match.job_ids:
                        try:
                            job_ids_for_fetching.append(int(job_id))
                        except (ValueError, TypeError):
                            job_ids_for_fetching.append(str(job_id))

                    from app.services.job_service import JobService

                    # Fetch jobs via synchronous bridge in Celery worker
                    jobs = JobService.fetch_jobs_sync(
                        job_ids_for_fetching, user_data["token"]
                    )

                    if not jobs:
                        raise Exception(
                            f"No job details found for IDs: {match.job_ids}"
                        )

                except Exception as e:
                    logger.error(f"Error fetching jobs: {e}")
                    matching_repo.fail_match_sync(
                        match_uuid, f"Job fetching failed: {str(e)}"
                    )
                    return {
                        "status": "error",
                        "error": f"Job fetching failed: {str(e)}",
                    }

                matching_service.repository.update_match_progress_sync(
                    match_uuid, 70, "Running ML matching algorithms"
                )

                # Step 4: ML-first matching via Ensemble engine; domain fallback
                try:
                    # Build resume_data compatible with ensemble engine
                    resume_data_for_ml = {
                        "summary": parsed_resume_dict.get("summary"),
                        "raw_text": raw_text,  # provide full resume context to the ensemble
                        "skills": parsed_resume_dict.get("skills", []),
                        "experiences": parsed_resume_dict.get("experiences", []),
                        "education": parsed_resume_dict.get("education", []),
                        "certifications": parsed_resume_dict.get("certifications", []),
                        "normalized_features": parsed_resume_dict.get(
                            "normalized_features", {}
                        ),
                    }

                    # Convert jobs into plain dicts and ensure essential fields
                    job_dicts: list[dict] = []
                    for job in jobs:
                        jd = (
                            job.model_dump()
                            if hasattr(job, "model_dump")
                            else dict(job)
                        )
                        # Ensure description key presence for ML; fall back to existing fields
                        description = (
                            jd.get("description")
                            or jd.get("job_description")
                            or jd.get("details")
                            or ""
                        )
                        jd["description"] = description
                        job_dicts.append(jd)

                    # Invoke ensemble engine (singleton to avoid cold starts)
                    global _ENSEMBLE_ENGINE_SINGLETON
                    if _ENSEMBLE_ENGINE_SINGLETON is None:
                        from app.core.matching.ensemble.ensemble_manager import (
                            EnsembleMatchingEngine,
                        )

                        _ENSEMBLE_ENGINE_SINGLETON = EnsembleMatchingEngine()
                    engine = _ENSEMBLE_ENGINE_SINGLETON

                    # Call ensemble synchronously via a blocking wrapper to avoid mixing event loops
                    ensemble_results = engine.match_resume_to_jobs_sync(
                        resume_data=resume_data_for_ml,
                        jobs=job_dicts,
                        top_n=len(job_dicts),
                        include_explanations=True,
                    )

                    # Map ensemble results by job_id for fast lookup
                    def _job_id_of(j: dict) -> str:
                        return str(j.get("job_id", j.get("id", "")))

                    ensemble_by_id = {
                        _job_id_of(r.get("job_details", {})): r
                        for r in ensemble_results
                    }

                    match_results = []
                    for i, job in enumerate(jobs):
                        job_dict = (
                            job.model_dump()
                            if hasattr(job, "model_dump")
                            else dict(job)
                        )
                        job_id_str = str(job_dict.get("job_id", job_dict.get("id", i)))

                        # Domain skill objects (for listing matching/missing if desired)
                        job_skills_list = job_dict.get("skills", [])
                        job_skills = Skills(
                            [
                                Skill(name=skill_name, level=SkillLevel.INTERMEDIATE)
                                for skill_name in job_skills_list
                                if isinstance(skill_name, str)
                            ]
                        )

                        # Use domain overlap to populate matching/missing lists for transparency
                        skill_score_domain, matching_skills, missing_skills = (
                            matching_service.calculate_skill_match_score(
                                resume_skills, job_skills
                            )
                        )

                        # Experience: derive per-job required experience (as previously implemented)
                        req_years = None
                        try:
                            req_years = (
                                job_dict.get("required_experience_years")
                                or job_dict.get("min_experience_years")
                                or job_dict.get("experience_years_required")
                            )
                            if isinstance(req_years, str):
                                req_years = float(req_years)
                            elif isinstance(req_years, int):
                                req_years = float(req_years)
                            elif not isinstance(req_years, (int, float)):
                                req_years = None
                        except Exception:
                            req_years = None

                        level_map = {
                            "entry": ExperienceLevel.ENTRY,
                            "junior": ExperienceLevel.JUNIOR,
                            "mid": ExperienceLevel.MID,
                            "senior": ExperienceLevel.SENIOR,
                            "lead": ExperienceLevel.LEAD,
                            "principal": ExperienceLevel.PRINCIPAL,
                        }
                        req_level = None
                        try:
                            raw_level = (
                                job_dict.get("required_experience_level")
                                or job_dict.get("experience_level")
                                or job_dict.get("seniority")
                                or job_dict.get("level")
                            )
                            if isinstance(raw_level, str):
                                req_level = level_map.get(raw_level.strip().lower())
                        except Exception:
                            req_level = None

                        required_experience = Experience(
                            total_years=(
                                float(req_years) if req_years is not None else 2.0
                            ),
                            level=(
                                req_level
                                if req_level is not None
                                else ExperienceLevel.JUNIOR
                            ),
                        )
                        experience_score = matching_service.calculate_experience_score(
                            resume_experience, required_experience
                        )

                        # Prefer ensemble overall score; fallback to domain aggregation if missing
                        ensemble_result = ensemble_by_id.get(job_id_str, {})
                        ml_overall = ensemble_result.get("overall_score")
                        # Prefer ML skill component if available for score breakdown to reflect model signal
                        ml_skill_component = ensemble_result.get("skill_score", None)

                        from app.domain.matching.value_objects import Score

                        if (
                            isinstance(ml_overall, (int, float))
                            and 0.0 <= ml_overall <= 1.0
                        ):
                            overall_score = Score(float(ml_overall))
                            # If ML provides a skill component, expose it in breakdown; else use domain skill score
                            skill_score_for_breakdown = (
                                Score(float(ml_skill_component))
                                if isinstance(ml_skill_component, (int, float))
                                and 0.0 <= ml_skill_component <= 1.0
                                else skill_score_domain
                            )
                        else:
                            # ML did not return a valid overall; fallback to domain aggregation (non-invasive)
                            overall_score = matching_service.calculate_overall_score(
                                skill_score=skill_score_domain,
                                experience_score=experience_score,
                                location_score=skill_score_domain,
                                salary_score=skill_score_domain,
                            )
                            skill_score_for_breakdown = skill_score_domain

                        confidence = matching_service.calculate_match_confidence(
                            skill_score=skill_score_for_breakdown,
                            experience_score=experience_score,
                            data_completeness=0.9,
                        )

                        # Ensure string fields are safe
                        def _to_str_if_needed(value):
                            try:
                                return str(value) if value is not None else None
                            except Exception:
                                return None

                        apply_link_value = job_dict.get("apply_link")
                        if apply_link_value is not None and not isinstance(
                            apply_link_value, str
                        ):
                            apply_link_value = _to_str_if_needed(apply_link_value)

                        job_title_value = job_dict.get("job_title", "")
                        company_name_value = job_dict.get("company_name", "")
                        location_value = job_dict.get("location", "")
                        job_type_value = job_dict.get("job_type", "")

                        match_result = MatchResult(
                            job_id=job_id_str,
                            original_job_id=job_dict.get("job_id", job_dict.get("id")),
                            overall_score=overall_score,
                            confidence=confidence,
                            matching_skills=matching_skills,
                            missing_skills=missing_skills,
                            score_breakdown={
                                "skills": skill_score_for_breakdown.value,
                                "experience": experience_score.value,
                                "overall": overall_score.value,
                            },
                            explanation=f"Match score: {overall_score.percentage():.1f}% - "
                            f"{len(matching_skills)} matching skills, "
                            f"{len(missing_skills)} missing skills",
                            job_title=_to_str_if_needed(job_title_value) or "",
                            company_name=_to_str_if_needed(company_name_value) or "",
                            location=_to_str_if_needed(location_value) or "",
                            job_type=_to_str_if_needed(job_type_value) or "",
                            apply_link=apply_link_value,
                            rank_position=i + 1,
                        )
                        match_results.append(match_result)

                    matching_service.repository.update_match_progress_sync(
                        match_uuid, 90, "Finalizing results"
                    )

                    # Complete the match using domain service (synchronous)
                    completed = matching_service.repository.complete_match_sync(
                        match_uuid, match_results, parsed_resume_id or 0
                    )

                    if not completed:
                        raise Exception("Failed to complete match job")

                    logger.info(
                        f"Match job {match_job_id} completed successfully with {len(match_results)} results"
                    )

                    return {
                        "status": "success",
                        "match_job_id": match_job_id,
                        "matches_count": len(match_results),
                        "parsed_resume_id": parsed_resume_id,
                        "processing_duration": match.get_processing_duration(),
                    }

                except Exception as e:
                    logger.error(f"Matching process failed: {e}", exc_info=True)
                    matching_service.repository.fail_match_sync(
                        match_uuid, f"Matching failed: {str(e)}"
                    )
                    return {"status": "error", "error": f"Matching failed: {str(e)}"}

            except Exception as e:
                logger.error(
                    f"Unexpected error in match processing: {e}", exc_info=True
                )
                try:
                    if "match_uuid" in locals():
                        matching_service.repository.fail_match_sync(
                            match_uuid, f"Unexpected error: {str(e)}"
                        )
                except Exception:
                    pass
                return {"status": "error", "error": f"Unexpected error: {str(e)}"}

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}", exc_info=True)
        return {"status": "error", "error": f"Service initialization failed: {str(e)}"}


@app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 60},
)
def analyze_resume(self, resume_file_data: Dict):
    """
    Celery task for analyzing resume data using domain services.
    """
    logger.info("Starting resume analysis with domain services")

    try:
        # Initialize domain services
        db = next(get_db())
        matching_repo = SQLAlchemyMatchingRepository(db)
        matching_service = MatchingDomainService(matching_repo)

        # Process resume file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f"_{resume_file_data['filename']}"
        ) as temp_file:
            temp_file.write(resume_file_data["file_bytes"])
            temp_path = Path(temp_file.name)

        try:
            # Parse resume
            from app.services.dependencies import (
                get_resume_parser,
                get_resume_extractor,
            )

            resume_parser = get_resume_parser()
            resume_extractor = get_resume_extractor()

            raw_text, metadata = resume_parser.parse(temp_path)
            resume_data_obj = resume_extractor.extract(raw_text, metadata)

            if hasattr(resume_data_obj, "__dict__"):
                parsed_resume_dict = vars(resume_data_obj).copy()
            elif isinstance(resume_data_obj, dict):
                parsed_resume_dict = resume_data_obj.copy()
            else:
                raise Exception(f"Unexpected resume data type: {type(resume_data_obj)}")

            # Extract domain objects
            resume_skills_list = parsed_resume_dict.get("skills", [])
            resume_skills = Skills(
                [
                    Skill(name=skill_name, level=SkillLevel.INTERMEDIATE)
                    for skill_name in resume_skills_list
                    if isinstance(skill_name, str)
                ]
            )

            analysis_result = {
                "status": "success",
                "skills_count": len(resume_skills),
                "skills": [skill.name for skill in resume_skills.skills],
                "experience_years": parsed_resume_dict.get("experience_years", 0),
                "metadata": metadata,
                "filename": resume_file_data["filename"],
            }

            logger.info(
                f"Resume analysis completed: {analysis_result['skills_count']} skills found"
            )
            return analysis_result

        finally:
            if temp_path.exists():
                temp_path.unlink()

    except Exception as e:
        logger.error(f"Resume analysis failed: {e}", exc_info=True)
        raise

    finally:
        try:
            db.close()
        except:
            pass


@app.task(bind=True)
def process_match_job_enhanced_ml(
    self, match_job_id: str, resume_file_data: Dict, user_data: Dict, enhanced_ml_config: Dict = None
):
    """
    Enhanced ML Pipeline task using domain services for synchronous processing.
    Integrates advanced ML capabilities while maintaining domain-driven architecture.
    """
    logger.info(f"Starting Enhanced ML Pipeline process for job ID: {match_job_id}")
    
    if enhanced_ml_config is None:
        enhanced_ml_config = {}

    try:
        # Use synchronous database session for Celery worker
        with SyncSessionLocal() as db:
            logger.info("Enhanced ML Pipeline database connection established")

            # Initialize domain services with sync repositories
            matching_repo = SQLAlchemyMatchingRepositorySync(db)
            matching_service = MatchingDomainService(matching_repo)

            try:
                # Convert match_job_id to UUID
                match_uuid = UUID(match_job_id)

                # Get match job using domain service
                match = matching_repo.get_match_by_id_sync(match_uuid)
                if not match:
                    logger.error(f"Match job {match_job_id} not found")
                    return {"status": "error", "error": "Match job not found"}

                # Start processing with Enhanced ML Pipeline
                started = matching_repo.start_processing_sync(match_uuid, self.request.id)
                if not started:
                    logger.error(f"Failed to start Enhanced ML processing for match {match_job_id}")
                    return {"status": "error", "error": "Failed to start processing"}

                # Enhanced ML Pipeline specific progress tracking
                matching_repo.update_match_progress_sync(
                    match_uuid, 15, "Enhanced ML Pipeline initialization"
                )

                # Step 1: Enhanced resume processing
                logger.info(f"Enhanced ML Pipeline: Processing resume for match job {match_job_id}")

                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=f"_{resume_file_data['filename']}"
                    ) as temp_file:
                        temp_file.write(resume_file_data["file_bytes"])
                        temp_path = Path(temp_file.name)

                    try:
                        # Parse resume using legacy services (until migrated to domain)
                        from app.services.dependencies import (
                            get_resume_parser,
                            get_resume_extractor,
                        )

                        resume_parser = get_resume_parser()
                        resume_extractor = get_resume_extractor()

                        raw_text, metadata = resume_parser.parse(temp_path)
                        resume_data_obj = resume_extractor.extract(raw_text, metadata)

                        # Convert to dict
                        if hasattr(resume_data_obj, "__dict__"):
                            parsed_resume_dict = vars(resume_data_obj).copy()
                        elif isinstance(resume_data_obj, dict):
                            parsed_resume_dict = resume_data_obj.copy()
                        else:
                            raise Exception(f"Unexpected resume data type: {type(resume_data_obj)}")

                        # Extract skills using domain value objects
                        resume_skills_list = parsed_resume_dict.get("skills", [])
                        resume_skills = Skills(
                            [
                                Skill(name=skill_name, level=SkillLevel.INTERMEDIATE)
                                for skill_name in resume_skills_list
                                if isinstance(skill_name, str)
                            ]
                        )

                        # Extract experience using domain value objects
                        experience_years = parsed_resume_dict.get("experience_years", 0)
                        experience_level = ExperienceLevel.ENTRY
                        if experience_years >= 8:
                            experience_level = ExperienceLevel.SENIOR
                        elif experience_years >= 5:
                            experience_level = ExperienceLevel.MID
                        elif experience_years >= 2:
                            experience_level = ExperienceLevel.JUNIOR

                        resume_experience = Experience(
                            total_years=experience_years, level=experience_level
                        )

                        # Update progress
                        matching_repo.update_match_progress_sync(
                            match_uuid, 35, "Enhanced ML Pipeline: Resume analysis complete"
                        )

                    finally:
                        if temp_path.exists():
                            temp_path.unlink()

                except Exception as e:
                    logger.error(f"Enhanced ML Pipeline resume processing failed: {e}", exc_info=True)
                    matching_repo.fail_match_sync(
                        match_uuid, f"Enhanced ML Pipeline resume processing failed: {str(e)}"
                    )
                    return {
                        "status": "error",
                        "error": f"Enhanced ML Pipeline resume processing failed: {str(e)}",
                    }

                # Step 2: Save to profile service (optional)
                parsed_resume_id = None
                try:
                    from app.services.profile_client import ProfileServiceClient

                    parsed_resume_response = ProfileServiceClient.push_parsed_resume_sync(
                        token=user_data["token"],
                        filename=resume_file_data["filename"],
                        file_bytes=resume_file_data["file_bytes"],
                        raw_text=raw_text,
                        parsed_data=parsed_resume_dict,
                        metadata=metadata,
                    )

                    if parsed_resume_response and isinstance(parsed_resume_response.get("id"), int):
                        parsed_resume_id = parsed_resume_response["id"]
                        logger.info(f"Enhanced ML Pipeline: Resume saved to profile service with ID: {parsed_resume_id}")

                except Exception as e:
                    logger.warning(f"Enhanced ML Pipeline: Failed to save resume to profile service: {e}")

                matching_repo.update_match_progress_sync(
                    match_uuid, 55, "Enhanced ML Pipeline: Fetching jobs"
                )

                # Step 3: Fetch jobs using domain service
                logger.info(f"Enhanced ML Pipeline: Fetching jobs for IDs: {match.job_ids}")
                try:
                    job_ids_for_fetching = []
                    for job_id in match.job_ids:
                        try:
                            job_ids_for_fetching.append(int(job_id))
                        except (ValueError, TypeError):
                            job_ids_for_fetching.append(str(job_id))

                    from app.services.job_service import JobService

                    jobs = JobService.fetch_jobs_sync(job_ids_for_fetching, user_data["token"])

                    if not jobs:
                        raise Exception(f"No job details found for IDs: {match.job_ids}")

                except Exception as e:
                    logger.error(f"Enhanced ML Pipeline: Error fetching jobs: {e}")
                    matching_repo.fail_match_sync(
                        match_uuid, f"Enhanced ML Pipeline job fetching failed: {str(e)}"
                    )
                    return {
                        "status": "error",
                        "error": f"Enhanced ML Pipeline job fetching failed: {str(e)}",
                    }

                matching_service.repository.update_match_progress_sync(
                    match_uuid, 75, "Enhanced ML Pipeline: Running advanced ML matching"
                )

                # Step 4: Enhanced ML matching with ensemble engine
                try:
                    # Build resume data compatible with Enhanced ML Pipeline
                    resume_data_for_enhanced_ml = {
                        "summary": parsed_resume_dict.get("summary"),
                        "raw_text": raw_text,
                        "skills": parsed_resume_dict.get("skills", []),
                        "experiences": parsed_resume_dict.get("experiences", []),
                        "education": parsed_resume_dict.get("education", []),
                        "certifications": parsed_resume_dict.get("certifications", []),
                        "normalized_features": parsed_resume_dict.get("normalized_features", {}),
                    }

                    # Convert jobs into plain dicts
                    job_dicts: list[dict] = []
                    for job in jobs:
                        jd = job.model_dump() if hasattr(job, "model_dump") else dict(job)
                        description = (
                            jd.get("description")
                            or jd.get("job_description")
                            or jd.get("details")
                            or ""
                        )
                        jd["description"] = description
                        job_dicts.append(jd)

                    # Use Enhanced ML Pipeline ensemble engine with singleton pattern
                    global _ENSEMBLE_ENGINE_SINGLETON
                    if _ENSEMBLE_ENGINE_SINGLETON is None:
                        from app.core.matching.ensemble.ensemble_manager import EnsembleMatchingEngine
                        
                        _ENSEMBLE_ENGINE_SINGLETON = EnsembleMatchingEngine()
                    engine = _ENSEMBLE_ENGINE_SINGLETON

                    # Enhanced ML Pipeline matching
                    ensemble_results = engine.match_resume_to_jobs_sync(
                        resume_data=resume_data_for_enhanced_ml,
                        jobs=job_dicts,
                        top_n=len(job_dicts),
                        include_explanations=True,
                    )

                    logger.info(f"Enhanced ML Pipeline: Generated {len(ensemble_results)} enhanced matches")

                    # Map ensemble results for enhanced processing
                    def _job_id_of(j: dict) -> str:
                        return str(j.get("job_id", j.get("id", "")))

                    ensemble_by_id = {
                        _job_id_of(r.get("job_details", {})): r for r in ensemble_results
                    }

                    match_results = []
                    for i, job in enumerate(jobs):
                        job_dict = job.model_dump() if hasattr(job, "model_dump") else dict(job)
                        job_id_str = str(job_dict.get("job_id", job_dict.get("id", i)))

                        # Domain skill objects for transparency
                        job_skills_list = job_dict.get("skills", [])
                        job_skills = Skills(
                            [
                                Skill(name=skill_name, level=SkillLevel.INTERMEDIATE)
                                for skill_name in job_skills_list
                                if isinstance(skill_name, str)
                            ]
                        )

                        # Use domain service for skill matching analysis
                        skill_score_domain, matching_skills, missing_skills = (
                            matching_service.calculate_skill_match_score(resume_skills, job_skills)
                        )

                        # Experience scoring
                        req_years = None
                        try:
                            req_years = (
                                job_dict.get("required_experience_years")
                                or job_dict.get("min_experience_years")
                                or job_dict.get("experience_years_required")
                            )
                            if isinstance(req_years, str):
                                req_years = float(req_years)
                            elif isinstance(req_years, int):
                                req_years = float(req_years)
                            elif not isinstance(req_years, (int, float)):
                                req_years = None
                        except Exception:
                            req_years = None

                        level_map = {
                            "entry": ExperienceLevel.ENTRY,
                            "junior": ExperienceLevel.JUNIOR,
                            "mid": ExperienceLevel.MID,
                            "senior": ExperienceLevel.SENIOR,
                            "lead": ExperienceLevel.LEAD,
                            "principal": ExperienceLevel.PRINCIPAL,
                        }
                        req_level = None
                        try:
                            raw_level = (
                                job_dict.get("required_experience_level")
                                or job_dict.get("experience_level")
                                or job_dict.get("seniority")
                                or job_dict.get("level")
                            )
                            if isinstance(raw_level, str):
                                req_level = level_map.get(raw_level.strip().lower())
                        except Exception:
                            req_level = None

                        required_experience = Experience(
                            total_years=float(req_years) if req_years is not None else 2.0,
                            level=req_level if req_level is not None else ExperienceLevel.JUNIOR,
                        )
                        experience_score = matching_service.calculate_experience_score(
                            resume_experience, required_experience
                        )

                        # Enhanced ML Pipeline scoring priority
                        ensemble_result = ensemble_by_id.get(job_id_str, {})
                        enhanced_ml_overall = ensemble_result.get("overall_score")
                        enhanced_ml_skill_component = ensemble_result.get("skill_score", None)

                        from app.domain.matching.value_objects import Score

                        if (
                            isinstance(enhanced_ml_overall, (int, float))
                            and 0.0 <= enhanced_ml_overall <= 1.0
                        ):
                            overall_score = Score(float(enhanced_ml_overall))
                            skill_score_for_breakdown = (
                                Score(float(enhanced_ml_skill_component))
                                if isinstance(enhanced_ml_skill_component, (int, float))
                                and 0.0 <= enhanced_ml_skill_component <= 1.0
                                else skill_score_domain
                            )
                        else:
                            # Fallback to domain aggregation
                            overall_score = matching_service.calculate_overall_score(
                                skill_score=skill_score_domain,
                                experience_score=experience_score,
                                location_score=skill_score_domain,
                                salary_score=skill_score_domain,
                            )
                            skill_score_for_breakdown = skill_score_domain

                        confidence = matching_service.calculate_match_confidence(
                            skill_score=skill_score_for_breakdown,
                            experience_score=experience_score,
                            data_completeness=0.95,  # Higher confidence for Enhanced ML
                        )

                        # Safe string conversion
                        def _to_str_if_needed(value):
                            try:
                                return str(value) if value is not None else None
                            except Exception:
                                return None

                        apply_link_value = job_dict.get("apply_link")
                        if apply_link_value is not None and not isinstance(apply_link_value, str):
                            apply_link_value = _to_str_if_needed(apply_link_value)

                        job_title_value = job_dict.get("job_title", "")
                        company_name_value = job_dict.get("company_name", "")
                        location_value = job_dict.get("location", "")
                        job_type_value = job_dict.get("job_type", "")

                        match_result = MatchResult(
                            job_id=job_id_str,
                            original_job_id=job_dict.get("job_id", job_dict.get("id")),
                            overall_score=overall_score,
                            confidence=confidence,
                            matching_skills=matching_skills,
                            missing_skills=missing_skills,
                            score_breakdown={
                                "skills": skill_score_for_breakdown.value,
                                "experience": experience_score.value,
                                "overall": overall_score.value,
                                "enhanced_ml_enabled": True,
                            },
                            explanation=f"Enhanced ML match score: {overall_score.percentage():.1f}% - "
                            f"{len(matching_skills)} matching skills, "
                            f"{len(missing_skills)} missing skills (Advanced ML)",
                            job_title=_to_str_if_needed(job_title_value) or "",
                            company_name=_to_str_if_needed(company_name_value) or "",
                            location=_to_str_if_needed(location_value) or "",
                            job_type=_to_str_if_needed(job_type_value) or "",
                            apply_link=apply_link_value,
                            rank_position=i + 1,
                        )
                        match_results.append(match_result)

                    matching_service.repository.update_match_progress_sync(
                        match_uuid, 95, "Enhanced ML Pipeline: Finalizing results"
                    )

                    # Complete the match using domain service
                    completed = matching_service.repository.complete_match_sync(
                        match_uuid, match_results, parsed_resume_id or 0
                    )

                    if not completed:
                        raise Exception("Failed to complete Enhanced ML Pipeline match job")

                    logger.info(
                        f"Enhanced ML Pipeline match job {match_job_id} completed successfully with {len(match_results)} results"
                    )

                    return {
                        "status": "success",
                        "match_job_id": match_job_id,
                        "matches_count": len(match_results),
                        "parsed_resume_id": parsed_resume_id,
                        "processing_duration": match.get_processing_duration(),
                        "enhanced_ml_processed": True,
                        "features_used": ["ensemble_matching", "domain_services", "advanced_scoring"],
                    }

                except Exception as e:
                    logger.error(f"Enhanced ML Pipeline matching process failed: {e}", exc_info=True)
                    matching_service.repository.fail_match_sync(
                        match_uuid, f"Enhanced ML Pipeline matching failed: {str(e)}"
                    )
                    return {
                        "status": "error", 
                        "error": f"Enhanced ML Pipeline matching failed: {str(e)}"
                    }

            except Exception as e:
                logger.error(f"Enhanced ML Pipeline unexpected error: {e}", exc_info=True)
                try:
                    if "match_uuid" in locals():
                        matching_service.repository.fail_match_sync(
                            match_uuid, f"Enhanced ML Pipeline unexpected error: {str(e)}"
                        )
                except Exception:
                    pass
                return {
                    "status": "error", 
                    "error": f"Enhanced ML Pipeline unexpected error: {str(e)}"
                }

    except Exception as e:
        logger.error(f"Enhanced ML Pipeline failed to initialize services: {e}", exc_info=True)
        return {
            "status": "error", 
            "error": f"Enhanced ML Pipeline service initialization failed: {str(e)}"
        }


@app.task
def test_simple_task():
    """Simple test task to verify domain service integration."""
    logger.info("Testing domain service integration")

    try:
        db = next(get_db())
        matching_repo = SQLAlchemyMatchingRepository(db)
        matching_service = MatchingDomainService(matching_repo)

        # Test basic functionality
        skills = Skills([Skill("python"), Skill("javascript")])
        logger.info(f"Created skills object with {len(skills)} skills")

        return {
            "status": "success",
            "message": "Domain service integration working",
            "skills_test": [skill.name for skill in skills.skills],
        }

    except Exception as e:
        logger.error(f"Test task failed: {e}")
        return {"status": "error", "error": str(e)}

    finally:
        try:
            db.close()
        except:
            pass
