from datetime import datetime
from typing import Dict, Any
import os
import json
from decimal import Decimal
from celery import Celery
from kombu import Queue, Exchange
from app.utils.logger import logger
from app.db.session import get_db

# from app.core.learning import ModelUpdater, PerformanceMonitor
from app.data.repositories import FeedbackRepository, ModelVersionRepository
from app.core.config import get_settings

settings = get_settings()


def make_json_serializable(obj: Any) -> Any:
    """
    Convert objects to JSON-serializable format.
    Handles HttpUrl, Decimal, and other problematic types.
    """
    if obj is None:
        return None

    # Check for HttpUrl/URL types first, before any iteration attempts
    if hasattr(obj, "__class__"):
        class_name = str(obj.__class__)
        if (
            "HttpUrl" in class_name
            or "AnyUrl" in class_name
            or "pydantic" in class_name.lower()
            or "Url" in class_name
        ):
            try:
                return str(obj)
            except Exception as e:
                logger.warning(f"Failed to convert URL object to string: {e}")
                return str(type(obj))

    # Handle Pydantic models
    if hasattr(obj, "__pydantic_serializer__") or hasattr(obj, "model_dump"):
        try:
            if hasattr(obj, "model_dump"):
                return make_json_serializable(obj.model_dump())
            else:
                return make_json_serializable(dict(obj))
        except Exception as e:
            logger.warning(f"Failed to serialize Pydantic model: {e}")
            return str(obj)

    # Handle basic types
    if isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, Decimal):
        return float(obj)
    elif hasattr(obj, "isoformat"):
        # Handle datetime objects
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return [make_json_serializable(item) for item in obj]
    else:
        # For any other object, try to convert to string as fallback
        try:
            # Test if it's already JSON serializable
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            # Convert to string if not serializable
            logger.debug(
                f"Converting non-serializable object {type(obj)} to string: {obj}"
            )
            try:
                return str(obj)
            except Exception:
                return f"<{type(obj).__name__} object>"


# Configure Celery with environment variables
app = Celery("resume_matcher")
app.conf.update(
    broker_url=settings.CELERY_BROKER_URL,
    result_backend=settings.CELERY_RESULT_BACKEND,
)

# Custom task queue configuration
app.conf.task_queues = (
    Queue("default", Exchange("default"), routing_key="default"),
    Queue("model_training", Exchange("model_training"), routing_key="model.training"),
    Queue("analysis", Exchange("analysis"), routing_key="analysis"),
)

app.conf.task_routes = {
    "app.services.task_queue.retrain_model": {"queue": "model_training"},
    "app.services.task_queue.analyze_resume": {"queue": "analysis"},
    "app.services.task_queue.process_match_job": {"queue": "default"},
    "app.services.task_queue.test_simple_task": {"queue": "default"},
}

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    result_expires=3600,  # Results expire after 1 hour
    task_time_limit=600,  # Hard timeout at 10 minutes
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": 3600},
    # Windows-specific settings
    worker_pool_restarts=True,
    # Ensure tasks are sent to the right queue
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
)


# @app.task(bind=True, max_retries=3)
# def retrain_model(self, days_of_feedback: int = 30):
#     """Celery task for model retraining"""
#     try:
#         db = next(get_db())
#         try:
#             feedback_repo = FeedbackRepository(db)
#             model_repo = ModelVersionRepository(db)
#             model_updater = ModelUpdater()
#             performance_monitor = PerformanceMonitor()

#             # Training logic
#             new_version = model_updater.retrain_model(
#                 feedback_repo, epochs=3, train_size=0.8
#             )

#             # Save and activate new version
#             db_version = model_repo.create_version(
#                 version_name=new_version.version_id,
#                 model_type="hybrid",
#                 storage_path=str(model_updater.model_storage / new_version.version_id),
#                 metrics=new_version.metrics,
#                 description=new_version.description,
#             )

#             model_repo.activate_version(db_version.id)
#             performance_monitor.record_metrics(
#                 new_version.version_id, new_version.metrics
#             )

#             return {
#                 "status": "success",
#                 "version_id": new_version.version_id,
#                 "metrics": new_version.metrics,
#             }
#         finally:
#             db.close()

#     except Exception as e:
#         logger.error(f"Model training failed: {str(e)}")
#         self.retry(exc=e, countdown=60)


@app.task(bind=True)
def process_match_job(self, match_job_id: str, resume_file_data: Dict, user_data: Dict):
    """
    Celery task for processing match job asynchronously
    """
    logger.info(f"Starting process_match_job for job ID: {match_job_id}")

    try:
        # Import dependencies inside the task to avoid import issues
        from app.models.match_job import MatchJob
        from app.services.job_service import JobService
        from app.services.profile_client import ProfileServiceClient
        from app.services.dependencies import (
            get_resume_parser,
            get_resume_extractor,
            get_job_processor,
            get_matching_service,
        )
        import asyncio
        import tempfile
        import os
        from pathlib import Path

        logger.info("All imports successful")

    except Exception as e:
        logger.error(f"Failed to import dependencies: {e}", exc_info=True)
        return {"status": "error", "error": f"Import failed: {str(e)}"}

    try:
        db = next(get_db())
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database connection failed: {e}", exc_info=True)
        return {"status": "error", "error": f"Database connection failed: {str(e)}"}

    try:
        # Get match job from database
        match_job = db.query(MatchJob).filter(MatchJob.id == match_job_id).first()
        if not match_job:
            logger.error(f"Match job {match_job_id} not found")
            return {"status": "error", "error": "Match job not found"}

        # Update status to processing
        match_job.status = "processing"
        match_job.started_at = datetime.utcnow()
        match_job.task_id = self.request.id
        match_job.current_step = "Processing resume"
        match_job.progress_percentage = 10.0
        db.commit()

        try:
            # Step 1: Process resume file
            logger.info(f"Processing resume for match job {match_job_id}")

            # Create temporary file from bytes
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f"_{resume_file_data['filename']}"
            ) as temp_file:
                temp_file.write(resume_file_data["file_bytes"])
                temp_path = Path(temp_file.name)

            try:
                # Parse resume
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

                # Update progress
                match_job.current_step = "Saving resume to profile service"
                match_job.progress_percentage = 30.0
                db.commit()

                # Step 2: Save to profile service (optional - non-blocking)
                try:
                    parsed_resume_response = asyncio.run(
                        ProfileServiceClient.push_parsed_resume(
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
                        match_job.parsed_resume_id = parsed_resume_response["id"]
                        logger.info(
                            f"Resume saved to profile service with ID: {match_job.parsed_resume_id}"
                        )
                    else:
                        logger.warning(
                            "ProfileService returned invalid response, continuing without saving"
                        )

                except Exception as e:
                    logger.warning(
                        f"Failed to save resume to profile service (non-critical): {e}"
                    )
                    # Continue processing even if profile service fails
                    logger.info(
                        "Continuing matching process without profile service save"
                    )

                # Step 3: Add preferences to resume data
                parsed_resume_dict["normalized_features"] = {
                    "job_type_preference": match_job.preferences.get(
                        "preferred_job_types", [""]
                    )[0].lower(),
                    "location_preference": match_job.preferences.get(
                        "preferred_locations", [""]
                    )[0].lower(),
                    "salary_expectation": match_job.preferences.get(
                        "salary_expectation", "not specified"
                    ),
                    "target_title": match_job.preferences.get(
                        "target_title", ""
                    ).lower(),
                    "preferred_companies": [
                        c.lower()
                        for c in match_job.preferences.get("preferred_companies", [])
                    ],
                }

                # Update progress
                match_job.current_step = "Fetching job details via RabbitMQ"
                match_job.progress_percentage = 50.0
                db.commit()

                # Step 4: Fetch jobs via RabbitMQ with fallback
                logger.info(f"Fetching jobs via RabbitMQ for IDs: {match_job.job_ids}")
                try:
                    jobs = asyncio.run(
                        JobService.fetch_jobs(match_job.job_ids, user_data["token"])
                    )

                    if not jobs:
                        logger.warning(
                            f"No job details found for IDs: {match_job.job_ids}"
                        )
                        # Raise exception when no jobs found
                        raise Exception(
                            f"No job details found for IDs: {match_job.job_ids}"
                        )

                except Exception as e:
                    logger.error(f"Error fetching jobs: {e}")
                    raise

                # Update progress
                match_job.current_step = "Processing jobs for matching"
                match_job.progress_percentage = 70.0
                db.commit()

                # Step 5: Process jobs
                processed_jobs = []
                orig_id_by_apply_link = {}

                for job in jobs:
                    job_dict = (
                        job.model_dump() if hasattr(job, "model_dump") else dict(job)
                    )

                    # Create original ID lookup
                    apply_link = job_dict.get("apply_link", "")
                    original_job_id = job_dict.get("job_id", job_dict.get("id"))
                    if apply_link and original_job_id is not None:
                        orig_id_by_apply_link[apply_link] = original_job_id

                    # Process job data
                    processed_job = {
                        "job_title": job_dict.get("job_title", ""),
                        "company_name": job_dict.get("company_name", ""),
                        "location": job_dict.get("location", ""),
                        "job_type": job_dict.get("job_type", ""),
                        "apply_link": job_dict.get("apply_link", ""),
                        "skills": (
                            job_dict.get("skills", [])
                            if isinstance(job_dict.get("skills"), list)
                            else []
                        ),
                        "requirements": (
                            job_dict.get("requirements", [])
                            if isinstance(job_dict.get("requirements"), list)
                            else []
                        ),
                        "responsibilities": (
                            job_dict.get("responsibilities", [])
                            if isinstance(job_dict.get("responsibilities"), list)
                            else []
                        ),
                        "benefits": (
                            job_dict.get("benefits", [])
                            if isinstance(job_dict.get("benefits"), list)
                            else []
                        ),
                        "qualifications": (
                            job_dict.get("qualifications", [])
                            if isinstance(job_dict.get("qualifications"), list)
                            else []
                        ),
                        "description": job_dict.get("description", ""),
                    }

                    # Add normalized features
                    processed_job["normalized_features"] = {
                        "job_type": processed_job["job_type"],
                        "location": processed_job["location"],
                        "salary": job_dict.get("salary", ""),
                        "company_name": processed_job["company_name"],
                        "skills": processed_job["skills"],
                        "title": processed_job["job_title"],
                    }
                    processed_jobs.append(processed_job)

                # Process job batch
                job_processor = get_job_processor()
                if job_processor:
                    processed_jobs = job_processor.process_job_batch(processed_jobs)

                # Convert processed jobs to JSON-serializable format before matching
                logger.debug(
                    f"Jobs before serialization: {len(processed_jobs) if processed_jobs else 'None'}"
                )
                try:
                    serialized_jobs = make_json_serializable(processed_jobs)
                    if serialized_jobs is not None:
                        processed_jobs = serialized_jobs
                        logger.debug(
                            f"Jobs after serialization: {len(processed_jobs) if processed_jobs else 'None'}"
                        )
                    else:
                        logger.warning(
                            "Serialization returned None, keeping original processed_jobs"
                        )
                except Exception as e:
                    logger.error(
                        f"Error during job serialization: {e}, keeping original processed_jobs"
                    )

                # Safety check for processed_jobs
                if not processed_jobs:
                    processed_jobs = []
                    logger.warning(
                        "processed_jobs is empty or None, setting to empty list"
                    )

                # Update progress
                match_job.current_step = "Running ML matching algorithm"
                match_job.progress_percentage = 85.0
                db.commit()

                # Step 6: Perform matching
                matching_service = get_matching_service()
                results = matching_service.match_resume_to_jobs(
                    parsed_resume_dict,
                    processed_jobs,
                    top_n=len(processed_jobs),
                )

                if not results:
                    logger.info("No matches found by matching service")
                    results = []

                # Convert raw results to JSON-serializable format immediately
                try:
                    serialized_results = make_json_serializable(results)
                    if serialized_results is not None:
                        results = serialized_results
                    else:
                        logger.warning(
                            "Results serialization returned None, keeping original results"
                        )
                except Exception as e:
                    logger.error(
                        f"Error during results serialization: {e}, keeping original results"
                    )

                # Safety check for results
                if results is None:
                    results = []
                    logger.warning("Results is None, setting to empty list")

                # Step 7: Format results
                formatted_results = []
                for result in results:
                    job_details_from_match = result.get("job_details", {})
                    apply_link = job_details_from_match.get("apply_link", "")
                    original_id = (
                        orig_id_by_apply_link.get(apply_link) if apply_link else None
                    )

                    # Ensure apply_link is a string, not HttpUrl
                    apply_link_str = str(apply_link) if apply_link else None

                    formatted_result = {
                        "job_id": str(result.get("job_id", "")),
                        "original_job_id": original_id,
                        "overall_score": float(result.get("overall_score", 0.0)),
                        "score_breakdown": make_json_serializable(
                            result.get("score_breakdown", {})
                        ),
                        "missing_skills": (
                            result.get("missing_skills", [])
                            if isinstance(result.get("missing_skills"), list)
                            else []
                        ),
                        "matching_skills": (
                            result.get("matching_skills", [])
                            if isinstance(result.get("matching_skills"), list)
                            else []
                        ),
                        "explanation": str(result.get("explanation", "")),
                        "job_details": {
                            "job_title": str(
                                job_details_from_match.get("job_title", "")
                            ),
                            "company_name": str(
                                job_details_from_match.get("company_name", "")
                            ),
                            "location": str(job_details_from_match.get("location", "")),
                            "job_type": str(job_details_from_match.get("job_type", "")),
                            "apply_link": apply_link_str,
                        },
                    }
                    formatted_results.append(formatted_result)

                # Step 8: Save results and complete
                # Convert to JSON-serializable format
                try:
                    serializable_results = make_json_serializable(formatted_results)
                    if serializable_results is not None:
                        match_job.match_results = serializable_results
                    else:
                        logger.warning(
                            "Final serialization returned None, using formatted_results"
                        )
                        match_job.match_results = formatted_results
                except Exception as e:
                    logger.error(
                        f"Error during final serialization: {e}, using formatted_results"
                    )
                    match_job.match_results = formatted_results
                match_job.status = "completed"
                match_job.completed_at = datetime.utcnow()
                match_job.current_step = "Completed"
                match_job.progress_percentage = 100.0

                # Commit with better error handling
                try:
                    db.commit()
                except Exception as commit_error:
                    logger.error(f"Failed to commit results: {commit_error}")
                    db.rollback()
                    # Try to save without the detailed results as fallback
                    match_job.match_results = {
                        "error": "Results could not be serialized",
                        "count": len(formatted_results),
                    }
                    match_job.status = "completed_with_errors"
                    match_job.error_message = (
                        f"Result serialization failed: {str(commit_error)}"
                    )
                    db.commit()

                logger.info(
                    f"Match job {match_job_id} completed successfully with {len(formatted_results)} matches"
                )

                return {
                    "status": "success",
                    "match_job_id": match_job_id,
                    "matches_count": len(formatted_results),
                    "parsed_resume_id": match_job.parsed_resume_id,
                }

            finally:
                # Clean up temp file
                if temp_path.exists():
                    os.unlink(temp_path)

        except Exception as e:
            # Handle errors
            logger.error(
                f"Error processing match job {match_job_id}: {e}", exc_info=True
            )

            # Mark as failed
            match_job.status = "failed"
            match_job.error_message = str(e)
            match_job.completed_at = datetime.utcnow()
            db.commit()

            return {"status": "error", "match_job_id": match_job_id, "error": str(e)}

    finally:
        db.close()


@app.task(bind=True)
def analyze_resume(self, resume_data: Dict, job_data: Dict = None):
    """Celery task for comprehensive resume analysis"""
    try:
        from app.core.analysis.gap_analyzer import GapAnalyzer

        analyzer = GapAnalyzer()

        # Work history analysis
        work_history = resume_data.get("experiences", [])
        gap_analysis = analyzer.analyze_work_history(work_history)

        # Skill gap analysis (if job data provided)
        skill_analysis = {"missing_skills": [], "categorized_skills": {}}
        if job_data:
            resume_skills = resume_data.get("skills", [])
            job_skills = job_data.get("skills", [])
            skill_analysis = analyzer.analyze_skill_gaps(resume_skills, job_skills)

        # Generate recommendations
        recommendations = analyzer.generate_recommendations(
            gap_analysis, skill_analysis
        )

        return {
            "gap_analysis": gap_analysis,
            "skill_analysis": skill_analysis,
            "recommendations": recommendations,
        }

    except Exception as e:
        logger.error(f"Resume analysis failed: {str(e)}")
        return {"status": "error", "error": str(e)}


def start_workers():
    """Start Celery workers programmatically (for development)"""
    from celery.bin import worker

    worker = worker.worker(app=app)
    worker.run(loglevel="INFO")
