from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
    Request,
)
from typing import List, Dict, Any, Optional
import json
import uuid
from sqlalchemy.orm import Session
from pathlib import Path

from app.core.auth import get_current_user
from app.db.session import get_db
from app.services.job_service import JobService
from app.services.match_history_service import MatchHistoryService
from app.dependencies import get_matching_service, get_job_service, get_user_service
from app.domain.matching.services import MatchingDomainService
from app.domain.jobs.services import JobDomainService
from app.domain.users.services import UserDomainService
from app.services.task_queue_domain import process_match_job_domain, process_match_job_enhanced_ml


# Use the schema for response model if defined, otherwise base
from app.schemas.match_history import (
    MatchHistory,
    MatchHistoryBase,
)
from app.utils.logger import logger
from app.utils.responses import (
    MatchJobResponseBuilder,
    ResponseHelper,
    success_response,
    error_response,
    not_found_response,
)
from app.utils.validation import DataValidator, FileValidator, BusinessLogicValidator
from app.utils.file_processing import SecureFileHandler
from app.core.constants import (
    APIConstants,
    ErrorCodes,
    ErrorMessages,
    BusinessRules,
    HTTPStatusMessages,
)

# Ensure the client is correctly imported
from app.services.profile_client import ProfileServiceClient

router = APIRouter(
    tags=["matching"],
    responses={404: {"description": "Not found"}},
)


@router.get("/history", response_model=List[MatchHistoryBase])
async def get_match_history(
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 10,
):
    """
    Retrieves the match history for the currently authenticated user.
    """
    correlation_id = ResponseHelper.get_correlation_id(request)

    try:
        user_id = current_user.get("user_id")
        if user_id is None:
            logger.error("User ID not found in token payload during history fetch.")
            return error_response(
                error_code=ErrorCodes.AUTH_TOKEN_INVALID,
                message="Invalid authentication token",
                correlation_id=correlation_id,
            )

        history = MatchHistoryService.get_user_match_history(db, user_id, limit)

        return success_response(
            data=history,
            message="Match history retrieved successfully",
            correlation_id=correlation_id,
        )

    except Exception as e:
        logger.error(f"Error retrieving match history: {e}", exc_info=True)
        return error_response(
            error_code=ErrorCodes.SYSTEM_INTERNAL_ERROR,
            message="Failed to retrieve match history",
            correlation_id=correlation_id,
        )


@router.post("/new-match-async", status_code=202)
async def new_match_async(
    request: Request,
    resume_file: UploadFile = File(
        ..., description="The user's resume file (.pdf, .docx, etc.)."
    ),
    job_ids: str = Form(
        ..., description="Comma-separated string of job IDs to match against."
    ),
    preferences: str = Form(
        ..., description="JSON string containing user's job preferences."
    ),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    matching_service: MatchingDomainService = Depends(get_matching_service),
):
    """
    Starts an asynchronous resume matching job and returns immediately with a job ID.
    The actual ML processing happens in the background via Celery.
    """
    from app.models.match_job import MatchJob

    correlation_id = ResponseHelper.get_correlation_id(request)

    try:
        logger.info(
            f"New async match request received for user_id: {current_user.get('user_id')}"
        )

        # --- 1. Input Validation ---

        # Validate user authentication
        user_id = current_user.get("user_id")
        auth_token = current_user.get("token")
        if not user_id or not auth_token:
            logger.error("Missing user_id or token in current_user object.")
            return error_response(
                error_code=ErrorCodes.AUTH_TOKEN_INVALID,
                message="Authentication token invalid or incomplete",
                correlation_id=correlation_id,
            )

        # Validate file upload
        if (
            not resume_file.filename
            or resume_file.size == 0
            or resume_file.size is None
        ):
            logger.warning("Invalid resume file uploaded.")
            return error_response(
                error_code=ErrorCodes.VALIDATION_INVALID_FILE_TYPE,
                message="Invalid or empty resume file provided",
                correlation_id=correlation_id,
            )

        # Read and validate file content
        file_bytes = await resume_file.read()
        file_validation = FileValidator.validate_file_upload(
            filename=resume_file.filename,
            file_size=len(file_bytes),
            content_type=resume_file.content_type,
            file_content=file_bytes,
        )

        if not file_validation.is_valid:
            return error_response(
                error_code=ErrorCodes.VALIDATION_INVALID_FILE_TYPE,
                message="File validation failed",
                details={
                    "validation_errors": [
                        error.to_dict() for error in file_validation.errors
                    ]
                },
                correlation_id=correlation_id,
            )

        # Validate and parse job IDs
        try:
            job_ids_list = [
                int(id_str.strip())
                for id_str in job_ids.split(",")
                if id_str.strip().isdigit()
            ]
            if not job_ids_list:
                raise ValueError("No valid job IDs provided")
        except ValueError as e:
            return error_response(
                error_code=ErrorCodes.VALIDATION_INVALID_JOB_IDS,
                message=f"Invalid job IDs format: {str(e)}",
                correlation_id=correlation_id,
            )

        # Validate job IDs against business rules
        job_ids_validation = BusinessLogicValidator.validate_job_ids(job_ids_list)
        if not job_ids_validation.is_valid:
            return error_response(
                error_code=ErrorCodes.VALIDATION_INVALID_JOB_IDS,
                message="Job IDs validation failed",
                details={
                    "validation_errors": [
                        error.to_dict() for error in job_ids_validation.errors
                    ]
                },
                correlation_id=correlation_id,
            )

        # Validate and parse preferences
        try:
            pref_data = json.loads(preferences)
        except json.JSONDecodeError as e:
            return error_response(
                error_code=ErrorCodes.VALIDATION_INVALID_JSON,
                message=f"Invalid preferences JSON format: {str(e)}",
                correlation_id=correlation_id,
            )

        # Validate preferences against business rules
        preferences_validation = BusinessLogicValidator.validate_user_preferences(
            pref_data
        )
        if not preferences_validation.is_valid:
            return error_response(
                error_code=ErrorCodes.VALIDATION_INVALID_PREFERENCES,
                message="User preferences validation failed",
                details={
                    "validation_errors": [
                        error.to_dict() for error in preferences_validation.errors
                    ]
                },
                correlation_id=correlation_id,
            )

        logger.debug(f"Parsed preferences: {pref_data}, Job IDs: {job_ids_list}")

        # --- 2. Create Match Job Record using Domain Service ---
        try:
            match = await matching_service.create_match_job(
                user_id=user_id,
                resume_filename=resume_file.filename,
                job_ids=[str(jid) for jid in job_ids_list],  # Convert to strings
                preferences=pref_data,
            )
            match_job_id = str(match.id)
            logger.info(f"Created match job with ID: {match_job_id}")
        except ValueError as e:
            logger.error(f"Failed to create match job: {e}")
            return error_response(
                error_code=ErrorCodes.VALIDATION_INVALID_JOB_IDS,
                message=f"Match job creation failed: {str(e)}",
                correlation_id=correlation_id,
            )

        # --- 3. Start Celery Task (using domain-based processing) ---
        resume_file_data = {"filename": resume_file.filename, "file_bytes": file_bytes}
        user_data = {"user_id": user_id, "token": auth_token}

        try:
            # Use domain-based task processing
            from app.services.task_queue_domain import process_match_job_domain

            task = process_match_job_domain.delay(
                match_job_id, resume_file_data, user_data
            )
            logger.info(f"Successfully queued domain-based task: {task.id}")

            # Update match job with task ID using domain service
            await matching_service.start_processing(match.id, task.id)
        except ValueError as e:
            logger.error(f"Failed to start processing: {e}")
            return error_response(
                error_code=ErrorCodes.BUSINESS_INVALID_JOB_STATE,
                message=f"Failed to start processing: {str(e)}",
                correlation_id=correlation_id,
            )
        except Exception as e:
            logger.error(f"Failed to queue Celery task: {e}", exc_info=True)
            return error_response(
                error_code=ErrorCodes.PROCESSING_TASK_QUEUE_FAILED,
                message="Failed to queue background task",
                details={"error": str(e)},
                correlation_id=correlation_id,
            )

        logger.info(f"Queued Celery task {task.id} for match job {match_job_id}")

        # --- 4. Return Standardized Response ---
        return MatchJobResponseBuilder.job_accepted(
            match_job_id=match_job_id, correlation_id=correlation_id
        )

    except Exception as e:
        logger.error(f"Unexpected error in async match: {str(e)}", exc_info=True)
        return error_response(
            error_code=ErrorCodes.SYSTEM_INTERNAL_ERROR,
            message="An unexpected error occurred while queuing the match job",
            correlation_id=correlation_id,
        )


@router.get("/job/{match_job_id}/status")
async def get_match_job_status(
    match_job_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    matching_service: MatchingDomainService = Depends(get_matching_service),
):
    """
    Get the status of an async match job
    """
    from uuid import UUID

    correlation_id = ResponseHelper.get_correlation_id(request)

    try:
        # Get match job using domain service
        try:
            match_job_uuid = UUID(match_job_id)
        except ValueError:
            return error_response(
                error_code=ErrorCodes.VALIDATION_INVALID_FORMAT,
                message="Invalid match job ID format",
                correlation_id=correlation_id,
            )

        match = await matching_service.get_match_by_id(match_job_uuid)
        if not match or match.user_id != int(current_user.get("user_id")):
            return not_found_response(
                resource_type="Match job",
                resource_id=match_job_id,
                correlation_id=correlation_id,
            )

        # Return status using standardized response builder
        return MatchJobResponseBuilder.job_status(
            match_job_id=match_job_id,
            status=match.status.value,
            current_step=match.current_step,
            progress_percentage=match.progress_percentage,
            created_at=match.created_at,
            started_at=match.started_at,
            completed_at=match.completed_at,
            error_message=match.error_message,
            matches_count=(len(match.match_results) if match.match_results else None),
            parsed_resume_id=match.parsed_resume_id,
            correlation_id=correlation_id,
        )

    except Exception as e:
        logger.error(f"Error getting match job status: {e}", exc_info=True)
        return error_response(
            error_code=ErrorCodes.SYSTEM_INTERNAL_ERROR,
            message="Error retrieving match job status",
            correlation_id=correlation_id,
        )


@router.get("/job/{match_job_id}/results")
async def get_match_job_results(
    match_job_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    matching_service: MatchingDomainService = Depends(get_matching_service),
):
    """
    Get the results of a completed match job
    """
    from uuid import UUID

    correlation_id = ResponseHelper.get_correlation_id(request)

    try:
        # Get match job using domain service
        try:
            match_job_uuid = UUID(match_job_id)
        except ValueError:
            return error_response(
                error_code=ErrorCodes.VALIDATION_INVALID_FORMAT,
                message="Invalid match job ID format",
                correlation_id=correlation_id,
            )

        match = await matching_service.get_match_by_id(match_job_uuid)
        if not match or match.user_id != int(current_user.get("user_id")):
            return not_found_response(
                resource_type="Match job",
                resource_id=match_job_id,
                correlation_id=correlation_id,
            )

        # Check job status and return appropriate response
        if match.is_processing() or not match.is_completed():
            return MatchJobResponseBuilder.job_still_processing(
                match_job_id=match_job_id,
                current_step=match.current_step,
                progress_percentage=match.progress_percentage,
                correlation_id=correlation_id,
            )

        if match.is_failed():
            return error_response(
                error_code=ErrorCodes.PROCESSING_RESUME_PARSE_FAILED,
                message=f"Match job failed: {match.error_message}",
                correlation_id=correlation_id,
            )

        if not match.is_completed():
            return error_response(
                error_code=ErrorCodes.BUSINESS_INVALID_JOB_STATE,
                message=f"Match job is in unexpected status: {match.status.value}",
                correlation_id=correlation_id,
            )

        # Calculate processing time
        processing_time_seconds = match.get_processing_duration()

        # Convert domain match results to API format
        api_matches = []
        if match.match_results:
            api_matches = [result.to_dict() for result in match.match_results]

        # Return results using standardized response builder
        return MatchJobResponseBuilder.job_results(
            match_job_id=match_job_id,
            matches=api_matches,
            parsed_resume_id=match.parsed_resume_id,
            completed_at=match.completed_at,
            processing_time_seconds=processing_time_seconds,
            correlation_id=correlation_id,
        )

    except Exception as e:
        logger.error(f"Error getting match job results: {e}", exc_info=True)
        return error_response(
            error_code=ErrorCodes.SYSTEM_INTERNAL_ERROR,
            message="Error retrieving match job results",
            correlation_id=correlation_id,
        )


@router.post("/new-match", status_code=202)
async def new_match(
    request: Request,
    resume_file: UploadFile = File(
        ..., description="The user's resume file (.pdf, .docx, etc.)."
    ),
    job_ids: str = Form(
        ..., description="Comma-separated string of job IDs to match against."
    ),
    preferences: str = Form(
        ..., description="JSON string containing user's job preferences."
    ),
    use_ab_testing: bool = Form(
        False, description="Enable A/B testing for this request."
    ),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    matching_service: MatchingDomainService = Depends(get_matching_service),
):
    """
    Resume matching using domain-driven architecture with optional enhanced ML features.
    Falls back to domain-based processing if enhanced ML is unavailable.
    """
    from app.core.enhanced_ml_config import get_enhanced_ml_manager

    correlation_id = ResponseHelper.get_correlation_id(request)

    try:
        logger.info(
            f"New Enhanced ML Pipeline match request received for user_id: {current_user.get('user_id')}"
        )

        # Check Enhanced ML Pipeline availability
        manager = get_enhanced_ml_manager()
        use_enhanced_ml = manager and manager.is_initialized

        if not use_enhanced_ml:
            logger.info(
                "Enhanced ML Pipeline not available, using domain-based matching"
            )

        # --- 1. Input Validation (same as legacy) ---

        # Validate user authentication
        user_id = current_user.get("user_id")
        auth_token = current_user.get("token")
        if not user_id or not auth_token:
            logger.error("Missing user_id or token in current_user object.")
            return error_response(
                error_code=ErrorCodes.AUTH_TOKEN_INVALID,
                message="Authentication token invalid or incomplete",
                correlation_id=correlation_id,
            )

        # Validate file upload
        if (
            not resume_file.filename
            or resume_file.size == 0
            or resume_file.size is None
        ):
            logger.warning("Invalid resume file uploaded.")
            return error_response(
                error_code=ErrorCodes.VALIDATION_INVALID_FILE_TYPE,
                message="Invalid or empty resume file provided",
                correlation_id=correlation_id,
            )

        # Read and validate file content
        file_bytes = await resume_file.read()
        file_validation = FileValidator.validate_file_upload(
            filename=resume_file.filename,
            file_size=len(file_bytes),
            content_type=resume_file.content_type,
            file_content=file_bytes,
        )

        if not file_validation.is_valid:
            return error_response(
                error_code=ErrorCodes.VALIDATION_INVALID_FILE_TYPE,
                message="File validation failed",
                details={
                    "validation_errors": [
                        error.to_dict() for error in file_validation.errors
                    ]
                },
                correlation_id=correlation_id,
            )

        # Validate and parse job IDs
        try:
            job_ids_list = [
                int(id_str.strip())
                for id_str in job_ids.split(",")
                if id_str.strip().isdigit()
            ]
            if not job_ids_list:
                raise ValueError("No valid job IDs provided")
        except ValueError as e:
            return error_response(
                error_code=ErrorCodes.VALIDATION_INVALID_JOB_IDS,
                message=f"Invalid job IDs format: {str(e)}",
                correlation_id=correlation_id,
            )

        # Validate and parse preferences
        try:
            pref_data = json.loads(preferences)
        except json.JSONDecodeError as e:
            return error_response(
                error_code=ErrorCodes.VALIDATION_INVALID_JSON,
                message=f"Invalid preferences JSON format: {str(e)}",
                correlation_id=correlation_id,
            )

        # --- 2. Track Analytics Event ---
        if use_enhanced_ml:
            analytics_engine = manager.get_analytics_engine()
            if analytics_engine:
                try:
                    await analytics_engine.track_event(
                        event_type="match_request_enhanced_ml",
                        user_id=str(user_id),
                        session_id=correlation_id,
                        data={
                            "job_count": len(job_ids_list),
                            "file_type": resume_file.content_type,
                            "file_size": len(file_bytes),
                            "ab_testing_enabled": use_ab_testing,
                            "preferences": pref_data,
                        },
                    )
                except Exception as analytics_error:
                    logger.warning(f"Analytics tracking failed: {analytics_error}")
        else:
            logger.info(
                f"Domain-based match request for user_id: {user_id}, {len(job_ids_list)} jobs"
            )

        # --- 3. Create Match Job Record using Domain Service ---
        try:
            match = await matching_service.create_match_job(
                user_id=user_id,
                resume_filename=resume_file.filename,
                job_ids=[str(jid) for jid in job_ids_list],  # Convert to strings
                preferences=pref_data,
            )
            match_job_id = str(match.id)
            pipeline_type = (
                "Enhanced ML Pipeline" if use_enhanced_ml else "Domain-based Pipeline"
            )
            logger.info(f"Created {pipeline_type} match job with ID: {match_job_id}")
        except ValueError as e:
            logger.error(f"Failed to create match job: {e}")
            return error_response(
                error_code=ErrorCodes.VALIDATION_INVALID_JOB_IDS,
                message=f"Match job creation failed: {str(e)}",
                correlation_id=correlation_id,
            )

        # --- 4. Start Celery Task (Domain-based with Enhanced ML support) ---
        resume_file_data = {"filename": resume_file.filename, "file_bytes": file_bytes}
        user_data = {"user_id": user_id, "token": auth_token}

        try:
            if use_enhanced_ml:
                # Use enhanced ML pipeline task from domain-based task queue
                enhanced_ml_config = {
                    "use_ab_testing": use_ab_testing,
                    "correlation_id": correlation_id,
                    "enable_analytics": True,
                    "enable_model_monitoring": True,
                }

                task = process_match_job_enhanced_ml.delay(
                    match_job_id, resume_file_data, user_data, enhanced_ml_config
                )
                logger.info(f"Successfully queued Enhanced ML Pipeline task: {task.id}")
            else:
                # Use domain-based task processing

                task = process_match_job_domain.delay(
                    match_job_id, resume_file_data, user_data
                )
                logger.info(f"Successfully queued domain-based task: {task.id}")

            # Update match job with task ID using domain service
            await matching_service.start_processing(match.id, task.id)
        except ValueError as e:
            logger.error(f"Failed to start processing: {e}")
            return error_response(
                error_code=ErrorCodes.BUSINESS_INVALID_JOB_STATE,
                message=f"Failed to start processing: {str(e)}",
                correlation_id=correlation_id,
            )
        except Exception as e:
            logger.error(f"Failed to queue Celery task: {e}", exc_info=True)
            return error_response(
                error_code=ErrorCodes.PROCESSING_TASK_QUEUE_FAILED,
                message="Failed to queue background task",
                details={"error": str(e)},
                correlation_id=correlation_id,
            )

        pipeline_type = "Enhanced ML Pipeline" if use_enhanced_ml else "Domain-based"
        logger.info(
            f"Queued {pipeline_type} Celery task {task.id} for match job {match_job_id}"
        )

        # --- 5. Return Response ---
        response_data = MatchJobResponseBuilder.job_accepted(
            match_job_id=match_job_id, correlation_id=correlation_id
        )

        # Add pipeline-specific information
        if isinstance(response_data, dict) and "data" in response_data:
            response_data["data"]["enhanced_ml_enabled"] = use_enhanced_ml
            response_data["data"]["domain_based_enabled"] = not use_enhanced_ml
            if use_enhanced_ml:
                response_data["data"]["ab_testing_enabled"] = use_ab_testing
                response_data["data"]["enhanced_features"] = [
                    "semantic_matching",
                    "neural_ranking",
                    "performance_optimization",
                    "real_time_analytics",
                ]
            else:
                response_data["data"]["domain_features"] = [
                    "domain_driven_design",
                    "business_logic_separation",
                    "clean_architecture",
                    "unified_matching_service",
                ]

        return response_data

    except Exception as e:
        logger.error(
            f"Unexpected error in match processing: {str(e)}",
            exc_info=True,
        )
        return error_response(
            error_code=ErrorCodes.SYSTEM_INTERNAL_ERROR,
            message="An unexpected error occurred while queuing the match job",
            correlation_id=correlation_id,
        )
