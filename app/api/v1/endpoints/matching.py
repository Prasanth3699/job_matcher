from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
)
from typing import List, Dict, Any, Optional  # Added Optional
import json
import uuid
from sqlalchemy.orm import Session
from pathlib import Path

from app.core.auth import get_current_user
from app.db.session import get_db
from app.services.job_service import JobService  # Assuming this exists and works
from app.services.match_history_service import MatchHistoryService

# Use the schema for response model if defined, otherwise base
from app.schemas.match_history import (
    MatchHistory,
    MatchHistoryBase,
)  # Ensure MatchHistoryBase/MatchHistory includes parsed_resume_id
from app.utils.logger import logger
from app.services.dependencies import (
    get_resume_parser,
    get_resume_extractor,
    get_job_processor,
    get_matching_service,
    # get_secure_resume_file, # This dependency wasn't used, commented out
)


from app.schemas.matching import (
    MatchErrorResponse,
    JobMatchResult,
    MatchResponse,
    JobDetails,
)
from app.utils.file_helpers import cleanup_temp_file

# Ensure the client is correctly imported
from app.services.profile_client import ProfileServiceClient

router = APIRouter(
    tags=["matching"],
    responses={404: {"description": "Not found"}},
)


# Helper function to safely convert values to string
def safe_str(val):
    """Safely converts a value to a string, returning empty string for None."""
    if val is None:
        return ""
    # Convert known non-str types to str
    if not isinstance(val, str):
        try:
            return str(val)
        except Exception:  # Catch potential errors during string conversion
            logger.warning(f"Could not convert value of type {type(val)} to string.")
            return ""
    return val


# Helper function to sanitize sequences (lists) into lists of non-empty strings
def sanitize_sequence(seq):
    """Ensure all elements in seq are non-empty strings, return cleaned list."""
    if not isinstance(seq, (list, tuple, set)):  # Check if it's an iterable sequence
        if seq is not None and safe_str(seq).strip():
            return [safe_str(seq).strip()]  # Handle single non-list items
        else:
            return []
    # Process actual sequences
    return [safe_str(x).strip() for x in seq if x is not None and safe_str(x).strip()]


@router.get(
    "/history", response_model=List[MatchHistoryBase]
)  # Use Base or specific response schema
async def get_match_history(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 10,
):
    """
    Retrieves the match history for the currently authenticated user.
    """
    user_id = current_user.get("user_id")
    if user_id is None:
        logger.error("User ID not found in token payload during history fetch.")
        raise HTTPException(
            status_code=401, detail="Invalid authentication token."
        )  # Or 500 if it's an internal issue

    return MatchHistoryService.get_user_match_history(db, user_id, limit)


@router.post(
    "/new-match-async",
    status_code=202,  # 202 Accepted - Request accepted for processing
)
async def new_match_async(
    resume_file: UploadFile = File(
        ..., description="The user's resume file (.pdf, .docx, etc.)."
    ),
    job_ids: str = Form(
        ..., description="Comma-separated string of job IDs to match against."
    ),
    preferences: str = Form(
        ...,
        description="JSON string containing user's job preferences (e.g., location, salary).",
    ),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Starts an asynchronous resume matching job and returns immediately with a job ID.
    The actual ML processing happens in the background via Celery.

    Returns:
    - match_job_id: Use this to check status and get results
    - status: "accepted"
    - message: Instructions for checking progress
    """
    from app.models.match_job import MatchJob
    from app.services.task_queue import process_match_job

    try:
        # --- 1. Input Validation ---
        logger.info(
            f"New async match request received for user_id: {current_user.get('user_id')}"
        )

        if (
            not resume_file.filename
            or resume_file.size == 0
            or resume_file.size is None
        ):
            logger.warning("Invalid resume file uploaded.")
            raise HTTPException(
                status_code=400, detail="Invalid or empty resume file provided."
            )

        try:
            pref_data = json.loads(preferences)
            job_ids_list = [
                int(id_str.strip())
                for id_str in job_ids.split(",")
                if id_str.strip().isdigit()
            ]
            if not job_ids_list:
                raise ValueError("No valid job IDs provided.")
            logger.debug(f"Parsed preferences: {pref_data}, Job IDs: {job_ids_list}")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Invalid input format for preferences or job_ids: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format for preferences or job_ids: {e}",
            )

        user_id = current_user.get("user_id")
        auth_token = current_user.get("token")
        if not user_id or not auth_token:
            logger.error("Missing user_id or token in current_user object.")
            raise HTTPException(
                status_code=401, detail="Authentication token invalid or incomplete."
            )

        # --- 2. Read Resume File ---
        file_bytes = await resume_file.read()

        # --- 3. Create Match Job Record ---
        match_job = MatchJob(
            user_id=user_id,
            status="pending",
            resume_filename=resume_file.filename,
            job_ids=job_ids_list,
            preferences=pref_data,
            current_step="Queued for processing",
            progress_percentage=0.0,
        )

        db.add(match_job)
        db.commit()
        db.refresh(match_job)

        match_job_id = str(match_job.id)
        logger.info(f"Created match job with ID: {match_job_id}")

        # --- 4. Start Celery Task ---
        resume_file_data = {"filename": resume_file.filename, "file_bytes": file_bytes}

        user_data = {"user_id": user_id, "token": auth_token}

        # Queue the Celery task
        try:
            task = process_match_job.delay(match_job_id, resume_file_data, user_data)
            logger.info(f"Successfully queued task: {task.id}")
        except Exception as e:
            logger.error(f"Failed to queue Celery task: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to queue background task: {str(e)}"
            )

        # Update match job with task ID
        match_job.task_id = task.id
        db.commit()

        logger.info(f"Queued Celery task {task.id} for match job {match_job_id}")

        # --- 5. Return Response Immediately ---
        return {
            "match_job_id": match_job_id,
            "status": "accepted",
            "message": "Match job has been queued for processing. Use the match_job_id to check status and get results.",
            "endpoints": {
                "status": f"/api/v1/matching/job/{match_job_id}/status",
                "results": f"/api/v1/matching/job/{match_job_id}/results",
            },
        }

    except HTTPException as http_exc:
        logger.warning(
            f"HTTP Exception in async match: {http_exc.status_code} - {http_exc.detail}"
        )
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error in async match: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while queuing the match job.",
        )


@router.get("/job/{match_job_id}/status")
async def get_match_job_status(
    match_job_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get the status of an async match job
    """
    from app.models.match_job import MatchJob

    try:
        # Get match job
        match_job = (
            db.query(MatchJob)
            .filter(
                MatchJob.id == match_job_id,
                MatchJob.user_id == int(current_user.get("user_id")),
            )
            .first()
        )

        if not match_job:
            raise HTTPException(
                status_code=404,
                detail="Match job not found or you don't have access to it",
            )

        response = {
            "match_job_id": match_job_id,
            "status": match_job.status,
            "current_step": match_job.current_step,
            "progress_percentage": match_job.progress_percentage,
            "created_at": (
                match_job.created_at.isoformat() if match_job.created_at else None
            ),
            "started_at": (
                match_job.started_at.isoformat() if match_job.started_at else None
            ),
            "completed_at": (
                match_job.completed_at.isoformat() if match_job.completed_at else None
            ),
        }

        # Add error message if failed
        if match_job.status == "failed" and match_job.error_message:
            response["error_message"] = match_job.error_message

        # Add results info if completed
        if match_job.status == "completed" and match_job.match_results:
            response["results_available"] = True
            response["matches_count"] = len(match_job.match_results)
            response["parsed_resume_id"] = match_job.parsed_resume_id

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting match job status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving match job status")


@router.get("/job/{match_job_id}/results")
async def get_match_job_results(
    match_job_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get the results of a completed match job
    """
    from app.models.match_job import MatchJob

    try:
        # Get match job
        match_job = (
            db.query(MatchJob)
            .filter(
                MatchJob.id == match_job_id,
                MatchJob.user_id == int(current_user.get("user_id")),
            )
            .first()
        )

        if not match_job:
            raise HTTPException(
                status_code=404,
                detail="Match job not found or you don't have access to it",
            )

        if match_job.status == "pending":
            raise HTTPException(
                status_code=202,  # 202 Accepted - still processing
                detail="Match job is still pending. Please check status first.",
            )

        if match_job.status == "processing":
            raise HTTPException(
                status_code=202,  # 202 Accepted - still processing
                detail=f"Match job is still processing. Current step: {match_job.current_step}",
            )

        if match_job.status == "failed":
            raise HTTPException(
                status_code=400, detail=f"Match job failed: {match_job.error_message}"
            )

        if match_job.status != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Match job is in unexpected status: {match_job.status}",
            )

        # Return results
        return {
            "match_job_id": match_job_id,
            "status": "completed",
            "matches": match_job.match_results or [],
            "parsed_resume_id": match_job.parsed_resume_id,
            "completed_at": (
                match_job.completed_at.isoformat() if match_job.completed_at else None
            ),
            "processing_time_seconds": (
                (match_job.completed_at - match_job.started_at).total_seconds()
                if match_job.started_at and match_job.completed_at
                else None
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting match job results: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Error retrieving match job results"
        )
