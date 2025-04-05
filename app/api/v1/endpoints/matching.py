from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
)
from typing import List
import json
import uuid
from sqlalchemy.orm import Session
from pathlib import Path

from app.core.auth import get_current_user
from app.db.session import get_db
from app.services.job_service import JobService
from app.services.match_history_service import MatchHistoryService
from app.schemas.match_history import MatchHistory
from app.utils.logger import logger
from app.services.dependencies import (
    get_resume_parser,
    get_resume_extractor,
    get_job_processor,
    get_matching_service,
    get_secure_resume_file,
)


from app.schemas.matching import (
    MatchErrorResponse,
    JobMatchResult,
    JobPostingRequest,
)
from app.utils.file_helpers import cleanup_temp_file


router = APIRouter(
    tags=["matching"],
    responses={404: {"description": "Not found"}},
)


def validate_job_structure(job: dict) -> dict:
    """Ensure job has required structure with normalized_features"""
    job_dict = job.dict() if hasattr(job, "dict") else dict(job)

    # Remove unwanted fields
    job_dict.pop("detail_url", None)

    # Ensure normalized_features exists with proper structure
    if "normalized_features" not in job_dict or not isinstance(
        job_dict["normalized_features"], dict
    ):
        job_dict["normalized_features"] = {}

    # Set default values for required fields
    required_features = {
        "job_type": job_dict.get("job_type", ""),
        "location": job_dict.get("location", ""),
        "salary": job_dict.get("salary", ""),
        "company_name": job_dict.get("company_name", ""),
        "skills": job_dict.get("skills", []),
        "title": job_dict.get("job_title", ""),
    }

    # Update only missing fields to preserve any existing data
    for key, default_value in required_features.items():
        job_dict["normalized_features"].setdefault(key, default_value)

    return job_dict


@router.get("/history", response_model=List[MatchHistory])
async def get_match_history(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 10,
):
    return MatchHistoryService.get_user_match_history(
        db, current_user["user_id"], limit
    )


@router.post(
    "/new-matchs",
    response_model=List[JobMatchResult],  # Updated response model
    responses={
        400: {"model": MatchErrorResponse},
        404: {"model": MatchErrorResponse},
        500: {"model": MatchErrorResponse},
    },
)
async def new_match_endpoint(
    resume_file: UploadFile = File(...),
    job_ids: str = Form(...),
    preferences: str = Form(...),
    resume_parser=Depends(get_resume_parser),
    resume_extractor=Depends(get_resume_extractor),
    job_processor=Depends(get_job_processor),
    matching_service=Depends(get_matching_service),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Endpoint that properly handles the matching service response structure"""
    temp_path = None
    try:
        # Input validation (same as before)
        if not resume_file.filename or resume_file.size == 0:
            raise HTTPException(status_code=400, detail="Invalid resume file")

        try:
            pref_data = json.loads(preferences)
            job_ids_list = [
                int(id_str.strip()) for id_str in job_ids.split(",") if id_str.strip()
            ]

        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail="Invalid input format")

        # Process resume (same as before)
        temp_path = Path(f"temp_{uuid.uuid4().hex}_{resume_file.filename}")
        with open(temp_path, "wb") as buffer:
            await resume_file.seek(0)
            buffer.write(await resume_file.read())

        raw_text, metadata = resume_parser.parse(temp_path)
        resume_data = resume_extractor.extract(raw_text, metadata)

        resume_data.normalized_features = {
            "job_type_preference": pref_data.get("preferred_job_types", [""])[
                0
            ].lower(),
            "location_preference": pref_data.get("preferred_locations", [""])[
                0
            ].lower(),
            "salary_expectation": pref_data.get("salary_expectation", "not specified"),
            "target_title": pref_data.get("target_title", "").lower(),
            "preferred_companies": [
                c.lower() for c in pref_data.get("preferred_companies", [])
            ],
        }

        # Process jobs with proper field mapping
        jobs = await JobService.fetch_jobs(job_ids_list)
        processed_jobs = []

        for job in jobs:
            job_dict = job.model_dump() if hasattr(job, "dict") else dict(job)

            # Map fields to match expected structure
            processed_job = {
                "job_title": job_dict.get("job_title", ""),
                "company_name": job_dict.get("company_name", ""),
                "location": job_dict.get("location", ""),
                "job_type": job_dict.get("job_type", ""),
                "apply_link": str(job_dict.get("apply_link", "")),
                "skills": job_dict.get("skills", []),
                "description": job_dict.get("description", ""),
            }

            # Add normalized_features with all required fields
            processed_job["normalized_features"] = {
                "job_type": processed_job["job_type"],
                "location": processed_job["location"],
                "salary": job_dict.get("salary", ""),
                "company_name": processed_job["company_name"],
                "skills": processed_job["skills"],
                "title": processed_job["job_title"],
            }

            processed_jobs.append(processed_job)

        # Perform matching with proper parameter structure
        processed_jobs = job_processor.process_job_batch(processed_jobs)
        results = matching_service.match_resume_to_jobs(
            vars(resume_data), processed_jobs, top_n=5  # Convert to dict
        )

        if not results:
            raise HTTPException(status_code=404, detail="No matches found")

        # Transform results to match our response model
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "job_id": result.get("job_id", ""),
                    "overall_score": result.get("overall_score", 0),
                    "score_breakdown": result.get("score_breakdown", {}),
                    "missing_skills": result.get("missing_skills", []),
                    "matching_skills": result.get("matching_skills", []),
                    "explanation": result.get("explanation", ""),
                    "job_details": {
                        "job_title": result.get("job_details", {}).get("job_title", ""),
                        "company_name": result.get("job_details", {}).get(
                            "company_name", ""
                        ),
                        "location": result.get("job_details", {}).get("location", ""),
                        "job_type": result.get("job_details", {}).get("job_type", ""),
                        "apply_link": str(
                            result.get("job_details", {}).get("apply_link", "")
                        ),
                    },
                }
            )
        # Save match history
        MatchHistoryService.create_match_history(
            db=db,
            user_id=current_user["user_id"],
            resume_filename=resume_file.filename,
            job_ids=job_ids_list,
            preferences=pref_data,
            results=formatted_results,
        )
        return formatted_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Matching failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=MatchErrorResponse(
                detail="Matching process failed",
                error_type="InternalError",
                context=str(e),
            ).model_dump(),
        )
    finally:
        if temp_path and temp_path.exists():
            background_tasks.add_task(cleanup_temp_file, temp_path)
