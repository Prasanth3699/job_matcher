from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import List, Optional
import base64
import json
import uuid
from pathlib import Path
from pydantic import BaseModel, ValidationError

from ..utils.logger import logger
from .dependencies import (
    get_resume_parser,
    get_resume_extractor,
    get_job_processor,
    get_matching_service,
    get_secure_resume_file,
)
from .api_models import (
    JobPostingRequest,
    ResumeUploadRequest,
    MatchResultResponse,
)

router = APIRouter(
    prefix="/v1/matching",
    tags=["matching"],
    responses={404: {"description": "Not found"}},
)


class MatchErrorResponse(BaseModel):
    detail: str
    error_type: str
    context: Optional[str] = None
    solution: Optional[str] = None


@router.get("/")
async def test():
    """Test function to ensure the router is working."""
    return {"message": "Resume Matcher API is working"}


@router.post(
    "/upload",
    response_model=List[MatchResultResponse],
    responses={400: {"model": MatchErrorResponse}, 500: {"model": MatchErrorResponse}},
)
async def match_resume_upload(
    resume_upload: ResumeUploadRequest,
    resume_file: Path = Depends(get_secure_resume_file),
    resume_parser=Depends(get_resume_parser),
    resume_extractor=Depends(get_resume_extractor),
    job_processor=Depends(get_job_processor),
    matching_service=Depends(get_matching_service),
):
    """Match a resume (file upload) against the system's job database."""
    try:
        raw_text, metadata = resume_parser.parse(resume_file)
        resume_data = resume_extractor.extract(raw_text, metadata)

        resume_data.normalized_features = {
            "job_type_preference": (
                resume_upload.preferred_job_types[0].value
                if resume_upload.preferred_job_types
                else "any"
            ),
            "location_preference": (
                resume_upload.preferred_locations[0]
                if resume_upload.preferred_locations
                else "any"
            ),
            "salary_expectation": resume_upload.salary_expectation,
            "target_title": resume_upload.target_title,
            "preferred_companies": [
                c.lower() for c in resume_upload.preferred_companies
            ],
        }

        jobs_data = []
        results = matching_service.match_resume_to_jobs(
            jsonable_encoder(resume_data), jobs_data
        )

        return results

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=MatchErrorResponse(
                detail="Validation error",
                error_type="ValidationError",
                context=str(e),
                solution="Check all required fields and data types",
            ).model_dump(),
        )
    except Exception as exc:
        logger.error(f"Resume matching failed: {str(exc)}", exc_info=exc)
        raise HTTPException(
            status_code=500,
            detail=MatchErrorResponse(
                detail="Resume matching failed", error_type="InternalServerError"
            ).model_dump(),
        )
    finally:
        if resume_file.exists():
            try:
                resume_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {str(e)}")


@router.post(
    "/job",
    response_model=MatchResultResponse,
    responses={400: {"model": MatchErrorResponse}, 500: {"model": MatchErrorResponse}},
)
async def match_resume_to_job(
    jobs: str = Form(...),
    resume_upload: str = Form(...),
    resume_file: UploadFile = File(...),
    resume_parser=Depends(get_resume_parser),
    resume_extractor=Depends(get_resume_extractor),
    job_processor=Depends(get_job_processor),
    matching_service=Depends(get_matching_service),
):
    """Match a resume against multiple job postings with enhanced validation and error handling."""
    temp_path = None
    try:
        # Validate and process file upload
        if not resume_file.filename:
            raise HTTPException(
                status_code=400,
                detail=MatchErrorResponse(
                    detail="No file name provided", error_type="ValidationError"
                ).model_dump(),
            )

        if resume_file.size == 0:
            raise HTTPException(
                status_code=400,
                detail=MatchErrorResponse(
                    detail="Uploaded file is empty", error_type="ValidationError"
                ).model_dump(),
            )

        file_extension = resume_file.filename.split(".")[-1].lower()
        if file_extension not in ["pdf", "docx", "txt"]:
            raise HTTPException(
                status_code=400,
                detail=MatchErrorResponse(
                    detail="Unsupported file format. Only PDF, DOCX, and TXT are allowed",
                    error_type="ValidationError",
                ).model_dump(),
            )

        # Parse JSON inputs
        try:
            jobs_data = json.loads(jobs.strip('"'))
            resume_upload_data = json.loads(resume_upload.strip('"'))
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=MatchErrorResponse(
                    detail="Invalid JSON format",
                    error_type="ValidationError",
                    context=str(e),
                    solution="Please validate your JSON input",
                ).model_dump(),
            )

        # Process file content and validate models
        try:
            file_content = base64.b64encode(await resume_file.read()).decode("utf-8")
            resume_upload_data.update(
                {"file_content": file_content, "file_format": file_extension}
            )

            jobs_list = [JobPostingRequest(**job) for job in jobs_data]
            resume_upload_obj = ResumeUploadRequest(**resume_upload_data)
        except ValidationError as e:
            raise HTTPException(
                status_code=400,
                detail=MatchErrorResponse(
                    detail="Validation error",
                    error_type="ValidationError",
                    context=str(e),
                    solution="Check all required fields and data types",
                ).model_dump(),
            )

        # Create temporary file
        try:
            temp_path = Path(f"temp_{uuid.uuid4().hex}_{resume_file.filename}")
            with open(temp_path, "wb") as buffer:
                await resume_file.seek(0)
                buffer.write(await resume_file.read())
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=MatchErrorResponse(
                    detail="Failed to process uploaded file",
                    error_type="InternalServerError",
                ).model_dump(),
            )

        # Parse and extract resume
        try:
            raw_text, metadata = resume_parser.parse(temp_path)
            resume_data = resume_extractor.extract(raw_text, metadata)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=MatchErrorResponse(
                    detail="Resume parsing failed",
                    error_type="ValidationError",
                    context=str(e),
                    solution="Please check your resume file format and content",
                ).model_dump(),
            )

        # Prepare resume features
        resume_data.normalized_features = {
            "job_type_preference": (
                resume_upload_obj.preferred_job_types[0].value
                if resume_upload_obj.preferred_job_types
                else "any"
            ),
            "location_preference": (
                resume_upload_obj.preferred_locations[0]
                if resume_upload_obj.preferred_locations
                else "any"
            ),
            "salary_expectation": resume_upload_obj.salary_expectation
            or "not specified",
            "target_title": resume_upload_obj.target_title or "not specified",
            "preferred_companies": (
                [c.lower() for c in resume_upload_obj.preferred_companies]
                if resume_upload_obj.preferred_companies
                else []
            ),
        }

        # Process jobs and match
        try:
            processed_jobs = job_processor.process_job_batch(
                [jsonable_encoder(job) for job in jobs_list]
            )
            results = matching_service.match_resume_to_jobs(
                jsonable_encoder(resume_data), processed_jobs
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=MatchErrorResponse(
                    detail="Matching process failed",
                    error_type="InternalServerError",
                    context=str(e),
                    solution="Please try again or contact support",
                ).model_dump(),
            )

        if not results:
            raise HTTPException(
                status_code=404,
                detail=MatchErrorResponse(
                    detail="No matching results found", error_type="NotFound"
                ).model_dump(),
            )

        return results[0]

    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                logger.warning(f"Temp file cleanup failed: {str(e)}")
