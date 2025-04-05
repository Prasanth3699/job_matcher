from fastapi import Depends, HTTPException, status
from typing import Annotated

from ..services.api_models import ResumeUploadRequest
from app.core.document_processing import ResumeParser, ResumeExtractor
from app.core.job_processing.service import JobProcessingService
from app.core.matching.service import MatchingService
from app.utils.logger import logger
from app.utils.security import SecurityUtils
from pathlib import Path
import base64
import os


def get_resume_parser() -> ResumeParser:
    return ResumeParser()


def get_resume_extractor() -> ResumeExtractor:
    return ResumeExtractor()


def get_job_processor() -> JobProcessingService:
    return JobProcessingService()


def get_matching_service() -> MatchingService:
    return MatchingService()


def secure_temp_file(content: str) -> Path:
    """Create a secure temp file from base64 content"""
    try:
        decoded = base64.b64decode(content)
        temp_path = SecurityUtils.create_secure_temp_file(decoded)
        return temp_path
    except Exception as e:
        logger.error(f"Temp file creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file content"
        )


def cleanup_temp_file(path: Path):
    """Cleanup temp file after use"""
    try:
        if path.exists():
            os.unlink(path)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {path}: {str(e)}")


def get_secure_resume_file(
    upload: ResumeUploadRequest, temp_file: Annotated[Path, Depends(secure_temp_file)]
) -> Path:
    """Dependency to get validated resume file"""
    try:
        # Validate file extension
        ext = temp_file.suffix.lower()
        if upload.file_format.value not in ext:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File format mismatch. Expected {upload.file_format}, got {ext}",
            )

        # Validate file content
        if os.path.getsize(temp_file) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file content"
            )

        return temp_file
    except HTTPException:
        cleanup_temp_file(temp_file)
        raise
    except Exception as e:
        cleanup_temp_file(temp_file)
        logger.error(f"Resume file validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File processing error",
        )
