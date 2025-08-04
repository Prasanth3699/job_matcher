"""
Standardized API response utilities for the resume job matcher service.

This module provides centralized response formatting functionality to ensure
consistent API responses across all endpoints following industry standards.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import logging

from app.core.constants import (
    APIConstants,
    HTTPStatusMessages,
    ErrorCodes,
    ErrorMessages,
    EndpointPaths,
    ResponseFields,
)
from app.utils.serialization import JSONSerializer

logger = logging.getLogger(__name__)


class APIResponse:
    """
    Standardized API response builder following best practices.

    Provides consistent response format across all API endpoints with
    proper error handling, correlation IDs, and metadata.
    """

    @staticmethod
    def success(
        data: Any = None,
        message: str = APIConstants.SUCCESS_DEFAULT,
        meta: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        status_code: int = 200,
    ) -> Dict[str, Any]:
        """
        Create a standardized success response.

        Args:
            data: Response data (will be serialized)
            message: Success message
            meta: Additional metadata
            correlation_id: Request correlation ID
            status_code: HTTP status code

        Returns:
            Standardized success response dictionary
        """
        try:
            # Serialize data to ensure JSON compatibility
            serialized_data = (
                JSONSerializer.make_serializable(data) if data is not None else None
            )

            response = {
                ResponseFields.SUCCESS: True,
                ResponseFields.MESSAGE: message,
                ResponseFields.DATA: serialized_data,
                ResponseFields.META: meta or {},
                ResponseFields.TIMESTAMP: datetime.utcnow().isoformat(),
                ResponseFields.CORRELATION_ID: correlation_id or str(uuid.uuid4()),
                ResponseFields.VERSION: APIConstants.API_VERSION,
            }

            # Add status code to meta if different from 200
            if status_code != 200:
                response[ResponseFields.META]["status_code"] = status_code

            return response

        except Exception as e:
            logger.error(f"Failed to create success response: {e}")
            # Fallback to basic response if serialization fails
            return {
                ResponseFields.SUCCESS: True,
                ResponseFields.MESSAGE: message,
                ResponseFields.DATA: None,
                ResponseFields.META: {"serialization_error": str(e)},
                ResponseFields.TIMESTAMP: datetime.utcnow().isoformat(),
                ResponseFields.CORRELATION_ID: correlation_id or str(uuid.uuid4()),
                ResponseFields.VERSION: APIConstants.API_VERSION,
            }

    @staticmethod
    def error(
        error_code: str,
        message: Optional[str] = None,
        details: Any = None,
        correlation_id: Optional[str] = None,
        status_code: int = 400,
    ) -> Dict[str, Any]:
        """
        Create a standardized error response.

        Args:
            error_code: Standardized error code
            message: Error message (auto-generated if not provided)
            details: Additional error details
            correlation_id: Request correlation ID
            status_code: HTTP status code

        Returns:
            Standardized error response dictionary
        """
        try:
            # Get default message if not provided
            if not message:
                message = ErrorMessages.get_message(error_code)

            # Serialize details to ensure JSON compatibility
            serialized_details = (
                JSONSerializer.make_serializable(details)
                if details is not None
                else None
            )

            response = {
                ResponseFields.SUCCESS: False,
                ResponseFields.ERROR: {
                    ResponseFields.ERROR_CODE: error_code,
                    ResponseFields.ERROR_MESSAGE: message,
                    ResponseFields.ERROR_DETAILS: serialized_details,
                },
                ResponseFields.TIMESTAMP: datetime.utcnow().isoformat(),
                ResponseFields.CORRELATION_ID: correlation_id or str(uuid.uuid4()),
                ResponseFields.VERSION: APIConstants.API_VERSION,
            }

            return response

        except Exception as e:
            logger.error(f"Failed to create error response: {e}")
            # Fallback to basic error response
            return {
                ResponseFields.SUCCESS: False,
                ResponseFields.ERROR: {
                    ResponseFields.ERROR_CODE: ErrorCodes.SYSTEM_INTERNAL_ERROR,
                    ResponseFields.ERROR_MESSAGE: "An internal error occurred while formatting the response",
                    ResponseFields.ERROR_DETAILS: {"original_error": str(e)},
                },
                ResponseFields.TIMESTAMP: datetime.utcnow().isoformat(),
                ResponseFields.CORRELATION_ID: correlation_id or str(uuid.uuid4()),
                ResponseFields.VERSION: APIConstants.API_VERSION,
            }

    @staticmethod
    def validation_error(
        validation_errors: List[Dict[str, Any]], correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized validation error response.

        Args:
            validation_errors: List of validation error dictionaries
            correlation_id: Request correlation ID

        Returns:
            Standardized validation error response
        """
        return APIResponse.error(
            error_code=ErrorCodes.VALIDATION_INVALID_FORMAT,
            message="Request validation failed",
            details={"validation_errors": validation_errors},
            correlation_id=correlation_id,
            status_code=422,
        )

    @staticmethod
    def not_found(
        resource_type: str = "Resource",
        resource_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a standardized not found error response.

        Args:
            resource_type: Type of resource that was not found
            resource_id: ID of the resource that was not found
            correlation_id: Request correlation ID

        Returns:
            Standardized not found error response
        """
        if resource_id:
            message = f"{resource_type} with ID '{resource_id}' not found"
        else:
            message = f"{resource_type} not found"

        return APIResponse.error(
            error_code=ErrorCodes.RESOURCE_MATCH_JOB_NOT_FOUND,
            message=message,
            correlation_id=correlation_id,
            status_code=404,
        )

    @staticmethod
    def unauthorized(
        message: str = "Authentication required", correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized unauthorized error response.

        Args:
            message: Unauthorized error message
            correlation_id: Request correlation ID

        Returns:
            Standardized unauthorized error response
        """
        return APIResponse.error(
            error_code=ErrorCodes.AUTH_TOKEN_INVALID,
            message=message,
            correlation_id=correlation_id,
            status_code=401,
        )

    @staticmethod
    def forbidden(
        message: str = "Access denied", correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized forbidden error response.

        Args:
            message: Forbidden error message
            correlation_id: Request correlation ID

        Returns:
            Standardized forbidden error response
        """
        return APIResponse.error(
            error_code=ErrorCodes.AUTH_INSUFFICIENT_PERMISSIONS,
            message=message,
            correlation_id=correlation_id,
            status_code=403,
        )

    @staticmethod
    def service_unavailable(
        service_name: str, correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized service unavailable error response.

        Args:
            service_name: Name of the unavailable service
            correlation_id: Request correlation ID

        Returns:
            Standardized service unavailable error response
        """
        return APIResponse.error(
            error_code=ErrorCodes.SERVICE_RABBITMQ_UNAVAILABLE,
            message=f"{service_name} service is temporarily unavailable",
            correlation_id=correlation_id,
            status_code=503,
        )


class MatchJobResponseBuilder:
    """
    Specialized response builder for match job operations.

    Provides convenient methods for creating responses specific to
    the resume matching workflow.
    """

    @staticmethod
    def job_accepted(
        match_job_id: str, correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create response for accepted match job.

        Args:
            match_job_id: ID of the accepted match job
            correlation_id: Request correlation ID

        Returns:
            Standardized job accepted response
        """
        endpoints = {
            ResponseFields.STATUS_ENDPOINT: EndpointPaths.get_job_status_path(
                match_job_id
            ),
            ResponseFields.RESULTS_ENDPOINT: EndpointPaths.get_job_results_path(
                match_job_id
            ),
        }

        return APIResponse.success(
            data={
                ResponseFields.MATCH_JOB_ID: match_job_id,
                ResponseFields.STATUS: APIConstants.STATUS_ACCEPTED,
                ResponseFields.ENDPOINTS: endpoints,
            },
            message=APIConstants.MATCH_JOB_QUEUED,
            correlation_id=correlation_id,
            status_code=202,
        )

    @staticmethod
    def job_status(
        match_job_id: str,
        status: str,
        current_step: str,
        progress_percentage: float,
        created_at: Optional[datetime] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        error_message: Optional[str] = None,
        matches_count: Optional[int] = None,
        parsed_resume_id: Optional[int] = None,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create response for match job status.

        Args:
            match_job_id: ID of the match job
            status: Current status of the job
            current_step: Current processing step
            progress_percentage: Progress percentage (0-100)
            created_at: Job creation timestamp
            started_at: Job start timestamp
            completed_at: Job completion timestamp
            error_message: Error message if failed
            matches_count: Number of matches found
            parsed_resume_id: ID of parsed resume
            correlation_id: Request correlation ID

        Returns:
            Standardized job status response
        """
        data = {
            ResponseFields.MATCH_JOB_ID: match_job_id,
            ResponseFields.STATUS: status,
            ResponseFields.CURRENT_STEP: current_step,
            ResponseFields.PROGRESS_PERCENTAGE: progress_percentage,
        }

        # Add timestamps if available
        if created_at:
            data[ResponseFields.CREATED_AT] = created_at.isoformat()
        if started_at:
            data[ResponseFields.STARTED_AT] = started_at.isoformat()
        if completed_at:
            data[ResponseFields.COMPLETED_AT] = completed_at.isoformat()

        # Add error message if failed
        if status == APIConstants.STATUS_FAILED and error_message:
            data[ResponseFields.ERROR_MESSAGE_FIELD] = error_message

        # Add results info if completed
        if status == APIConstants.STATUS_COMPLETED:
            data[ResponseFields.RESULTS_AVAILABLE] = True
            if matches_count is not None:
                data[ResponseFields.MATCHES_COUNT] = matches_count
            if parsed_resume_id is not None:
                data[ResponseFields.PARSED_RESUME_ID] = parsed_resume_id

        return APIResponse.success(
            data=data,
            message=f"Match job status: {status}",
            correlation_id=correlation_id,
        )

    @staticmethod
    def job_results(
        match_job_id: str,
        matches: List[Dict[str, Any]],
        parsed_resume_id: Optional[int] = None,
        completed_at: Optional[datetime] = None,
        processing_time_seconds: Optional[float] = None,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create response for match job results.

        Args:
            match_job_id: ID of the match job
            matches: List of job matches
            parsed_resume_id: ID of parsed resume
            completed_at: Job completion timestamp
            processing_time_seconds: Total processing time
            correlation_id: Request correlation ID

        Returns:
            Standardized job results response
        """
        data = {
            ResponseFields.MATCH_JOB_ID: match_job_id,
            ResponseFields.STATUS: APIConstants.STATUS_COMPLETED,
            ResponseFields.MATCHES: matches,
        }

        if parsed_resume_id is not None:
            data[ResponseFields.PARSED_RESUME_ID] = parsed_resume_id

        if completed_at:
            data[ResponseFields.COMPLETED_AT] = completed_at.isoformat()

        if processing_time_seconds is not None:
            data[ResponseFields.PROCESSING_TIME_SECONDS] = processing_time_seconds

        meta = {"matches_count": len(matches), "processing_completed": True}

        return APIResponse.success(
            data=data,
            message=APIConstants.MATCH_JOB_COMPLETED,
            meta=meta,
            correlation_id=correlation_id,
        )

    @staticmethod
    def job_still_processing(
        match_job_id: str,
        current_step: str,
        progress_percentage: float,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create response for job still processing.

        Args:
            match_job_id: ID of the match job
            current_step: Current processing step
            progress_percentage: Progress percentage
            correlation_id: Request correlation ID

        Returns:
            Standardized still processing response
        """
        return APIResponse.error(
            error_code=ErrorCodes.PROCESSING_TIMEOUT,
            message=f"Match job is still processing. Current step: {current_step}",
            details={
                ResponseFields.MATCH_JOB_ID: match_job_id,
                ResponseFields.CURRENT_STEP: current_step,
                ResponseFields.PROGRESS_PERCENTAGE: progress_percentage,
            },
            correlation_id=correlation_id,
            status_code=202,
        )


class ResponseHelper:
    """
    Helper utilities for response handling and HTTP responses.
    """

    @staticmethod
    def get_correlation_id(request: Optional[Request] = None) -> str:
        """
        Extract or generate correlation ID for request tracking.

        Args:
            request: FastAPI request object

        Returns:
            Correlation ID string
        """
        if request and hasattr(request, "headers"):
            # Try to get correlation ID from headers
            correlation_id = request.headers.get("X-Correlation-ID")
            if correlation_id:
                return correlation_id

            # Try alternative header names
            correlation_id = request.headers.get("X-Request-ID")
            if correlation_id:
                return correlation_id

        # Generate new correlation ID
        return str(uuid.uuid4())

    @staticmethod
    def create_json_response(
        response_data: Dict[str, Any],
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
    ) -> JSONResponse:
        """
        Create FastAPI JSONResponse with proper headers.

        Args:
            response_data: Response data dictionary
            status_code: HTTP status code
            headers: Additional headers

        Returns:
            FastAPI JSONResponse object
        """
        # Add correlation ID to headers if not present
        response_headers = headers or {}
        if ResponseFields.CORRELATION_ID in response_data:
            response_headers["X-Correlation-ID"] = response_data[
                ResponseFields.CORRELATION_ID
            ]

        # Add standard headers
        response_headers.update(
            {
                "X-API-Version": APIConstants.API_VERSION,
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
            }
        )

        return JSONResponse(
            content=response_data, status_code=status_code, headers=response_headers
        )

    @staticmethod
    def handle_exception(
        exc: Exception, correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert exception to standardized error response.

        Args:
            exc: Exception to handle
            correlation_id: Request correlation ID

        Returns:
            Standardized error response
        """
        if isinstance(exc, HTTPException):
            # Handle FastAPI HTTPException
            error_code = getattr(exc, "error_code", ErrorCodes.SYSTEM_INTERNAL_ERROR)
            return APIResponse.error(
                error_code=error_code,
                message=exc.detail,
                details={"status_code": exc.status_code},
                correlation_id=correlation_id,
                status_code=exc.status_code,
            )

        # Handle other exceptions
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return APIResponse.error(
            error_code=ErrorCodes.SYSTEM_INTERNAL_ERROR,
            message="An unexpected error occurred",
            details={"exception_type": type(exc).__name__, "message": str(exc)},
            correlation_id=correlation_id,
            status_code=500,
        )


# Convenience functions for common response patterns
def success_response(
    data: Any = None,
    message: str = APIConstants.SUCCESS_DEFAULT,
    correlation_id: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience function for success responses."""
    return APIResponse.success(data, message, correlation_id=correlation_id, **kwargs)


def error_response(
    error_code: str,
    message: Optional[str] = None,
    correlation_id: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience function for error responses."""
    return APIResponse.error(
        error_code, message, correlation_id=correlation_id, **kwargs
    )


def validation_error_response(
    validation_errors: List[Dict[str, Any]], correlation_id: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function for validation error responses."""
    return APIResponse.validation_error(validation_errors, correlation_id)


def not_found_response(
    resource_type: str = "Resource",
    resource_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function for not found responses."""
    return APIResponse.not_found(resource_type, resource_id, correlation_id)
