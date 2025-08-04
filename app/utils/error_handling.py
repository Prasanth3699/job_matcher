"""
Unified error handling utilities for the resume job matcher service.

This module provides centralized error handling functionality to ensure
consistent error management and logging across the application.
"""

import traceback
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Type
from functools import wraps
import logging
from contextlib import contextmanager

from app.core.constants import (
    ErrorCodes,
    ErrorMessages,
    APIConstants,
    HTTPStatusMessages
)
from app.utils.serialization import JSONSerializer

logger = logging.getLogger(__name__)


class BaseApplicationError(Exception):
    """
    Base exception class for all application-specific errors.
    
    Provides structured error information with error codes, correlation IDs,
    and additional context for debugging and user feedback.
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = ErrorCodes.SYSTEM_INTERNAL_ERROR,
        correlation_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
        user_message: Optional[str] = None
    ):
        self.message = message
        self.error_code = error_code
        self.correlation_id = correlation_id
        self.details = details or {}
        self.original_error = original_error
        self.user_message = user_message or message
        self.timestamp = datetime.utcnow()
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format for API responses."""
        error_dict = {
            'error_code': self.error_code,
            'message': self.message,
            'user_message': self.user_message,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details
        }
        
        if self.correlation_id:
            error_dict['correlation_id'] = self.correlation_id
        
        if self.original_error:
            error_dict['original_error'] = {
                'type': type(self.original_error).__name__,
                'message': str(self.original_error)
            }
        
        return error_dict
    
    def log_error(self, logger_instance: Optional[logging.Logger] = None):
        """Log error with appropriate level and context."""
        log = logger_instance or logger
        
        error_context = {
            'error_code': self.error_code,
            'correlation_id': self.correlation_id,
            'details': self.details
        }
        
        if self.original_error:
            log.error(
                f"Application error: {self.message}",
                extra=error_context,
                exc_info=self.original_error
            )
        else:
            log.error(
                f"Application error: {self.message}",
                extra=error_context
            )


class ValidationError(BaseApplicationError):
    """Exception for validation-related errors."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Any = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if field_name:
            details['field_name'] = field_name
        if field_value is not None:
            details['field_value'] = str(field_value)
        
        super().__init__(
            message=message,
            error_code=ErrorCodes.VALIDATION_INVALID_FORMAT,
            correlation_id=correlation_id,
            details=details,
            **kwargs
        )
        self.field_name = field_name
        self.field_value = field_value


class ProcessingError(BaseApplicationError):
    """Exception for processing-related errors."""
    
    def __init__(
        self,
        message: str,
        processing_stage: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if processing_stage:
            details['processing_stage'] = processing_stage
        
        super().__init__(
            message=message,
            error_code=ErrorCodes.PROCESSING_RESUME_PARSE_FAILED,
            correlation_id=correlation_id,
            details=details,
            **kwargs
        )
        self.processing_stage = processing_stage


class ServiceError(BaseApplicationError):
    """Exception for external service-related errors."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if service_name:
            details['service_name'] = service_name
        
        super().__init__(
            message=message,
            error_code=ErrorCodes.SERVICE_RABBITMQ_UNAVAILABLE,
            correlation_id=correlation_id,
            details=details,
            **kwargs
        )
        self.service_name = service_name


class AuthenticationError(BaseApplicationError):
    """Exception for authentication-related errors."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        correlation_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCodes.AUTH_TOKEN_INVALID,
            correlation_id=correlation_id,
            user_message="Authentication required. Please check your credentials.",
            **kwargs
        )


class AuthorizationError(BaseApplicationError):
    """Exception for authorization-related errors."""
    
    def __init__(
        self,
        message: str = "Access denied",
        resource: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if resource:
            details['resource'] = resource
        
        super().__init__(
            message=message,
            error_code=ErrorCodes.AUTH_INSUFFICIENT_PERMISSIONS,
            correlation_id=correlation_id,
            details=details,
            user_message="You don't have permission to access this resource.",
            **kwargs
        )


class ResourceNotFoundError(BaseApplicationError):
    """Exception for resource not found errors."""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ):
        if resource_id:
            message = f"{resource_type} with ID '{resource_id}' not found"
        else:
            message = f"{resource_type} not found"
        
        details = kwargs.get('details', {})
        details.update({
            'resource_type': resource_type,
            'resource_id': resource_id
        })
        
        super().__init__(
            message=message,
            error_code=ErrorCodes.RESOURCE_MATCH_JOB_NOT_FOUND,
            correlation_id=correlation_id,
            details=details,
            user_message=f"The requested {resource_type.lower()} could not be found.",
            **kwargs
        )


class ErrorHandler:
    """
    Centralized error handling utility class.
    
    Provides methods for error classification, logging, and response formatting.
    """
    
    @staticmethod
    def classify_error(error: Exception) -> Dict[str, Any]:
        """
        Classify error and determine appropriate response information.
        
        Args:
            error: Exception to classify
            
        Returns:
            Dictionary with error classification information
        """
        if isinstance(error, BaseApplicationError):
            return {
                'type': 'application_error',
                'error_code': error.error_code,
                'message': error.message,
                'user_message': error.user_message,
                'status_code': ErrorHandler._get_status_code_for_error_code(error.error_code),
                'details': error.details,
                'should_log': True,
                'log_level': 'error'
            }
        
        elif isinstance(error, ValueError):
            return {
                'type': 'validation_error',
                'error_code': ErrorCodes.VALIDATION_INVALID_FORMAT,
                'message': str(error),
                'user_message': "Invalid input provided. Please check your data and try again.",
                'status_code': 400,
                'details': {'original_error': str(error)},
                'should_log': True,
                'log_level': 'warning'
            }
        
        elif isinstance(error, FileNotFoundError):
            return {
                'type': 'file_error',
                'error_code': ErrorCodes.RESOURCE_RESUME_NOT_FOUND,
                'message': str(error),
                'user_message': "The requested file could not be found.",
                'status_code': 404,
                'details': {'file_path': str(error)},
                'should_log': True,
                'log_level': 'warning'
            }
        
        elif isinstance(error, PermissionError):
            return {
                'type': 'permission_error',
                'error_code': ErrorCodes.AUTH_INSUFFICIENT_PERMISSIONS,
                'message': str(error),
                'user_message': "Permission denied for this operation.",
                'status_code': 403,
                'details': {},
                'should_log': True,
                'log_level': 'error'
            }
        
        elif isinstance(error, ConnectionError):
            return {
                'type': 'connection_error',
                'error_code': ErrorCodes.SERVICE_RABBITMQ_UNAVAILABLE,
                'message': str(error),
                'user_message': "Service temporarily unavailable. Please try again later.",
                'status_code': 503,
                'details': {},
                'should_log': True,
                'log_level': 'error'
            }
        
        elif isinstance(error, TimeoutError):
            return {
                'type': 'timeout_error',
                'error_code': ErrorCodes.PROCESSING_TIMEOUT,
                'message': str(error),
                'user_message': "Request timeout. Please try again.",
                'status_code': 504,
                'details': {},
                'should_log': True,
                'log_level': 'warning'
            }
        
        else:
            # Unknown error - treat as internal server error
            return {
                'type': 'unknown_error',
                'error_code': ErrorCodes.SYSTEM_INTERNAL_ERROR,
                'message': str(error),
                'user_message': "An unexpected error occurred. Please try again later.",
                'status_code': 500,
                'details': {
                    'error_type': type(error).__name__,
                    'error_message': str(error)
                },
                'should_log': True,
                'log_level': 'error'
            }
    
    @staticmethod
    def _get_status_code_for_error_code(error_code: str) -> int:
        """Map error codes to HTTP status codes."""
        error_code_mappings = {
            # Authentication errors (401)
            ErrorCodes.AUTH_TOKEN_INVALID: 401,
            ErrorCodes.AUTH_TOKEN_EXPIRED: 401,
            ErrorCodes.AUTH_TOKEN_MISSING: 401,
            ErrorCodes.AUTH_USER_NOT_FOUND: 401,
            
            # Authorization errors (403)
            ErrorCodes.AUTH_INSUFFICIENT_PERMISSIONS: 403,
            
            # Validation errors (400)
            ErrorCodes.VALIDATION_REQUIRED_FIELD_MISSING: 400,
            ErrorCodes.VALIDATION_INVALID_FORMAT: 400,
            ErrorCodes.VALIDATION_VALUE_OUT_OF_RANGE: 400,
            ErrorCodes.VALIDATION_INVALID_FILE_TYPE: 400,
            ErrorCodes.VALIDATION_FILE_TOO_LARGE: 413,
            ErrorCodes.VALIDATION_FILE_TOO_SMALL: 400,
            ErrorCodes.VALIDATION_INVALID_JSON: 400,
            ErrorCodes.VALIDATION_INVALID_JOB_IDS: 400,
            ErrorCodes.VALIDATION_INVALID_PREFERENCES: 400,
            ErrorCodes.VALIDATION_FILE_CORRUPTED: 400,
            
            # Not found errors (404)
            ErrorCodes.RESOURCE_MATCH_JOB_NOT_FOUND: 404,
            ErrorCodes.RESOURCE_USER_NOT_FOUND: 404,
            ErrorCodes.RESOURCE_JOB_NOT_FOUND: 404,
            ErrorCodes.RESOURCE_RESUME_NOT_FOUND: 404,
            ErrorCodes.RESOURCE_MODEL_NOT_FOUND: 404,
            ErrorCodes.RESOURCE_ENDPOINT_NOT_FOUND: 404,
            
            # Processing errors (500)
            ErrorCodes.PROCESSING_RESUME_PARSE_FAILED: 500,
            ErrorCodes.PROCESSING_FEATURE_EXTRACTION_FAILED: 500,
            ErrorCodes.PROCESSING_ML_MATCHING_FAILED: 500,
            ErrorCodes.PROCESSING_JOB_FETCH_FAILED: 500,
            ErrorCodes.PROCESSING_RESULT_SERIALIZATION_FAILED: 500,
            ErrorCodes.PROCESSING_TASK_QUEUE_FAILED: 500,
            ErrorCodes.PROCESSING_TIMEOUT: 504,
            ErrorCodes.PROCESSING_INSUFFICIENT_DATA: 400,
            ErrorCodes.PROCESSING_MODEL_LOAD_FAILED: 500,
            ErrorCodes.PROCESSING_EMBEDDING_FAILED: 500,
            
            # Service errors (503)
            ErrorCodes.SERVICE_RABBITMQ_UNAVAILABLE: 503,
            ErrorCodes.SERVICE_REDIS_UNAVAILABLE: 503,
            ErrorCodes.SERVICE_DATABASE_UNAVAILABLE: 503,
            ErrorCodes.SERVICE_PROFILE_SERVICE_UNAVAILABLE: 503,
            ErrorCodes.SERVICE_JOB_SERVICE_UNAVAILABLE: 503,
            ErrorCodes.SERVICE_ML_SERVICE_UNAVAILABLE: 503,
            ErrorCodes.SERVICE_TIMEOUT: 504,
            ErrorCodes.SERVICE_RATE_LIMIT_EXCEEDED: 429,
            ErrorCodes.SERVICE_CIRCUIT_BREAKER_OPEN: 503,
            
            # Business logic errors (400)
            ErrorCodes.BUSINESS_INVALID_JOB_STATE: 400,
            ErrorCodes.BUSINESS_MATCH_ALREADY_EXISTS: 409,
            ErrorCodes.BUSINESS_INSUFFICIENT_SKILLS: 400,
            ErrorCodes.BUSINESS_NO_MATCHING_JOBS: 404,
            ErrorCodes.BUSINESS_QUOTA_EXCEEDED: 429,
            ErrorCodes.BUSINESS_OPERATION_NOT_ALLOWED: 403,
            ErrorCodes.BUSINESS_DATA_INCONSISTENCY: 500,
        }
        
        return error_code_mappings.get(error_code, 500)
    
    @staticmethod
    def log_error(
        error: Exception,
        correlation_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        Log error with appropriate level and context.
        
        Args:
            error: Exception to log
            correlation_id: Request correlation ID
            additional_context: Additional context information
            logger_instance: Logger instance to use
        """
        log = logger_instance or logger
        
        error_info = ErrorHandler.classify_error(error)
        
        # Prepare log context
        log_context = {
            'error_type': error_info['type'],
            'error_code': error_info['error_code'],
            'correlation_id': correlation_id,
            'details': error_info['details']
        }
        
        if additional_context:
            log_context.update(additional_context)
        
        # Log with appropriate level
        log_level = error_info.get('log_level', 'error')
        log_message = f"Error occurred: {error_info['message']}"
        
        if log_level == 'error':
            log.error(log_message, extra=log_context, exc_info=error)
        elif log_level == 'warning':
            log.warning(log_message, extra=log_context)
        else:
            log.info(log_message, extra=log_context)


def error_handler(
    default_error_code: str = ErrorCodes.SYSTEM_INTERNAL_ERROR,
    correlation_id_func: Optional[callable] = None,
    reraise: bool = False
):
    """
    Decorator for automatic error handling and logging.
    
    Args:
        default_error_code: Default error code for unhandled errors
        correlation_id_func: Function to extract correlation ID
        reraise: Whether to reraise the error after handling
        
    Example:
        @error_handler(ErrorCodes.PROCESSING_RESUME_PARSE_FAILED)
        def parse_resume(file_path):
            # Function implementation
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            correlation_id = None
            
            try:
                # Extract correlation ID if function provided
                if correlation_id_func:
                    correlation_id = correlation_id_func(*args, **kwargs)
                
                return func(*args, **kwargs)
                
            except BaseApplicationError:
                # Re-raise application errors as-is
                raise
                
            except Exception as e:
                # Log the error
                ErrorHandler.log_error(
                    error=e,
                    correlation_id=correlation_id,
                    additional_context={
                        'function_name': func.__name__,
                        'args': str(args)[:200],  # Truncate for security
                        'kwargs': str(kwargs)[:200]
                    }
                )
                
                if reraise:
                    raise
                
                # Convert to application error
                raise ProcessingError(
                    message=f"Error in {func.__name__}: {str(e)}",
                    error_code=default_error_code,
                    correlation_id=correlation_id,
                    original_error=e
                )
        
        return wrapper
    return decorator


@contextmanager
def error_context(
    operation_name: str,
    correlation_id: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None
):
    """
    Context manager for error handling within a code block.
    
    Args:
        operation_name: Name of the operation being performed
        correlation_id: Request correlation ID
        additional_context: Additional context information
        
    Example:
        with error_context("resume_parsing", correlation_id="123"):
            # Code that might raise errors
            parse_resume(file_path)
    """
    try:
        yield
        
    except BaseApplicationError:
        # Re-raise application errors as-is
        raise
        
    except Exception as e:
        # Log and convert to application error
        context = additional_context or {}
        context['operation_name'] = operation_name
        
        ErrorHandler.log_error(
            error=e,
            correlation_id=correlation_id,
            additional_context=context
        )
        
        raise ProcessingError(
            message=f"Error in {operation_name}: {str(e)}",
            processing_stage=operation_name,
            correlation_id=correlation_id,
            original_error=e,
            details=context
        )


# Convenience functions for common error scenarios
def handle_validation_error(
    message: str,
    field_name: Optional[str] = None,
    correlation_id: Optional[str] = None
) -> ValidationError:
    """Create and return a validation error."""
    return ValidationError(
        message=message,
        field_name=field_name,
        correlation_id=correlation_id
    )


def handle_not_found_error(
    resource_type: str,
    resource_id: Optional[str] = None,
    correlation_id: Optional[str] = None
) -> ResourceNotFoundError:
    """Create and return a resource not found error."""
    return ResourceNotFoundError(
        resource_type=resource_type,
        resource_id=resource_id,
        correlation_id=correlation_id
    )


def handle_service_error(
    service_name: str,
    message: Optional[str] = None,
    correlation_id: Optional[str] = None
) -> ServiceError:
    """Create and return a service error."""
    error_message = message or f"{service_name} service is unavailable"
    return ServiceError(
        message=error_message,
        service_name=service_name,
        correlation_id=correlation_id
    )