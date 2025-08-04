"""
Unified utilities package for the resume job matcher service.

This package provides centralized utilities to eliminate code duplication
and ensure consistent functionality across the application.
"""

# Import existing utilities
from .logger import logger

# from .file_helpers import cleanup_temp_file as legacy_cleanup_temp_file
from .security import *

# Import new unified utilities
from .serialization import (
    JSONSerializer,
    SerializationError,
    BatchSerializer,
    ValidationSerializer,
    make_json_serializable,
    serialize_to_json,
    deserialize_from_json,
)

from .validation import (
    ValidationError,
    ValidationResult,
    DataValidator,
    FileValidator,
    BusinessLogicValidator,
    validate_email,
    validate_url,
    validate_phone,
)

from .responses import (
    APIResponse,
    MatchJobResponseBuilder,
    ResponseHelper,
    success_response,
    error_response,
    validation_error_response,
    not_found_response,
)

from .file_processing import (
    FileProcessingError,
    SecureFileHandler,
    FileMetadataExtractor,
    FileContentAnalyzer,
    BulkFileProcessor,
    save_upload_securely,
    cleanup_temp_file,
    get_file_metadata,
    analyze_text_content,
)

from .error_handling import (
    BaseApplicationError,
    ValidationError as ValidationException,
    ProcessingError,
    ServiceError,
    AuthenticationError,
    AuthorizationError,
    ResourceNotFoundError,
    ErrorHandler,
    error_handler,
    error_context,
    handle_validation_error,
    handle_not_found_error,
    handle_service_error,
)

# Public API
__all__ = [
    # Legacy utilities
    "logger",
    # 'legacy_cleanup_temp_file',
    # Serialization utilities
    "JSONSerializer",
    "SerializationError",
    "BatchSerializer",
    "ValidationSerializer",
    "make_json_serializable",
    "serialize_to_json",
    "deserialize_from_json",
    # Validation utilities
    "ValidationError",
    "ValidationResult",
    "DataValidator",
    "FileValidator",
    "BusinessLogicValidator",
    "validate_email",
    "validate_url",
    "validate_phone",
    # Response utilities
    "APIResponse",
    "MatchJobResponseBuilder",
    "ResponseHelper",
    "success_response",
    "error_response",
    "validation_error_response",
    "not_found_response",
    # File processing utilities
    "FileProcessingError",
    "SecureFileHandler",
    "FileMetadataExtractor",
    "FileContentAnalyzer",
    "BulkFileProcessor",
    "save_upload_securely",
    "cleanup_temp_file",
    "get_file_metadata",
    "analyze_text_content",
    # Error handling utilities
    "BaseApplicationError",
    "ValidationException",
    "ProcessingError",
    "ServiceError",
    "AuthenticationError",
    "AuthorizationError",
    "ResourceNotFoundError",
    "ErrorHandler",
    "error_handler",
    "error_context",
    "handle_validation_error",
    "handle_not_found_error",
    "handle_service_error",
]
