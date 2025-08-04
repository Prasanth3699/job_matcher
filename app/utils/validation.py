"""
Unified validation utilities for the resume job matcher service.

This module provides centralized validation functionality to eliminate
code duplication and ensure consistent validation across the application.
"""

import re
import json
import mimetypes
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging

from app.core.constants import (
    ValidationRules,
    FileConstraints,
    InputLimits,
    ErrorCodes,
    ErrorMessages,
    BusinessValidationRules,
    ValidationHelpers,
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""

    def __init__(
        self,
        message: str,
        error_code: str = ErrorCodes.VALIDATION_INVALID_FORMAT,
        field_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.field_name = field_name
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation error to dictionary format."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "field": self.field_name,
            "details": self.details,
        }


class ValidationResult:
    """Represents the result of a validation operation."""

    def __init__(
        self, is_valid: bool = True, errors: Optional[List[ValidationError]] = None
    ):
        self.is_valid = is_valid
        self.errors = errors or []

    def add_error(self, error: ValidationError):
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False

    def add_errors(self, errors: List[ValidationError]):
        """Add multiple validation errors."""
        self.errors.extend(errors)
        if errors:
            self.is_valid = False

    def get_error_messages(self) -> List[str]:
        """Get all error messages."""
        return [error.message for error in self.errors]

    def get_first_error(self) -> Optional[ValidationError]:
        """Get the first validation error."""
        return self.errors[0] if self.errors else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary format."""
        return {
            "is_valid": self.is_valid,
            "errors": [error.to_dict() for error in self.errors],
        }


class DataValidator:
    """Core data validation utility class."""

    @staticmethod
    def validate_required_fields(
        data: Dict[str, Any], required_fields: List[str]
    ) -> ValidationResult:
        """
        Validate that all required fields are present and not empty.

        Args:
            data: Dictionary to validate
            required_fields: List of required field names

        Returns:
            ValidationResult indicating success or failure
        """
        result = ValidationResult()

        for field in required_fields:
            if field not in data:
                result.add_error(
                    ValidationError(
                        message=ErrorMessages.get_message(
                            ErrorCodes.VALIDATION_REQUIRED_FIELD_MISSING, field=field
                        ),
                        error_code=ErrorCodes.VALIDATION_REQUIRED_FIELD_MISSING,
                        field_name=field,
                    )
                )
            elif data[field] is None or (
                isinstance(data[field], str) and not data[field].strip()
            ):
                result.add_error(
                    ValidationError(
                        message=f"Field '{field}' cannot be empty",
                        error_code=ErrorCodes.VALIDATION_REQUIRED_FIELD_MISSING,
                        field_name=field,
                    )
                )

        return result

    @staticmethod
    def validate_string_field(
        value: Any,
        field_name: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[re.Pattern] = None,
        allow_empty: bool = False,
    ) -> ValidationResult:
        """
        Validate a string field with various constraints.

        Args:
            value: Value to validate
            field_name: Name of the field being validated
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            pattern: Regex pattern that must match
            allow_empty: Whether empty strings are allowed

        Returns:
            ValidationResult indicating success or failure
        """
        result = ValidationResult()

        # Check if value is string
        if not isinstance(value, str):
            if value is None and allow_empty:
                return result
            result.add_error(
                ValidationError(
                    message=f"Field '{field_name}' must be a string",
                    error_code=ErrorCodes.VALIDATION_INVALID_FORMAT,
                    field_name=field_name,
                )
            )
            return result

        # Check empty string
        if not value.strip() and not allow_empty:
            result.add_error(
                ValidationError(
                    message=f"Field '{field_name}' cannot be empty",
                    error_code=ErrorCodes.VALIDATION_REQUIRED_FIELD_MISSING,
                    field_name=field_name,
                )
            )
            return result

        # Check length constraints
        if min_length is not None and len(value) < min_length:
            result.add_error(
                ValidationError(
                    message=f"Field '{field_name}' must be at least {min_length} characters long",
                    error_code=ErrorCodes.VALIDATION_VALUE_OUT_OF_RANGE,
                    field_name=field_name,
                    details={"min_length": min_length, "actual_length": len(value)},
                )
            )

        if max_length is not None and len(value) > max_length:
            result.add_error(
                ValidationError(
                    message=f"Field '{field_name}' must be at most {max_length} characters long",
                    error_code=ErrorCodes.VALIDATION_VALUE_OUT_OF_RANGE,
                    field_name=field_name,
                    details={"max_length": max_length, "actual_length": len(value)},
                )
            )

        # Check pattern match
        if pattern and not pattern.match(value):
            result.add_error(
                ValidationError(
                    message=f"Field '{field_name}' has invalid format",
                    error_code=ErrorCodes.VALIDATION_INVALID_FORMAT,
                    field_name=field_name,
                )
            )

        # Check for security issues
        if not ValidationHelpers.is_safe_content(value):
            result.add_error(
                ValidationError(
                    message=f"Field '{field_name}' contains potentially unsafe content",
                    error_code=ErrorCodes.VALIDATION_INVALID_FORMAT,
                    field_name=field_name,
                )
            )

        return result

    @staticmethod
    def validate_numeric_field(
        value: Any,
        field_name: str,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        data_type: type = float,
    ) -> ValidationResult:
        """
        Validate a numeric field with range constraints.

        Args:
            value: Value to validate
            field_name: Name of the field being validated
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            data_type: Expected numeric type (int or float)

        Returns:
            ValidationResult indicating success or failure
        """
        result = ValidationResult()

        # Type conversion if needed
        if isinstance(value, str):
            try:
                value = data_type(value)
            except (ValueError, TypeError):
                result.add_error(
                    ValidationError(
                        message=f"Field '{field_name}' must be a valid {data_type.__name__}",
                        error_code=ErrorCodes.VALIDATION_INVALID_FORMAT,
                        field_name=field_name,
                    )
                )
                return result

        # Check type
        if not isinstance(value, (int, float)):
            result.add_error(
                ValidationError(
                    message=f"Field '{field_name}' must be numeric",
                    error_code=ErrorCodes.VALIDATION_INVALID_FORMAT,
                    field_name=field_name,
                )
            )
            return result

        # Check range constraints
        if min_value is not None and value < min_value:
            result.add_error(
                ValidationError(
                    message=f"Field '{field_name}' must be at least {min_value}",
                    error_code=ErrorCodes.VALIDATION_VALUE_OUT_OF_RANGE,
                    field_name=field_name,
                    details={"min_value": min_value, "actual_value": value},
                )
            )

        if max_value is not None and value > max_value:
            result.add_error(
                ValidationError(
                    message=f"Field '{field_name}' must be at most {max_value}",
                    error_code=ErrorCodes.VALIDATION_VALUE_OUT_OF_RANGE,
                    field_name=field_name,
                    details={"max_value": max_value, "actual_value": value},
                )
            )

        return result

    @staticmethod
    def validate_array_field(
        value: Any,
        field_name: str,
        min_items: Optional[int] = None,
        max_items: Optional[int] = None,
        item_validator: Optional[callable] = None,
    ) -> ValidationResult:
        """
        Validate an array field with item constraints.

        Args:
            value: Value to validate
            field_name: Name of the field being validated
            min_items: Minimum number of items allowed
            max_items: Maximum number of items allowed
            item_validator: Function to validate individual items

        Returns:
            ValidationResult indicating success or failure
        """
        result = ValidationResult()

        # Check if value is list-like
        if not isinstance(value, (list, tuple)):
            result.add_error(
                ValidationError(
                    message=f"Field '{field_name}' must be an array",
                    error_code=ErrorCodes.VALIDATION_INVALID_FORMAT,
                    field_name=field_name,
                )
            )
            return result

        # Check length constraints
        if min_items is not None and len(value) < min_items:
            result.add_error(
                ValidationError(
                    message=f"Field '{field_name}' must contain at least {min_items} items",
                    error_code=ErrorCodes.VALIDATION_VALUE_OUT_OF_RANGE,
                    field_name=field_name,
                    details={"min_items": min_items, "actual_items": len(value)},
                )
            )

        if max_items is not None and len(value) > max_items:
            result.add_error(
                ValidationError(
                    message=f"Field '{field_name}' must contain at most {max_items} items",
                    error_code=ErrorCodes.VALIDATION_VALUE_OUT_OF_RANGE,
                    field_name=field_name,
                    details={"max_items": max_items, "actual_items": len(value)},
                )
            )

        # Validate individual items
        if item_validator:
            for i, item in enumerate(value):
                try:
                    item_result = item_validator(item, f"{field_name}[{i}]")
                    if hasattr(item_result, "errors"):
                        result.add_errors(item_result.errors)
                except Exception as e:
                    result.add_error(
                        ValidationError(
                            message=f"Invalid item at index {i} in field '{field_name}': {str(e)}",
                            error_code=ErrorCodes.VALIDATION_INVALID_FORMAT,
                            field_name=f"{field_name}[{i}]",
                        )
                    )

        return result

    @staticmethod
    def validate_json_field(value: Any, field_name: str) -> ValidationResult:
        """
        Validate that a field contains valid JSON.

        Args:
            value: Value to validate
            field_name: Name of the field being validated

        Returns:
            ValidationResult indicating success or failure
        """
        result = ValidationResult()

        if isinstance(value, str):
            try:
                json.loads(value)
            except json.JSONDecodeError as e:
                result.add_error(
                    ValidationError(
                        message=f"Field '{field_name}' contains invalid JSON: {e.msg}",
                        error_code=ErrorCodes.VALIDATION_INVALID_JSON,
                        field_name=field_name,
                        details={"json_error": e.msg},
                    )
                )
        elif not isinstance(value, (dict, list)):
            result.add_error(
                ValidationError(
                    message=f"Field '{field_name}' must be valid JSON or a dictionary/list",
                    error_code=ErrorCodes.VALIDATION_INVALID_JSON,
                    field_name=field_name,
                )
            )

        return result


class FileValidator:
    """File upload and content validation utility class."""

    @staticmethod
    def validate_file_upload(
        filename: str,
        file_size: int,
        content_type: Optional[str] = None,
        file_content: Optional[bytes] = None,
    ) -> ValidationResult:
        """
        Comprehensive file upload validation.

        Args:
            filename: Name of the uploaded file
            file_size: Size of the file in bytes
            content_type: MIME type of the file
            file_content: File content for additional validation

        Returns:
            ValidationResult indicating success or failure
        """
        result = ValidationResult()

        # Validate filename
        if not filename or not filename.strip():
            result.add_error(
                ValidationError(
                    message="Filename is required",
                    error_code=ErrorCodes.VALIDATION_REQUIRED_FIELD_MISSING,
                    field_name="filename",
                )
            )
            return result

        # Validate file extension
        if not FileConstraints.is_valid_extension(filename):
            allowed_extensions = ", ".join(FileConstraints.ALLOWED_RESUME_EXTENSIONS)
            result.add_error(
                ValidationError(
                    message=f"File type not supported. Allowed types: {allowed_extensions}",
                    error_code=ErrorCodes.VALIDATION_INVALID_FILE_TYPE,
                    field_name="filename",
                    details={
                        "allowed_types": FileConstraints.ALLOWED_RESUME_EXTENSIONS
                    },
                )
            )

        # Validate file size
        if not FileConstraints.is_valid_size(file_size):
            if file_size < FileConstraints.MIN_FILE_SIZE:
                result.add_error(
                    ValidationError(
                        message=f"File size is too small. Minimum size is {FileConstraints.MIN_FILE_SIZE} bytes",
                        error_code=ErrorCodes.VALIDATION_FILE_TOO_SMALL,
                        field_name="file_size",
                        details={
                            "min_size": FileConstraints.MIN_FILE_SIZE,
                            "actual_size": file_size,
                        },
                    )
                )
            else:
                max_size_mb = FileConstraints.MAX_FILE_SIZE // (1024 * 1024)
                result.add_error(
                    ValidationError(
                        message=f"File size exceeds maximum limit of {max_size_mb}MB",
                        error_code=ErrorCodes.VALIDATION_FILE_TOO_LARGE,
                        field_name="file_size",
                        details={
                            "max_size": FileConstraints.MAX_FILE_SIZE,
                            "actual_size": file_size,
                        },
                    )
                )

        # Validate content type
        if content_type and not FileConstraints.is_valid_mime_type(content_type):
            result.add_error(
                ValidationError(
                    message=f"MIME type '{content_type}' is not supported",
                    error_code=ErrorCodes.VALIDATION_INVALID_FILE_TYPE,
                    field_name="content_type",
                    details={"allowed_types": FileConstraints.ALLOWED_MIME_TYPES},
                )
            )

        # Validate file content if provided
        if file_content is not None:
            content_result = FileValidator._validate_file_content(
                file_content, filename
            )
            result.add_errors(content_result.errors)

        return result

    @staticmethod
    def _validate_file_content(file_content: bytes, filename: str) -> ValidationResult:
        """
        Validate file content for safety and integrity.

        Args:
            file_content: File content as bytes
            filename: Name of the file

        Returns:
            ValidationResult indicating success or failure
        """
        result = ValidationResult()

        # Check if file is empty
        if not file_content:
            result.add_error(
                ValidationError(
                    message="File content is empty",
                    error_code=ErrorCodes.VALIDATION_FILE_CORRUPTED,
                    field_name="file_content",
                )
            )
            return result

        # Basic file signature validation
        file_extension = Path(filename).suffix.lower()

        if file_extension == ".pdf":
            # PDF files should start with %PDF
            if not file_content.startswith(b"%PDF"):
                result.add_error(
                    ValidationError(
                        message="PDF file appears to be corrupted or invalid",
                        error_code=ErrorCodes.VALIDATION_FILE_CORRUPTED,
                        field_name="file_content",
                    )
                )

        elif file_extension in [".docx", ".doc"]:
            # Office documents have specific signatures
            if file_extension == ".docx":
                # DOCX files are ZIP archives
                if not file_content.startswith(b"PK"):
                    result.add_error(
                        ValidationError(
                            message="DOCX file appears to be corrupted or invalid",
                            error_code=ErrorCodes.VALIDATION_FILE_CORRUPTED,
                            field_name="file_content",
                        )
                    )
            elif file_extension == ".doc":
                # DOC files start with specific signature
                if not (
                    file_content.startswith(b"\xd0\xcf\x11\xe0")
                    or file_content.startswith(b"\x0d\x44\x4f\x43")
                ):
                    result.add_error(
                        ValidationError(
                            message="DOC file appears to be corrupted or invalid",
                            error_code=ErrorCodes.VALIDATION_FILE_CORRUPTED,
                            field_name="file_content",
                        )
                    )

        return result


class BusinessLogicValidator:
    """Business logic validation utility class."""

    @staticmethod
    def validate_job_ids(job_ids: List[int]) -> ValidationResult:
        """
        Validate job IDs according to business rules.

        Args:
            job_ids: List of job IDs to validate

        Returns:
            ValidationResult indicating success or failure
        """
        result = ValidationResult()

        # Check if list is provided
        if not isinstance(job_ids, list):
            result.add_error(
                ValidationError(
                    message="Job IDs must be provided as a list",
                    error_code=ErrorCodes.VALIDATION_INVALID_JOB_IDS,
                    field_name="job_ids",
                )
            )
            return result

        # Check minimum and maximum count
        if len(job_ids) < BusinessValidationRules.MATCHING_RULES["min_resume_skills"]:
            result.add_error(
                ValidationError(
                    message=f"At least {BusinessValidationRules.MATCHING_RULES['min_resume_skills']} job IDs required",
                    error_code=ErrorCodes.VALIDATION_INVALID_JOB_IDS,
                    field_name="job_ids",
                )
            )

        if len(job_ids) > InputLimits.ARRAY_LIMITS["job_ids"]["max_items"]:
            max_items = InputLimits.ARRAY_LIMITS["job_ids"]["max_items"]
            result.add_error(
                ValidationError(
                    message=f"Maximum {max_items} job IDs allowed per request",
                    error_code=ErrorCodes.VALIDATION_INVALID_JOB_IDS,
                    field_name="job_ids",
                    details={"max_items": max_items, "actual_items": len(job_ids)},
                )
            )

        # Validate individual job IDs
        valid_ids = set()
        for i, job_id in enumerate(job_ids):
            if not isinstance(job_id, int) or job_id <= 0:
                result.add_error(
                    ValidationError(
                        message=f"Job ID at index {i} must be a positive integer",
                        error_code=ErrorCodes.VALIDATION_INVALID_JOB_IDS,
                        field_name=f"job_ids[{i}]",
                    )
                )
            elif job_id in valid_ids:
                result.add_error(
                    ValidationError(
                        message=f"Duplicate job ID found: {job_id}",
                        error_code=ErrorCodes.VALIDATION_INVALID_JOB_IDS,
                        field_name=f"job_ids[{i}]",
                    )
                )
            else:
                valid_ids.add(job_id)

        return result

    @staticmethod
    def validate_user_preferences(preferences: Dict[str, Any]) -> ValidationResult:
        """
        Validate user preferences according to business rules.

        Args:
            preferences: User preferences dictionary

        Returns:
            ValidationResult indicating success or failure
        """
        result = ValidationResult()

        # Validate preferred job types
        if "preferred_job_types" in preferences:
            job_types = preferences["preferred_job_types"]
            if isinstance(job_types, list):
                valid_types = BusinessValidationRules.PREFERENCE_RULES[
                    "valid_job_types"
                ]
                for job_type in job_types:
                    if job_type.lower() not in valid_types:
                        result.add_error(
                            ValidationError(
                                message=f"Invalid job type: {job_type}. Valid types: {valid_types}",
                                error_code=ErrorCodes.VALIDATION_INVALID_PREFERENCES,
                                field_name="preferred_job_types",
                            )
                        )

        # Validate preferred locations
        if "preferred_locations" in preferences:
            locations = preferences["preferred_locations"]
            if isinstance(locations, list):
                max_locations = BusinessValidationRules.PREFERENCE_RULES[
                    "max_preferred_locations"
                ]
                if len(locations) > max_locations:
                    result.add_error(
                        ValidationError(
                            message=f"Maximum {max_locations} preferred locations allowed",
                            error_code=ErrorCodes.VALIDATION_INVALID_PREFERENCES,
                            field_name="preferred_locations",
                        )
                    )

        # Validate salary expectation
        if "salary_expectation" in preferences:
            salary_range = preferences["salary_expectation"]
            if (
                isinstance(salary_range, dict)
                and "min" in salary_range
                and "max" in salary_range
            ):
                min_salary = salary_range.get("min", 0)
                max_salary = salary_range.get("max", 0)
                if max_salary > 0 and min_salary > 0:
                    ratio = max_salary / min_salary
                    max_ratio = BusinessValidationRules.PREFERENCE_RULES[
                        "max_salary_range_ratio"
                    ]
                    if ratio > max_ratio:
                        result.add_error(
                            ValidationError(
                                message=f"Salary range ratio too high. Maximum ratio: {max_ratio}",
                                error_code=ErrorCodes.VALIDATION_INVALID_PREFERENCES,
                                field_name="salary_expectation",
                            )
                        )

        return result


# Convenience functions for common validation scenarios
def validate_email(email: str, field_name: str = "email") -> ValidationResult:
    """Validate email address format."""
    result = ValidationResult()
    if not ValidationHelpers.is_valid_email(email):
        result.add_error(
            ValidationError(
                message=f"Invalid email address format: {email}",
                error_code=ErrorCodes.VALIDATION_INVALID_FORMAT,
                field_name=field_name,
            )
        )
    return result


def validate_url(url: str, field_name: str = "url") -> ValidationResult:
    """Validate URL format."""
    result = ValidationResult()
    if not ValidationHelpers.is_valid_url(url):
        result.add_error(
            ValidationError(
                message=f"Invalid URL format: {url}",
                error_code=ErrorCodes.VALIDATION_INVALID_FORMAT,
                field_name=field_name,
            )
        )
    return result


def validate_phone(phone: str, field_name: str = "phone") -> ValidationResult:
    """Validate phone number format."""
    result = ValidationResult()
    if not ValidationHelpers.is_valid_phone(phone):
        result.add_error(
            ValidationError(
                message=f"Invalid phone number format: {phone}",
                error_code=ErrorCodes.VALIDATION_INVALID_FORMAT,
                field_name=field_name,
            )
        )
    return result
