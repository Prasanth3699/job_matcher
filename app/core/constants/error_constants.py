"""
Error codes and standardized error messages for the resume job matcher service.
"""

from typing import Dict, Any


class ErrorCodes:
    """Standardized error codes following conventional patterns."""
    
    # Authentication and Authorization (1000-1099)
    AUTH_TOKEN_INVALID = "AUTH_1001"
    AUTH_TOKEN_EXPIRED = "AUTH_1002"
    AUTH_TOKEN_MISSING = "AUTH_1003"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_1004"
    AUTH_USER_NOT_FOUND = "AUTH_1005"
    AUTH_SERVICE_UNAVAILABLE = "AUTH_1006"
    
    # Input Validation (1100-1199)
    VALIDATION_REQUIRED_FIELD_MISSING = "VAL_1101"
    VALIDATION_INVALID_FORMAT = "VAL_1102"
    VALIDATION_VALUE_OUT_OF_RANGE = "VAL_1103"
    VALIDATION_INVALID_FILE_TYPE = "VAL_1104"
    VALIDATION_FILE_TOO_LARGE = "VAL_1105"
    VALIDATION_FILE_TOO_SMALL = "VAL_1106"
    VALIDATION_INVALID_JSON = "VAL_1107"
    VALIDATION_INVALID_JOB_IDS = "VAL_1108"
    VALIDATION_INVALID_PREFERENCES = "VAL_1109"
    VALIDATION_FILE_CORRUPTED = "VAL_1110"
    
    # Resource Not Found (1200-1299)
    RESOURCE_MATCH_JOB_NOT_FOUND = "RES_1201"
    RESOURCE_USER_NOT_FOUND = "RES_1202"
    RESOURCE_JOB_NOT_FOUND = "RES_1203"
    RESOURCE_RESUME_NOT_FOUND = "RES_1204"
    RESOURCE_MODEL_NOT_FOUND = "RES_1205"
    RESOURCE_ENDPOINT_NOT_FOUND = "RES_1206"
    
    # Processing Errors (1300-1399)
    PROCESSING_RESUME_PARSE_FAILED = "PROC_1301"
    PROCESSING_FEATURE_EXTRACTION_FAILED = "PROC_1302"
    PROCESSING_ML_MATCHING_FAILED = "PROC_1303"
    PROCESSING_JOB_FETCH_FAILED = "PROC_1304"
    PROCESSING_RESULT_SERIALIZATION_FAILED = "PROC_1305"
    PROCESSING_TASK_QUEUE_FAILED = "PROC_1306"
    PROCESSING_TIMEOUT = "PROC_1307"
    PROCESSING_INSUFFICIENT_DATA = "PROC_1308"
    PROCESSING_MODEL_LOAD_FAILED = "PROC_1309"
    PROCESSING_EMBEDDING_FAILED = "PROC_1310"
    
    # External Service Errors (1400-1499)
    SERVICE_RABBITMQ_UNAVAILABLE = "SVC_1401"
    SERVICE_REDIS_UNAVAILABLE = "SVC_1402"
    SERVICE_DATABASE_UNAVAILABLE = "SVC_1403"
    SERVICE_PROFILE_SERVICE_UNAVAILABLE = "SVC_1404"
    SERVICE_JOB_SERVICE_UNAVAILABLE = "SVC_1405"
    SERVICE_ML_SERVICE_UNAVAILABLE = "SVC_1406"
    SERVICE_TIMEOUT = "SVC_1407"
    SERVICE_RATE_LIMIT_EXCEEDED = "SVC_1408"
    SERVICE_CIRCUIT_BREAKER_OPEN = "SVC_1409"
    
    # System Errors (1500-1599)
    SYSTEM_INTERNAL_ERROR = "SYS_1501"
    SYSTEM_OUT_OF_MEMORY = "SYS_1502"
    SYSTEM_DISK_SPACE_FULL = "SYS_1503"
    SYSTEM_CONFIGURATION_ERROR = "SYS_1504"
    SYSTEM_DEPENDENCY_MISSING = "SYS_1505"
    SYSTEM_RESOURCE_EXHAUSTED = "SYS_1506"
    SYSTEM_MAINTENANCE_MODE = "SYS_1507"
    
    # Business Logic Errors (1600-1699)
    BUSINESS_INVALID_JOB_STATE = "BIZ_1601"
    BUSINESS_MATCH_ALREADY_EXISTS = "BIZ_1602"
    BUSINESS_INSUFFICIENT_SKILLS = "BIZ_1603"
    BUSINESS_NO_MATCHING_JOBS = "BIZ_1604"
    BUSINESS_QUOTA_EXCEEDED = "BIZ_1605"
    BUSINESS_OPERATION_NOT_ALLOWED = "BIZ_1606"
    BUSINESS_DATA_INCONSISTENCY = "BIZ_1607"


class ErrorMessages:
    """Standardized error messages corresponding to error codes."""
    
    # Authentication and Authorization Messages
    AUTH_MESSAGES = {
        ErrorCodes.AUTH_TOKEN_INVALID: "Authentication token is invalid or malformed",
        ErrorCodes.AUTH_TOKEN_EXPIRED: "Authentication token has expired",
        ErrorCodes.AUTH_TOKEN_MISSING: "Authentication token is required but not provided",
        ErrorCodes.AUTH_INSUFFICIENT_PERMISSIONS: "Insufficient permissions to access this resource",
        ErrorCodes.AUTH_USER_NOT_FOUND: "User account not found or inactive",
        ErrorCodes.AUTH_SERVICE_UNAVAILABLE: "Authentication service is temporarily unavailable"
    }
    
    # Input Validation Messages
    VALIDATION_MESSAGES = {
        ErrorCodes.VALIDATION_REQUIRED_FIELD_MISSING: "Required field '{field}' is missing",
        ErrorCodes.VALIDATION_INVALID_FORMAT: "Field '{field}' has invalid format: {details}",
        ErrorCodes.VALIDATION_VALUE_OUT_OF_RANGE: "Value for '{field}' is out of acceptable range",
        ErrorCodes.VALIDATION_INVALID_FILE_TYPE: "File type not supported. Allowed types: {allowed_types}",
        ErrorCodes.VALIDATION_FILE_TOO_LARGE: "File size exceeds maximum limit of {max_size_mb}MB",
        ErrorCodes.VALIDATION_FILE_TOO_SMALL: "File size is too small. Minimum size is {min_size_kb}KB",
        ErrorCodes.VALIDATION_INVALID_JSON: "Invalid JSON format in field '{field}'",
        ErrorCodes.VALIDATION_INVALID_JOB_IDS: "Invalid job IDs provided. Expected comma-separated integers",
        ErrorCodes.VALIDATION_INVALID_PREFERENCES: "Invalid preferences format. Expected JSON object",
        ErrorCodes.VALIDATION_FILE_CORRUPTED: "Uploaded file appears to be corrupted or unreadable"
    }
    
    # Resource Not Found Messages
    RESOURCE_MESSAGES = {
        ErrorCodes.RESOURCE_MATCH_JOB_NOT_FOUND: "Match job with ID '{job_id}' not found or access denied",
        ErrorCodes.RESOURCE_USER_NOT_FOUND: "User with ID '{user_id}' not found",
        ErrorCodes.RESOURCE_JOB_NOT_FOUND: "Job with ID '{job_id}' not found",
        ErrorCodes.RESOURCE_RESUME_NOT_FOUND: "Resume with ID '{resume_id}' not found",
        ErrorCodes.RESOURCE_MODEL_NOT_FOUND: "ML model '{model_name}' not found or unavailable",
        ErrorCodes.RESOURCE_ENDPOINT_NOT_FOUND: "API endpoint '{endpoint}' not found"
    }
    
    # Processing Error Messages
    PROCESSING_MESSAGES = {
        ErrorCodes.PROCESSING_RESUME_PARSE_FAILED: "Failed to parse resume file: {details}",
        ErrorCodes.PROCESSING_FEATURE_EXTRACTION_FAILED: "Failed to extract features from resume",
        ErrorCodes.PROCESSING_ML_MATCHING_FAILED: "ML matching algorithm failed: {details}",
        ErrorCodes.PROCESSING_JOB_FETCH_FAILED: "Failed to fetch job details: {details}",
        ErrorCodes.PROCESSING_RESULT_SERIALIZATION_FAILED: "Failed to serialize match results",
        ErrorCodes.PROCESSING_TASK_QUEUE_FAILED: "Failed to queue background task",
        ErrorCodes.PROCESSING_TIMEOUT: "Processing timeout after {timeout_seconds} seconds",
        ErrorCodes.PROCESSING_INSUFFICIENT_DATA: "Insufficient data for reliable matching",
        ErrorCodes.PROCESSING_MODEL_LOAD_FAILED: "Failed to load ML model: {model_name}",
        ErrorCodes.PROCESSING_EMBEDDING_FAILED: "Failed to generate text embeddings"
    }
    
    # External Service Error Messages
    SERVICE_MESSAGES = {
        ErrorCodes.SERVICE_RABBITMQ_UNAVAILABLE: "RabbitMQ message broker is unavailable",
        ErrorCodes.SERVICE_REDIS_UNAVAILABLE: "Redis cache service is unavailable",
        ErrorCodes.SERVICE_DATABASE_UNAVAILABLE: "Database service is unavailable",
        ErrorCodes.SERVICE_PROFILE_SERVICE_UNAVAILABLE: "Profile service is temporarily unavailable",
        ErrorCodes.SERVICE_JOB_SERVICE_UNAVAILABLE: "Job service is temporarily unavailable",
        ErrorCodes.SERVICE_ML_SERVICE_UNAVAILABLE: "ML service is temporarily unavailable",
        ErrorCodes.SERVICE_TIMEOUT: "Service request timeout after {timeout_seconds} seconds",
        ErrorCodes.SERVICE_RATE_LIMIT_EXCEEDED: "Rate limit exceeded. Please try again later",
        ErrorCodes.SERVICE_CIRCUIT_BREAKER_OPEN: "Service temporarily unavailable due to high failure rate"
    }
    
    # System Error Messages
    SYSTEM_MESSAGES = {
        ErrorCodes.SYSTEM_INTERNAL_ERROR: "An internal system error occurred",
        ErrorCodes.SYSTEM_OUT_OF_MEMORY: "System is out of memory",
        ErrorCodes.SYSTEM_DISK_SPACE_FULL: "System disk space is full",
        ErrorCodes.SYSTEM_CONFIGURATION_ERROR: "System configuration error: {details}",
        ErrorCodes.SYSTEM_DEPENDENCY_MISSING: "Required system dependency is missing: {dependency}",
        ErrorCodes.SYSTEM_RESOURCE_EXHAUSTED: "System resources are exhausted",
        ErrorCodes.SYSTEM_MAINTENANCE_MODE: "System is currently in maintenance mode"
    }
    
    # Business Logic Error Messages
    BUSINESS_MESSAGES = {
        ErrorCodes.BUSINESS_INVALID_JOB_STATE: "Match job is in invalid state for this operation",
        ErrorCodes.BUSINESS_MATCH_ALREADY_EXISTS: "A match for this resume and job set already exists",
        ErrorCodes.BUSINESS_INSUFFICIENT_SKILLS: "Resume contains insufficient skills for reliable matching",
        ErrorCodes.BUSINESS_NO_MATCHING_JOBS: "No jobs found matching the specified criteria",
        ErrorCodes.BUSINESS_QUOTA_EXCEEDED: "User quota exceeded for {resource_type}",
        ErrorCodes.BUSINESS_OPERATION_NOT_ALLOWED: "Operation not allowed in current context",
        ErrorCodes.BUSINESS_DATA_INCONSISTENCY: "Data inconsistency detected: {details}"
    }
    
    # Default fallback message
    DEFAULT_ERROR_MESSAGE = "An unexpected error occurred"
    
    @classmethod
    def get_message(cls, error_code: str, **kwargs) -> str:
        """Get formatted error message for given error code."""
        # Find the message in the appropriate category
        all_messages = {
            **cls.AUTH_MESSAGES,
            **cls.VALIDATION_MESSAGES,
            **cls.RESOURCE_MESSAGES,
            **cls.PROCESSING_MESSAGES,
            **cls.SERVICE_MESSAGES,
            **cls.SYSTEM_MESSAGES,
            **cls.BUSINESS_MESSAGES
        }
        
        message_template = all_messages.get(error_code, cls.DEFAULT_ERROR_MESSAGE)
        
        try:
            return message_template.format(**kwargs)
        except KeyError:
            # If formatting fails due to missing keys, return template as-is
            return message_template


class ValidationMessages:
    """Specific validation error messages for common validation scenarios."""
    
    # File validation messages
    FILE_VALIDATION = {
        "empty_file": "Uploaded file is empty",
        "invalid_pdf": "PDF file is corrupted or password protected",
        "invalid_docx": "DOCX file is corrupted or in unsupported format",
        "no_text_content": "No readable text content found in the file",
        "insufficient_content": "File contains insufficient content for analysis"
    }
    
    # Data validation messages
    DATA_VALIDATION = {
        "invalid_email": "Email address format is invalid",
        "invalid_phone": "Phone number format is invalid",
        "invalid_url": "URL format is invalid",
        "invalid_date": "Date format is invalid or out of range",
        "invalid_number": "Number format is invalid or out of range"
    }
    
    # Job matching validation messages
    MATCHING_VALIDATION = {
        "no_job_ids": "At least one job ID must be provided",
        "too_many_jobs": "Maximum {max_jobs} job IDs allowed per request",
        "invalid_job_id": "Job ID must be a positive integer",
        "duplicate_job_ids": "Duplicate job IDs found in request",
        "job_id_not_exists": "Job ID {job_id} does not exist"
    }
    
    # Resume validation messages
    RESUME_VALIDATION = {
        "no_skills_found": "No recognizable skills found in resume",
        "no_experience_found": "No work experience information found",
        "no_education_found": "No education information found",
        "insufficient_detail": "Resume lacks sufficient detail for accurate matching"
    }
    
    # Preference validation messages
    PREFERENCE_VALIDATION = {
        "invalid_location": "Location preference format is invalid",
        "invalid_salary": "Salary expectation format is invalid",
        "invalid_job_type": "Job type preference is not supported",
        "invalid_company": "Company preference format is invalid"
    }


class UserFriendlyMessages:
    """User-friendly error messages for end-user display."""
    
    # General user-friendly messages
    FRIENDLY_MESSAGES = {
        "file_upload_failed": "We couldn't process your resume file. Please check the file format and try again.",
        "matching_failed": "We encountered an issue while matching your resume. Please try again in a few minutes.",
        "service_unavailable": "Our service is temporarily unavailable. Please try again later.",
        "invalid_request": "Your request contains invalid information. Please check your input and try again.",
        "session_expired": "Your session has expired. Please log in again.",
        "quota_exceeded": "You've reached your usage limit. Please try again later or upgrade your plan.",
        "no_matches_found": "We couldn't find any matching jobs for your profile. Try expanding your search criteria.",
        "processing_delay": "Your request is taking longer than usual. We'll notify you when it's complete."
    }
    
    # Action-oriented suggestions
    SUGGESTIONS = {
        "retry_action": "Please try again",
        "contact_support": "If this problem persists, please contact our support team",
        "check_file_format": "Please ensure your file is in PDF or DOCX format",
        "reduce_file_size": "Please reduce your file size to under {max_size}MB",
        "improve_resume": "Consider adding more skills or experience details to your resume",
        "expand_criteria": "Try expanding your job search criteria for better matches"
    }