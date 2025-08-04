"""
API-related constants for standardized responses and endpoints.
"""

from typing import Dict, Any


class APIConstants:
    """API response constants and standardized messages."""
    
    # Standard API response version
    API_VERSION = "1.0"
    
    # Success response messages
    SUCCESS_DEFAULT = "Operation completed successfully"
    MATCH_JOB_QUEUED = "Match job has been queued for processing"
    MATCH_JOB_COMPLETED = "Match job completed successfully"
    DATA_RETRIEVED = "Data retrieved successfully"
    STATUS_UPDATED = "Status updated successfully"
    
    # Processing status messages
    STATUS_PENDING = "pending"
    STATUS_PROCESSING = "processing"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_ACCEPTED = "accepted"
    STATUS_COMPLETED_WITH_ERRORS = "completed_with_errors"
    
    # Current step descriptions
    STEP_QUEUED = "Queued for processing"
    STEP_PROCESSING_RESUME = "Processing resume"
    STEP_SAVING_RESUME = "Saving resume to profile service"
    STEP_FETCHING_JOBS = "Fetching job details via RabbitMQ"
    STEP_PROCESSING_JOBS = "Processing jobs for matching"
    STEP_RUNNING_ML = "Running ML matching algorithm"
    STEP_COMPLETED = "Completed"
    
    # Progress percentages for tracking
    PROGRESS_QUEUED = 0.0
    PROGRESS_RESUME_PARSED = 30.0
    PROGRESS_JOBS_FETCHED = 50.0
    PROGRESS_JOBS_PROCESSED = 70.0
    PROGRESS_ML_MATCHING = 85.0
    PROGRESS_COMPLETED = 100.0


class HTTPStatusMessages:
    """HTTP status code related messages."""
    
    # Success messages (2xx)
    SUCCESS_200 = "OK"
    ACCEPTED_202 = "Request accepted for processing"
    
    # Client error messages (4xx)
    BAD_REQUEST_400 = "Invalid request data provided"
    UNAUTHORIZED_401 = "Authentication required or token invalid"
    FORBIDDEN_403 = "Access denied for this resource"
    NOT_FOUND_404 = "Resource not found or access denied"
    CONFLICT_409 = "Request conflicts with current state"
    UNPROCESSABLE_422 = "Request data validation failed"
    TOO_MANY_REQUESTS_429 = "Rate limit exceeded"
    
    # Server error messages (5xx)
    INTERNAL_ERROR_500 = "Internal server error occurred"
    SERVICE_UNAVAILABLE_503 = "Service temporarily unavailable"
    GATEWAY_TIMEOUT_504 = "Request timeout - please try again"


class EndpointPaths:
    """Standardized endpoint path templates."""
    
    # Base paths
    API_V1_BASE = "/api/v1"
    MATCHING_BASE = f"{API_V1_BASE}/matching"
    HEALTH_BASE = f"{API_V1_BASE}/health"
    
    # Matching endpoints
    NEW_MATCH_ASYNC = f"{MATCHING_BASE}/new-match-async"
    MATCH_HISTORY = f"{MATCHING_BASE}/history"
    JOB_STATUS = f"{MATCHING_BASE}/job/{{match_job_id}}/status"
    JOB_RESULTS = f"{MATCHING_BASE}/job/{{match_job_id}}/results"
    
    # Health endpoints
    HEALTH_CHECK = f"{HEALTH_BASE}"
    HEALTH_DETAILED = f"{HEALTH_BASE}/detailed"
    
    @classmethod
    def get_job_status_path(cls, match_job_id: str) -> str:
        """Get formatted job status path."""
        return cls.JOB_STATUS.format(match_job_id=match_job_id)
    
    @classmethod
    def get_job_results_path(cls, match_job_id: str) -> str:
        """Get formatted job results path."""
        return cls.JOB_RESULTS.format(match_job_id=match_job_id)


class ResponseFields:
    """Standard response field names."""
    
    # Core response fields
    SUCCESS = "success"
    MESSAGE = "message"
    DATA = "data"
    META = "meta"
    TIMESTAMP = "timestamp"
    CORRELATION_ID = "correlation_id"
    VERSION = "version"
    
    # Error response fields
    ERROR = "error"
    ERROR_CODE = "code"
    ERROR_MESSAGE = "message"
    ERROR_DETAILS = "details"
    
    # Match job specific fields
    MATCH_JOB_ID = "match_job_id"
    STATUS = "status"
    CURRENT_STEP = "current_step"
    PROGRESS_PERCENTAGE = "progress_percentage"
    CREATED_AT = "created_at"
    STARTED_AT = "started_at"
    COMPLETED_AT = "completed_at"
    ERROR_MESSAGE_FIELD = "error_message"
    RESULTS_AVAILABLE = "results_available"
    MATCHES_COUNT = "matches_count"
    PARSED_RESUME_ID = "parsed_resume_id"
    MATCHES = "matches"
    PROCESSING_TIME_SECONDS = "processing_time_seconds"
    
    # Endpoint information
    ENDPOINTS = "endpoints"
    STATUS_ENDPOINT = "status"
    RESULTS_ENDPOINT = "results"


class ContentTypes:
    """Standard content type constants."""
    
    JSON = "application/json"
    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    DOC = "application/msword"
    TEXT = "text/plain"
    
    # Allowed resume file types
    ALLOWED_RESUME_TYPES = [PDF, DOCX, DOC, TEXT]


class CacheKeys:
    """Redis cache key templates."""
    
    # Match job caching
    MATCH_JOB_PREFIX = "match_job"
    JOB_DATA_PREFIX = "job_data"
    EMBEDDING_PREFIX = "embedding"
    USER_PROFILE_PREFIX = "user_profile"
    
    # Cache TTL values (in seconds)
    MATCH_JOB_TTL = 3600  # 1 hour
    JOB_DATA_TTL = 7200   # 2 hours
    EMBEDDING_TTL = 86400  # 24 hours
    USER_PROFILE_TTL = 1800  # 30 minutes
    
    @classmethod
    def get_match_job_key(cls, match_job_id: str) -> str:
        """Get formatted match job cache key."""
        return f"{cls.MATCH_JOB_PREFIX}:{match_job_id}"
    
    @classmethod
    def get_job_data_key(cls, job_id: int) -> str:
        """Get formatted job data cache key."""
        return f"{cls.JOB_DATA_PREFIX}:{job_id}"
    
    @classmethod
    def get_embedding_key(cls, text_hash: str) -> str:
        """Get formatted embedding cache key."""
        return f"{cls.EMBEDDING_PREFIX}:{text_hash}"
    
    @classmethod
    def get_user_profile_key(cls, user_id: int) -> str:
        """Get formatted user profile cache key."""
        return f"{cls.USER_PROFILE_PREFIX}:{user_id}"