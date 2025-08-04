"""
Business rules and processing limits for the resume job matcher service.
"""

from typing import List, Dict


class BusinessRules:
    """Core business logic constants and rules."""
    
    # Job matching limits
    MAX_JOB_IDS_PER_REQUEST = 50
    MIN_JOB_IDS_PER_REQUEST = 1
    DEFAULT_TOP_MATCHES = 5
    MAX_TOP_MATCHES = 20
    
    # Skill matching rules
    MIN_SKILLS_FOR_RELIABLE_MATCH = 3
    MAX_SKILLS_PER_JOB = 50
    MIN_SKILL_CONFIDENCE = 0.8
    
    # Score thresholds
    MIN_MATCH_SCORE = 0.3
    EXCELLENT_MATCH_THRESHOLD = 0.8
    GOOD_MATCH_THRESHOLD = 0.6
    FAIR_MATCH_THRESHOLD = 0.4
    
    # Experience matching
    MAX_EXPERIENCE_YEARS = 50
    MIN_EXPERIENCE_YEARS = 0
    EXPERIENCE_WEIGHT_THRESHOLD = 0.7
    
    # Location matching
    LOCATION_EXACT_MATCH_SCORE = 1.0
    LOCATION_PARTIAL_MATCH_SCORE = 0.7
    LOCATION_NO_MATCH_SCORE = 0.0
    REMOTE_WORK_KEYWORDS = ["remote", "work from home", "wfh", "telecommute", "virtual"]
    
    # Salary matching tolerances
    SALARY_EXACT_MATCH_TOLERANCE = 0.1  # 10%
    SALARY_GOOD_MATCH_TOLERANCE = 0.2   # 20%
    SALARY_FAIR_MATCH_TOLERANCE = 0.3   # 30%


class ProcessingLimits:
    """File and data processing limits."""
    
    # File size limits (in bytes)
    MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
    MIN_FILE_SIZE_BYTES = 100  # 100 bytes
    
    # Resume processing limits
    MAX_RESUME_TEXT_LENGTH = 50000  # characters
    MIN_RESUME_TEXT_LENGTH = 100    # characters
    MAX_RESUME_PAGES = 10
    
    # Job description limits
    MAX_JOB_DESCRIPTION_LENGTH = 10000  # characters
    MIN_JOB_DESCRIPTION_LENGTH = 50     # characters
    
    # Batch processing limits
    MAX_BATCH_SIZE = 100
    MIN_BATCH_SIZE = 1
    OPTIMAL_BATCH_SIZE = 32
    
    # Concurrent processing limits
    MAX_CONCURRENT_REQUESTS = 10
    MAX_WORKER_CONCURRENCY = 4
    DEFAULT_WORKER_CONCURRENCY = 2
    
    # Memory limits
    MAX_EMBEDDING_CACHE_SIZE = 10000
    MAX_JOB_CACHE_SIZE = 5000
    MAX_RESULT_CACHE_SIZE = 1000


class TimeoutSettings:
    """Timeout configurations for various operations."""
    
    # Celery task timeouts (in seconds)
    DEFAULT_TASK_TIMEOUT = 600      # 10 minutes
    RESUME_PARSING_TIMEOUT = 300    # 5 minutes
    JOB_FETCHING_TIMEOUT = 180      # 3 minutes
    ML_MATCHING_TIMEOUT = 600       # 10 minutes
    FILE_UPLOAD_TIMEOUT = 60        # 1 minute
    
    # HTTP request timeouts (in seconds)
    HTTP_REQUEST_TIMEOUT = 30
    HTTP_CONNECT_TIMEOUT = 10
    HTTP_READ_TIMEOUT = 60
    
    # RabbitMQ timeouts (in seconds)
    RABBITMQ_CONNECT_TIMEOUT = 30
    RABBITMQ_REQUEST_TIMEOUT = 45
    RABBITMQ_BULK_REQUEST_TIMEOUT = 60
    
    # Cache timeouts (in seconds)
    DEFAULT_CACHE_TTL = 3600        # 1 hour
    SHORT_CACHE_TTL = 300           # 5 minutes
    LONG_CACHE_TTL = 86400          # 24 hours
    
    # Database operation timeouts (in seconds)
    DB_QUERY_TIMEOUT = 30
    DB_TRANSACTION_TIMEOUT = 60
    DB_CONNECTION_TIMEOUT = 30


class QueueSettings:
    """Task queue and messaging configuration."""
    
    # Celery queue names
    DEFAULT_QUEUE = "default"
    MODEL_TRAINING_QUEUE = "model_training"
    ANALYSIS_QUEUE = "analysis"
    HIGH_PRIORITY_QUEUE = "high_priority"
    
    # Queue routing keys
    DEFAULT_ROUTING_KEY = "default"
    MODEL_TRAINING_ROUTING_KEY = "model.training"
    ANALYSIS_ROUTING_KEY = "analysis"
    
    # Queue priorities
    HIGH_PRIORITY = 9
    NORMAL_PRIORITY = 5
    LOW_PRIORITY = 1
    
    # Worker settings
    WORKER_PREFETCH_MULTIPLIER = 1
    WORKER_MAX_TASKS_PER_CHILD = 1000
    
    # Result settings
    RESULT_EXPIRES = 3600  # 1 hour
    TASK_TRACK_STARTED = True
    TASK_ACKS_LATE = True
    TASK_REJECT_ON_WORKER_LOST = True


class RetrySettings:
    """Retry configuration for various operations."""
    
    # Default retry settings
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 60  # seconds
    
    # Task-specific retry settings
    RESUME_PARSING_MAX_RETRIES = 2
    RESUME_PARSING_RETRY_DELAY = 30
    
    JOB_FETCHING_MAX_RETRIES = 3
    JOB_FETCHING_RETRY_DELAY = 60
    
    ML_MATCHING_MAX_RETRIES = 2
    ML_MATCHING_RETRY_DELAY = 120
    
    # HTTP retry settings
    HTTP_MAX_RETRIES = 3
    HTTP_RETRY_DELAY = 5
    HTTP_BACKOFF_FACTOR = 2.0
    
    # Database retry settings
    DB_MAX_RETRIES = 3
    DB_RETRY_DELAY = 1
    
    # RabbitMQ retry settings
    RABBITMQ_MAX_RETRIES = 5
    RABBITMQ_RETRY_DELAY = 10


class FeatureFlags:
    """Feature flags for enabling/disabling functionality."""
    
    # Core features
    ENABLE_ASYNC_PROCESSING = True
    ENABLE_CACHING = True
    ENABLE_MONITORING = True
    ENABLE_CIRCUIT_BREAKER = True
    
    # ML features
    ENABLE_SEMANTIC_MATCHING = True
    ENABLE_SKILL_EXTRACTION = True
    ENABLE_ENSEMBLE_MATCHING = False  # To be enabled with Enhanced ML Pipeline
    ENABLE_LEARNING_TO_RANK = False   # To be enabled with Enhanced ML Pipeline
    
    # Integration features
    ENABLE_PROFILE_SERVICE = True
    ENABLE_RABBITMQ = True
    ENABLE_HTTP_FALLBACK = True
    
    # Development features
    ENABLE_DEBUG_LOGGING = False
    ENABLE_PERFORMANCE_METRICS = True
    ENABLE_REQUEST_TRACING = True
    
    # Security features
    ENABLE_RATE_LIMITING = True
    ENABLE_INPUT_SANITIZATION = True
    ENABLE_FILE_VALIDATION = True


class DefaultValues:
    """Default values for various configurations."""
    
    # Pagination defaults
    DEFAULT_PAGE_SIZE = 10
    MAX_PAGE_SIZE = 100
    MIN_PAGE_SIZE = 1
    
    # Matching defaults
    DEFAULT_SIMILARITY_THRESHOLD = 0.75
    DEFAULT_CONFIDENCE_THRESHOLD = 0.6
    DEFAULT_MIN_SCORE = 0.3
    
    # File processing defaults
    DEFAULT_FILE_ENCODING = "utf-8"
    DEFAULT_TEXT_LANGUAGE = "en"
    
    # User preferences defaults
    DEFAULT_JOB_TYPE = "full-time"
    DEFAULT_LOCATION_PREFERENCE = "any"
    DEFAULT_SALARY_EXPECTATION = "not specified"
    
    # Cache defaults
    DEFAULT_CACHE_SIZE = 1000
    DEFAULT_CACHE_EXPIRY = 3600