"""
Staging environment configuration.
Production-like environment for pre-deployment testing and validation.
"""

from typing import List, Optional
from ..base_config import BaseConfig, Environment, LogLevel


class StagingConfig(BaseConfig):
    """
    Staging environment configuration.
    
    Features:
    - Production-like settings with safety nets
    - Enhanced debugging capabilities
    - Load testing friendly configuration
    - Integration testing optimized
    - Performance monitoring enabled
    """
    
    # Environment
    ENVIRONMENT: Environment = Environment.STAGING
    DEBUG: bool = False
    RELOAD: bool = False
    WORKERS: int = 2  # Moderate workers for staging
    
    # Logging - Detailed for staging validation
    LOG_LEVEL: LogLevel = LogLevel.INFO
    ENABLE_REQUEST_LOGGING: bool = True
    ENABLE_PERFORMANCE_LOGGING: bool = True
    LOG_FILE: str = "/var/log/resume_matcher/staging.log"
    DATABASE_ECHO: bool = False
    
    # CORS - Staging and testing origins
    CORS_ORIGINS: List[str] = [
        "https://staging.resumematcher.com",
        "https://staging-admin.resumematcher.com",
        "https://test.resumematcher.com",
        "http://localhost:3000",  # For local testing against staging
    ]
    
    # Security - Production-like with some relaxation
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60  # Longer for testing
    REFRESH_TOKEN_EXPIRE_DAYS: int = 14
    PASSWORD_MIN_LENGTH: int = 8
    ENABLE_RATE_LIMITING: bool = True
    ENABLE_HTTPS: bool = True
    SSL_CERT_PATH: str = "/etc/ssl/certs/staging-resumematcher.crt"
    SSL_KEY_PATH: str = "/etc/ssl/private/staging-resumematcher.key"
    ENABLE_API_KEY_AUTH: bool = True
    
    # Database - Staging PostgreSQL
    POSTGRES_SERVER: str = "staging-db.resume-matcher.internal"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "resume_matcher_staging"
    POSTGRES_PASSWORD: str = "${STAGING_DB_PASSWORD}"
    POSTGRES_DB: str = "resume_matcher_staging"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 15
    DATABASE_POOL_TIMEOUT: int = 30
    
    # Redis - Staging Redis
    REDIS_HOST: str = "staging-redis.resume-matcher.internal"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = "${STAGING_REDIS_PASSWORD}"
    REDIS_DB: int = 0
    REDIS_MAX_CONNECTIONS: int = 25
    REDIS_SOCKET_TIMEOUT: int = 5
    
    # Celery - Staging optimized
    CELERY_BROKER_URL: str = "redis://:${STAGING_REDIS_PASSWORD}@staging-redis.resume-matcher.internal:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://:${STAGING_REDIS_PASSWORD}@staging-redis.resume-matcher.internal:6379/1"
    CELERY_WORKER_PREFETCH_MULTIPLIER: int = 2
    CELERY_MAX_RETRIES: int = 3
    CELERY_RETRY_DELAY: int = 30  # Faster retry for staging
    
    # RabbitMQ - Staging cluster
    RABBITMQ_URL: str = "amqp://${STAGING_RABBITMQ_USER}:${STAGING_RABBITMQ_PASSWORD}@staging-rabbitmq.resume-matcher.internal:5672/"
    RABBITMQ_HOST: str = "staging-rabbitmq.resume-matcher.internal"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "resume_matcher_staging"
    RABBITMQ_PASSWORD: str = "${STAGING_RABBITMQ_PASSWORD}"
    
    # Service URLs - Staging microservices
    JOBS_SERVICE_URL: str = "https://staging-jobs-service.resume-matcher.internal"
    AUTH_SERVICE_URL: str = "https://staging-auth-service.resume-matcher.internal"
    PROFILE_SERVICE_URL: str = "https://staging-profile-service.resume-matcher.internal"
    NOTIFICATION_SERVICE_URL: str = "https://staging-notification-service.resume-matcher.internal"
    
    # File Processing - Moderate limits for testing
    MAX_FILE_SIZE: int = 8 * 1024 * 1024  # 8MB
    ALLOWED_FILE_TYPES: List[str] = [".pdf", ".docx", ".doc", ".txt"]
    UPLOAD_TEMP_DIR: str = "/tmp/staging-uploads"
    FILE_CLEANUP_AFTER_MINUTES: int = 45
    
    # ML Configuration - Staging optimized
    ML_MODEL_CACHE_SIZE: int = 150
    ML_MODEL_CACHE_TTL: int = 2700  # 45 minutes
    ML_BATCH_SIZE: int = 32
    ML_ENABLE_GPU: bool = True
    ML_MODEL_PATH: str = "/opt/models/staging"
    ML_SIMILARITY_THRESHOLD: float = 0.7
    
    # Business Rules - Staging limits
    MAX_JOBS_PER_REQUEST: int = 75
    MAX_RESUME_SIZE_MB: int = 8
    DEFAULT_MATCH_LIMIT: int = 30
    MIN_MATCH_SCORE: float = 0.3
    
    # Rate Limiting - Moderate for load testing
    RATE_LIMIT_PER_MINUTE: int = 120
    RATE_LIMIT_PER_HOUR: int = 2000
    RATE_LIMIT_PER_DAY: int = 20000
    
    # Monitoring - Comprehensive staging monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    PROMETHEUS_NAMESPACE: str = "resume_matcher_staging"
    ENABLE_TRACING: bool = True
    JAEGER_ENDPOINT: str = "https://staging-jaeger-collector.monitoring.internal:14268/api/traces"
    
    # Health Checks - Staging intervals
    HEALTH_CHECK_INTERVAL: int = 20  # seconds
    DATABASE_HEALTH_TIMEOUT: int = 8
    REDIS_HEALTH_TIMEOUT: int = 5
    SERVICE_HEALTH_TIMEOUT: int = 12
    
    # Cache - Staging optimized
    CACHE_DEFAULT_TTL: int = 2700  # 45 minutes
    CACHE_MAX_SIZE: int = 1500
    ENABLE_QUERY_CACHE: bool = True
    ENABLE_RESULT_CACHE: bool = True
    
    # Feature Flags - All enabled for staging validation
    ENABLE_AB_TESTING: bool = True
    ENABLE_ADVANCED_ML: bool = True
    ENABLE_ANALYTICS: bool = True
    ENABLE_FEEDBACK_COLLECTION: bool = True
    ENABLE_RECOMMENDATION_ENGINE: bool = True
    
    # API Documentation - Available for staging validation
    DOCS_URL: str = "/api/v1/docs"
    REDOC_URL: str = "/api/v1/redoc" 
    OPENAPI_URL: str = "/api/v1/openapi.json"
    
    # Staging-specific Settings
    ENABLE_DEBUG_ENDPOINTS: bool = True  # Useful for staging debugging
    ENABLE_MOCK_SERVICES: bool = False
    SHOW_STACK_TRACE: bool = True  # Helpful for debugging
    ENABLE_SQL_LOGGING: bool = False
    ENABLE_QUERY_PROFILING: bool = True  # Performance testing
    
    # Load Testing Settings
    ENABLE_LOAD_TEST_MODE: bool = True
    MAX_CONCURRENT_REQUESTS: int = 100
    BULK_OPERATION_LIMIT: int = 50
    
    # Performance Settings
    DATABASE_STATEMENT_TIMEOUT: int = 45000  # 45 seconds
    API_TIMEOUT: int = 45  # seconds
    ASYNC_TASK_TIMEOUT: int = 600  # 10 minutes
    
    # Integration Testing
    ENABLE_INTEGRATION_TESTS: bool = True
    EXTERNAL_SERVICE_TIMEOUT: int = 30
    RETRY_EXTERNAL_SERVICES: bool = True
    MAX_EXTERNAL_SERVICE_RETRIES: int = 2
    
    # Data Management
    ENABLE_DATA_SEEDING: bool = True
    STAGING_DATA_RESET_SCHEDULE: str = "0 3 * * 0"  # Weekly Sunday 3 AM
    ANONYMIZE_PRODUCTION_DATA: bool = True
    
    # Alerting Thresholds - Relaxed for staging
    CPU_ALERT_THRESHOLD: float = 85.0  # percent
    MEMORY_ALERT_THRESHOLD: float = 90.0  # percent
    DISK_ALERT_THRESHOLD: float = 85.0  # percent
    ERROR_RATE_ALERT_THRESHOLD: float = 10.0  # percent
    RESPONSE_TIME_ALERT_THRESHOLD: float = 3000.0  # milliseconds
    
    # SLA Requirements - Staging targets
    TARGET_AVAILABILITY: float = 99.0  # percent
    TARGET_RESPONSE_TIME: float = 500.0  # milliseconds
    TARGET_ERROR_RATE: float = 2.0  # percent
    
    # Backup Settings
    ENABLE_AUTOMATED_BACKUPS: bool = True
    BACKUP_RETENTION_DAYS: int = 7
    BACKUP_SCHEDULE: str = "0 4 * * *"  # Daily at 4 AM