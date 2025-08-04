"""
Production environment configuration.
Optimized for performance, security, and reliability in production deployment.
"""

from typing import List, Optional
from ..base_config import BaseConfig, Environment, LogLevel


class ProductionConfig(BaseConfig):
    """
    Production environment configuration.
    
    Features:
    - Enhanced security and authentication
    - Performance optimizations
    - Comprehensive monitoring and alerting
    - Strict rate limiting and validation
    - High availability settings
    """
    
    # Environment
    ENVIRONMENT: Environment = Environment.PRODUCTION
    DEBUG: bool = False
    RELOAD: bool = False
    WORKERS: int = 4  # Multiple workers for production
    
    # Logging - Structured and efficient
    LOG_LEVEL: LogLevel = LogLevel.INFO
    ENABLE_REQUEST_LOGGING: bool = True
    ENABLE_PERFORMANCE_LOGGING: bool = True
    LOG_FILE: str = "/var/log/resume_matcher/app.log"
    DATABASE_ECHO: bool = False  # No SQL logging in production
    
    # CORS - Restricted to known origins
    CORS_ORIGINS: List[str] = [
        "https://app.resumematcher.com",
        "https://admin.resumematcher.com",
        "https://api.resumematcher.com",
    ]
    
    # Security - Maximum security
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    PASSWORD_MIN_LENGTH: int = 12
    ENABLE_RATE_LIMITING: bool = True
    ENABLE_HTTPS: bool = True
    SSL_CERT_PATH: str = "/etc/ssl/certs/resumematcher.crt"
    SSL_KEY_PATH: str = "/etc/ssl/private/resumematcher.key"
    ENABLE_API_KEY_AUTH: bool = True
    ENABLE_IP_WHITELIST: bool = False  # Set to True with ALLOWED_IPS if needed
    
    # Database - Production PostgreSQL cluster
    POSTGRES_SERVER: str = "prod-db-cluster.resume-matcher.internal"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "resume_matcher_app"
    POSTGRES_PASSWORD: str = "${DB_PASSWORD}"  # From environment/secrets
    POSTGRES_DB: str = "resume_matcher_prod"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 30
    DATABASE_POOL_TIMEOUT: int = 30
    
    # Redis - Production Redis cluster
    REDIS_HOST: str = "prod-redis-cluster.resume-matcher.internal"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = "${REDIS_PASSWORD}"  # From environment/secrets
    REDIS_DB: int = 0
    REDIS_MAX_CONNECTIONS: int = 50
    REDIS_SOCKET_TIMEOUT: int = 10
    
    # Celery - Production optimized
    CELERY_BROKER_URL: str = "redis://:${REDIS_PASSWORD}@prod-redis-cluster.resume-matcher.internal:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://:${REDIS_PASSWORD}@prod-redis-cluster.resume-matcher.internal:6379/1"
    CELERY_WORKER_PREFETCH_MULTIPLIER: int = 4
    CELERY_MAX_RETRIES: int = 3
    CELERY_RETRY_DELAY: int = 60
    
    # RabbitMQ - Production cluster
    RABBITMQ_URL: str = "amqp://${RABBITMQ_USER}:${RABBITMQ_PASSWORD}@prod-rabbitmq-cluster.resume-matcher.internal:5672/"
    RABBITMQ_HOST: str = "prod-rabbitmq-cluster.resume-matcher.internal"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "resume_matcher_app"
    RABBITMQ_PASSWORD: str = "${RABBITMQ_PASSWORD}"
    
    # Service URLs - Production microservices
    JOBS_SERVICE_URL: str = "https://jobs-service.resume-matcher.internal"
    AUTH_SERVICE_URL: str = "https://auth-service.resume-matcher.internal"
    PROFILE_SERVICE_URL: str = "https://profile-service.resume-matcher.internal"
    NOTIFICATION_SERVICE_URL: str = "https://notification-service.resume-matcher.internal"
    
    # File Processing - Strict limits
    MAX_FILE_SIZE: int = 5 * 1024 * 1024  # 5MB
    ALLOWED_FILE_TYPES: List[str] = [".pdf", ".docx", ".doc"]
    UPLOAD_TEMP_DIR: str = "/tmp/secure-uploads"
    FILE_CLEANUP_AFTER_MINUTES: int = 30
    
    # ML Configuration - Production optimized
    ML_MODEL_CACHE_SIZE: int = 200
    ML_MODEL_CACHE_TTL: int = 3600  # 1 hour
    ML_BATCH_SIZE: int = 64  # Larger for production efficiency
    ML_ENABLE_GPU: bool = True
    ML_MODEL_PATH: str = "/opt/models/production"
    ML_SIMILARITY_THRESHOLD: float = 0.75  # Higher threshold
    
    # Business Rules - Production limits
    MAX_JOBS_PER_REQUEST: int = 50
    MAX_RESUME_SIZE_MB: int = 5
    DEFAULT_MATCH_LIMIT: int = 20
    MIN_MATCH_SCORE: float = 0.4  # Higher minimum score
    
    # Rate Limiting - Strict production limits
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    RATE_LIMIT_PER_DAY: int = 10000
    
    # Monitoring - Comprehensive production monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    PROMETHEUS_NAMESPACE: str = "resume_matcher_prod"
    ENABLE_TRACING: bool = True
    JAEGER_ENDPOINT: str = "https://jaeger-collector.monitoring.internal:14268/api/traces"
    
    # Health Checks - Production intervals
    HEALTH_CHECK_INTERVAL: int = 30  # seconds
    DATABASE_HEALTH_TIMEOUT: int = 5
    REDIS_HEALTH_TIMEOUT: int = 3
    SERVICE_HEALTH_TIMEOUT: int = 10
    
    # Cache - Production optimized
    CACHE_DEFAULT_TTL: int = 3600  # 1 hour
    CACHE_MAX_SIZE: int = 2000
    ENABLE_QUERY_CACHE: bool = True
    ENABLE_RESULT_CACHE: bool = True
    
    # Feature Flags - Production stable features
    ENABLE_AB_TESTING: bool = True
    ENABLE_ADVANCED_ML: bool = True
    ENABLE_ANALYTICS: bool = True
    ENABLE_FEEDBACK_COLLECTION: bool = True
    ENABLE_RECOMMENDATION_ENGINE: bool = True
    
    # API Documentation - Restricted in production
    DOCS_URL: Optional[str] = None  # Disabled in production
    REDOC_URL: Optional[str] = None  # Disabled in production
    OPENAPI_URL: Optional[str] = None  # Disabled in production
    
    # Production-specific Settings
    ENABLE_DEBUG_ENDPOINTS: bool = False
    ENABLE_MOCK_SERVICES: bool = False
    SHOW_STACK_TRACE: bool = False
    ENABLE_SQL_LOGGING: bool = False
    ENABLE_QUERY_PROFILING: bool = False
    
    # Performance Settings
    DATABASE_STATEMENT_TIMEOUT: int = 30000  # 30 seconds
    API_TIMEOUT: int = 30  # seconds
    ASYNC_TASK_TIMEOUT: int = 300  # 5 minutes
    
    # Backup and Recovery
    ENABLE_AUTOMATED_BACKUPS: bool = True
    BACKUP_RETENTION_DAYS: int = 30
    BACKUP_SCHEDULE: str = "0 2 * * *"  # Daily at 2 AM
    
    # Alerting Thresholds
    CPU_ALERT_THRESHOLD: float = 80.0  # percent
    MEMORY_ALERT_THRESHOLD: float = 85.0  # percent
    DISK_ALERT_THRESHOLD: float = 90.0  # percent
    ERROR_RATE_ALERT_THRESHOLD: float = 5.0  # percent
    RESPONSE_TIME_ALERT_THRESHOLD: float = 2000.0  # milliseconds
    
    # SLA Requirements
    TARGET_AVAILABILITY: float = 99.9  # percent
    TARGET_RESPONSE_TIME: float = 200.0  # milliseconds
    TARGET_ERROR_RATE: float = 0.5  # percent