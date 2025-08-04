"""
Development environment configuration.
Optimized for local development with enhanced debugging and relaxed security.
"""

from typing import List
from ..base_config import BaseConfig, Environment, LogLevel


class DevelopmentConfig(BaseConfig):
    """
    Development environment configuration.
    
    Features:
    - Enhanced debugging and logging
    - Auto-reload enabled
    - Relaxed security settings
    - Local service endpoints
    - Comprehensive error reporting
    """
    
    # Environment
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = True
    RELOAD: bool = True
    
    # Logging - Verbose for development
    LOG_LEVEL: LogLevel = LogLevel.DEBUG
    ENABLE_REQUEST_LOGGING: bool = True
    ENABLE_PERFORMANCE_LOGGING: bool = True
    LOG_FILE: str = "logs/development.log"
    DATABASE_ECHO: bool = True  # Show SQL queries
    
    # CORS - Allow all origins for development
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ]
    
    # Security - Relaxed for development
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 180  # 3 hours
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    ENABLE_RATE_LIMITING: bool = False
    ENABLE_HTTPS: bool = False
    ENABLE_API_KEY_AUTH: bool = False
    
    # Database - Local PostgreSQL
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "resume_matcher_dev"
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 5
    
    # Redis - Local Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    
    # Celery - Development settings
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    CELERY_WORKER_PREFETCH_MULTIPLIER: int = 1  # Lower for development
    
    # RabbitMQ - Local RabbitMQ
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672/"
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "guest"
    RABBITMQ_PASSWORD: str = "guest"
    
    # Service URLs - Local services
    JOBS_SERVICE_URL: str = "http://localhost:8001"
    AUTH_SERVICE_URL: str = "http://localhost:8002"
    PROFILE_SERVICE_URL: str = "http://localhost:8003"
    NOTIFICATION_SERVICE_URL: str = "http://localhost:8004"
    
    # File Processing - Permissive for development
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = [".pdf", ".docx", ".doc", ".txt", ".rtf"]
    UPLOAD_TEMP_DIR: str = "./temp/uploads"
    FILE_CLEANUP_AFTER_MINUTES: int = 30
    
    # ML Configuration - Development optimized
    ML_MODEL_CACHE_SIZE: int = 50
    ML_MODEL_CACHE_TTL: int = 1800  # 30 minutes
    ML_BATCH_SIZE: int = 16  # Smaller for development
    ML_ENABLE_GPU: bool = False
    ML_MODEL_PATH: str = "./models/dev"
    
    # Business Rules - Relaxed for testing
    MAX_JOBS_PER_REQUEST: int = 100
    MAX_RESUME_SIZE_MB: int = 10
    DEFAULT_MATCH_LIMIT: int = 50
    MIN_MATCH_SCORE: float = 0.1  # Lower threshold for testing
    
    # Rate Limiting - Disabled for development
    RATE_LIMIT_PER_MINUTE: int = 1000
    RATE_LIMIT_PER_HOUR: int = 10000
    RATE_LIMIT_PER_DAY: int = 100000
    
    # Monitoring - Enabled but less strict
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    ENABLE_TRACING: bool = True
    JAEGER_ENDPOINT: str = "http://localhost:14268/api/traces"
    
    # Health Checks - Frequent for development
    HEALTH_CHECK_INTERVAL: int = 15  # seconds
    DATABASE_HEALTH_TIMEOUT: int = 10
    REDIS_HEALTH_TIMEOUT: int = 5
    SERVICE_HEALTH_TIMEOUT: int = 15
    
    # Cache - Shorter TTL for development
    CACHE_DEFAULT_TTL: int = 900  # 15 minutes
    CACHE_MAX_SIZE: int = 500
    ENABLE_QUERY_CACHE: bool = True
    ENABLE_RESULT_CACHE: bool = True
    
    # Feature Flags - All enabled for development
    ENABLE_AB_TESTING: bool = True
    ENABLE_ADVANCED_ML: bool = True
    ENABLE_ANALYTICS: bool = True
    ENABLE_FEEDBACK_COLLECTION: bool = True
    ENABLE_RECOMMENDATION_ENGINE: bool = True
    
    # API Documentation - Full access in development
    DOCS_URL: str = "/api/v1/docs"
    REDOC_URL: str = "/api/v1/redoc"
    OPENAPI_URL: str = "/api/v1/openapi.json"
    
    # Additional Development Features
    ENABLE_DEBUG_ENDPOINTS: bool = True
    ENABLE_MOCK_SERVICES: bool = False
    SHOW_STACK_TRACE: bool = True
    ENABLE_SQL_LOGGING: bool = True
    ENABLE_QUERY_PROFILING: bool = True