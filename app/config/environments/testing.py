"""
Testing environment configuration.
Optimized for automated testing with fast execution and isolation.
"""

from typing import List
from ..base_config import BaseConfig, Environment, LogLevel


class TestingConfig(BaseConfig):
    """
    Testing environment configuration.
    
    Features:
    - Fast test execution
    - In-memory databases for isolation
    - Disabled external services
    - Minimal logging to reduce noise
    - Mock configurations
    """
    
    # Environment
    ENVIRONMENT: Environment = Environment.TESTING
    DEBUG: bool = False
    RELOAD: bool = False
    
    # Logging - Minimal for testing
    LOG_LEVEL: LogLevel = LogLevel.WARNING
    ENABLE_REQUEST_LOGGING: bool = False
    ENABLE_PERFORMANCE_LOGGING: bool = False
    LOG_FILE: str = "logs/testing.log"
    DATABASE_ECHO: bool = False
    
    # CORS - Permissive for testing
    CORS_ORIGINS: List[str] = ["*"]
    
    # Security - Relaxed for testing
    SECRET_KEY: str = "test-secret-key-not-for-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 1
    ENABLE_RATE_LIMITING: bool = False
    ENABLE_HTTPS: bool = False
    ENABLE_API_KEY_AUTH: bool = False
    
    # Database - In-memory SQLite for fast testing
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "test_user"
    POSTGRES_PASSWORD: str = "test_password"
    POSTGRES_DB: str = "test_resume_matcher"
    DATABASE_URL: str = "sqlite:///./test_resume_matcher.db"
    DATABASE_POOL_SIZE: int = 1
    DATABASE_MAX_OVERFLOW: int = 0
    DATABASE_POOL_TIMEOUT: int = 5
    
    # Redis - Mock Redis for testing
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 15  # Use a different DB for tests
    REDIS_PASSWORD: str = ""
    REDIS_MAX_CONNECTIONS: int = 5
    
    # Celery - Synchronous execution for testing
    CELERY_TASK_ALWAYS_EAGER: bool = True  # Execute tasks synchronously
    CELERY_TASK_EAGER_PROPAGATES: bool = True  # Propagate exceptions
    CELERY_BROKER_URL: str = "memory://"
    CELERY_RESULT_BACKEND: str = "cache+memory://"
    CELERY_WORKER_PREFETCH_MULTIPLIER: int = 1
    
    # RabbitMQ - Mock for testing
    RABBITMQ_URL: str = "memory://"
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "test"
    RABBITMQ_PASSWORD: str = "test"
    
    # Service URLs - Mock services for testing
    JOBS_SERVICE_URL: str = "http://mock-jobs-service"
    AUTH_SERVICE_URL: str = "http://mock-auth-service"
    PROFILE_SERVICE_URL: str = "http://mock-profile-service"
    NOTIFICATION_SERVICE_URL: str = "http://mock-notification-service"
    
    # File Processing - Test-friendly settings
    MAX_FILE_SIZE: int = 1024 * 1024  # 1MB for tests
    ALLOWED_FILE_TYPES: List[str] = [".pdf", ".docx", ".txt"]
    UPLOAD_TEMP_DIR: str = "./test_temp"
    FILE_CLEANUP_AFTER_MINUTES: int = 1  # Quick cleanup
    
    # ML Configuration - Fast testing
    ML_MODEL_CACHE_SIZE: int = 10
    ML_MODEL_CACHE_TTL: int = 60  # 1 minute
    ML_BATCH_SIZE: int = 4  # Small batches for tests
    ML_ENABLE_GPU: bool = False
    ML_MODEL_PATH: str = "./test_models"
    ML_SIMILARITY_THRESHOLD: float = 0.5
    
    # Business Rules - Test-friendly limits
    MAX_JOBS_PER_REQUEST: int = 10
    MAX_RESUME_SIZE_MB: int = 1
    DEFAULT_MATCH_LIMIT: int = 5
    MIN_MATCH_SCORE: float = 0.0  # Allow all matches for testing
    
    # Rate Limiting - Disabled for testing
    RATE_LIMIT_PER_MINUTE: int = 10000
    RATE_LIMIT_PER_HOUR: int = 100000
    RATE_LIMIT_PER_DAY: int = 1000000
    
    # Monitoring - Disabled for testing
    ENABLE_METRICS: bool = False
    METRICS_PORT: int = 9091
    ENABLE_TRACING: bool = False
    JAEGER_ENDPOINT: str = ""
    
    # Health Checks - Frequent for testing
    HEALTH_CHECK_INTERVAL: int = 5  # seconds
    DATABASE_HEALTH_TIMEOUT: int = 1
    REDIS_HEALTH_TIMEOUT: int = 1
    SERVICE_HEALTH_TIMEOUT: int = 2
    
    # Cache - Minimal for testing
    CACHE_DEFAULT_TTL: int = 60  # 1 minute
    CACHE_MAX_SIZE: int = 10
    ENABLE_QUERY_CACHE: bool = False  # Disable for predictable tests
    ENABLE_RESULT_CACHE: bool = False  # Disable for predictable tests
    
    # Feature Flags - Controllable for testing
    ENABLE_AB_TESTING: bool = False
    ENABLE_ADVANCED_ML: bool = False  # Use simple matching for tests
    ENABLE_ANALYTICS: bool = False
    ENABLE_FEEDBACK_COLLECTION: bool = False
    ENABLE_RECOMMENDATION_ENGINE: bool = False
    
    # API Documentation - Available for test inspection
    DOCS_URL: str = "/api/v1/docs"
    REDOC_URL: str = "/api/v1/redoc"
    OPENAPI_URL: str = "/api/v1/openapi.json"
    
    # Testing-specific Settings
    ENABLE_DEBUG_ENDPOINTS: bool = True
    ENABLE_MOCK_SERVICES: bool = True
    SHOW_STACK_TRACE: bool = True
    ENABLE_SQL_LOGGING: bool = False
    ENABLE_QUERY_PROFILING: bool = False
    
    # Test Database Settings
    TEST_DATABASE_URL: str = "sqlite:///:memory:"
    USE_IN_MEMORY_DB: bool = True
    RESET_DB_BETWEEN_TESTS: bool = True
    
    # Mock Settings
    USE_MOCK_ML_MODELS: bool = True
    USE_MOCK_EXTERNAL_SERVICES: bool = True
    MOCK_API_RESPONSES: bool = True
    
    # Performance Settings for Fast Tests
    ASYNC_TASK_TIMEOUT: int = 5  # seconds
    API_TIMEOUT: int = 5  # seconds
    DATABASE_STATEMENT_TIMEOUT: int = 1000  # 1 second
    
    # Test Data Settings
    CREATE_TEST_DATA: bool = True
    TEST_USER_EMAIL: str = "test@example.com"
    TEST_USER_PASSWORD: str = "testpassword123"
    
    def get_test_database_config(self) -> dict:
        """Get test-specific database configuration."""
        return {
            "url": self.TEST_DATABASE_URL,
            "echo": False,
            "pool_pre_ping": True,
            "connect_args": {"check_same_thread": False} if "sqlite" in self.TEST_DATABASE_URL else {},
        }