"""
Base configuration class for the job matcher application.
This provides the foundation for environment-specific configurations.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ConfigDict, AnyUrl
from typing import List, Dict, Any, Optional
from enum import Enum
import logging


class Environment(str, Enum):
    """Application environment enumeration."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class BaseConfig(BaseSettings):
    """
    Base configuration class containing common settings across all environments.
    Environment-specific configurations should inherit from this class.
    """

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Application Information
    APP_NAME: str = "Resume Job Matcher API"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = (
        "Enterprise resume-to-job matching service with ML-powered recommendations"
    )
    ENVIRONMENT: Environment = Environment.DEVELOPMENT

    # API Configuration
    API_V1_STR: str = "/api/v1"
    API_TITLE: str = "Resume Job Matcher API"
    OPENAPI_URL: str = "/api/v1/openapi.json"
    DOCS_URL: str = "/api/v1/docs"
    REDOC_URL: str = "/api/v1/redoc"

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    RELOAD: bool = False
    WORKERS: int = 1

    # Security Configuration
    SECRET_KEY: str = "development-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    PASSWORD_MIN_LENGTH: int = 8

    # Database Configuration
    POSTGRES_SERVER: str = ""
    POSTGRES_USER: str = ""
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = ""
    POSTGRES_PORT: int = 5432
    DATABASE_URL: Optional[AnyUrl] = None
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_ECHO: bool = False

    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    REDIS_MAX_CONNECTIONS: int = 10
    REDIS_SOCKET_TIMEOUT: int = 5

    # Celery Configuration
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_ACCEPT_CONTENT: List[str] = ["json"]
    CELERY_TIMEZONE: str = "UTC"
    CELERY_ENABLE_UTC: bool = True
    CELERY_WORKER_PREFETCH_MULTIPLIER: int = 4
    CELERY_MAX_RETRIES: int = 3
    CELERY_RETRY_DELAY: int = 60

    # RabbitMQ Configuration
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672/"
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "guest"
    RABBITMQ_PASSWORD: str = "guest"
    RABBITMQ_VHOST: str = "/"

    # CORS Configuration
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    # Logging Configuration
    LOG_LEVEL: LogLevel = LogLevel.INFO
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_FILE: Optional[str] = None
    LOG_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    ENABLE_REQUEST_LOGGING: bool = True
    ENABLE_PERFORMANCE_LOGGING: bool = True

    # Service URLs
    JOBS_SERVICE_URL: str = "http://localhost:8001"
    AUTH_SERVICE_URL: str = "http://localhost:8002"
    PROFILE_SERVICE_URL: str = "http://localhost:8003"
    NOTIFICATION_SERVICE_URL: str = "http://localhost:8004"

    # File Upload Configuration
    MAX_FILE_SIZE: int = 5 * 1024 * 1024  # 5MB
    ALLOWED_FILE_TYPES: List[str] = [".pdf", ".docx", ".doc", ".txt"]
    UPLOAD_TEMP_DIR: str = "/tmp/uploads"
    FILE_CLEANUP_AFTER_MINUTES: int = 60

    # ML Configuration
    ML_MODEL_CACHE_SIZE: int = 100
    ML_MODEL_CACHE_TTL: int = 3600  # 1 hour
    ML_BATCH_SIZE: int = 32
    ML_MAX_TOKENS: int = 512
    ML_SIMILARITY_THRESHOLD: float = 0.7
    ML_ENABLE_GPU: bool = False
    ML_MODEL_PATH: str = "./models"

    # Business Rules
    MAX_JOBS_PER_REQUEST: int = 50
    MAX_RESUME_SIZE_MB: int = 5
    DEFAULT_MATCH_LIMIT: int = 20
    MIN_MATCH_SCORE: float = 0.3
    SKILL_MATCH_BONUS: float = 0.1
    EXPERIENCE_MATCH_BONUS: float = 0.15

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    RATE_LIMIT_PER_DAY: int = 10000
    ENABLE_RATE_LIMITING: bool = True

    # Monitoring & Metrics
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    PROMETHEUS_NAMESPACE: str = "resume_matcher"
    ENABLE_TRACING: bool = True
    JAEGER_ENDPOINT: Optional[str] = None
    CORRELATION_ID_HEADER: str = "X-Correlation-ID"

    # Health Check Configuration
    HEALTH_CHECK_INTERVAL: int = 30  # seconds
    DATABASE_HEALTH_TIMEOUT: int = 5
    REDIS_HEALTH_TIMEOUT: int = 3
    SERVICE_HEALTH_TIMEOUT: int = 10

    # Cache Configuration
    CACHE_DEFAULT_TTL: int = 3600  # 1 hour
    CACHE_MAX_SIZE: int = 1000
    ENABLE_QUERY_CACHE: bool = True
    ENABLE_RESULT_CACHE: bool = True

    # Security Settings
    ENABLE_HTTPS: bool = False
    SSL_CERT_PATH: Optional[str] = None
    SSL_KEY_PATH: Optional[str] = None
    ENABLE_API_KEY_AUTH: bool = False
    API_KEY_HEADER: str = "X-API-Key"
    ENABLE_IP_WHITELIST: bool = False
    ALLOWED_IPS: List[str] = []

    # Feature Flags
    ENABLE_AB_TESTING: bool = True
    ENABLE_ADVANCED_ML: bool = True
    ENABLE_ANALYTICS: bool = True
    ENABLE_FEEDBACK_COLLECTION: bool = True
    ENABLE_RECOMMENDATION_ENGINE: bool = True

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], info) -> Optional[str]:
        """Build database URL from components if not provided directly."""
        if isinstance(v, str) and v:
            return v

        values = info.data
        user = values.get("POSTGRES_USER")
        password = values.get("POSTGRES_PASSWORD")
        host = values.get("POSTGRES_SERVER")
        port = values.get("POSTGRES_PORT")
        db = values.get("POSTGRES_DB")

        if not (user and host and db):
            # Not enough components to build a DSN
            return None

        # Build a postgres DSN manually compatible with pydantic v2 AnyUrl
        auth = user if password in (None, "") else f"{user}:{password}"
        # If password is empty, avoid trailing colon
        if password in (None, ""):
            auth = user

        port_part = f":{port}" if port else ""
        dsn = f"postgresql://{auth}@{host}{port_part}/{db}"
        return dsn

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v):
        """Ensure log level is valid."""
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration dictionary."""
        return {
            "url": self.DATABASE_URL,
            "pool_size": self.DATABASE_POOL_SIZE,
            "max_overflow": self.DATABASE_MAX_OVERFLOW,
            "pool_timeout": self.DATABASE_POOL_TIMEOUT,
            "echo": self.DATABASE_ECHO,
        }

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration dictionary."""
        return {
            "host": self.REDIS_HOST,
            "port": self.REDIS_PORT,
            "password": self.REDIS_PASSWORD,
            "db": self.REDIS_DB,
            "max_connections": self.REDIS_MAX_CONNECTIONS,
            "socket_timeout": self.REDIS_SOCKET_TIMEOUT,
        }

    def get_celery_config(self) -> Dict[str, Any]:
        """Get Celery configuration dictionary."""
        return {
            "broker_url": self.CELERY_BROKER_URL,
            "result_backend": self.CELERY_RESULT_BACKEND,
            "task_serializer": self.CELERY_TASK_SERIALIZER,
            "result_serializer": self.CELERY_RESULT_SERIALIZER,
            "accept_content": self.CELERY_ACCEPT_CONTENT,
            "timezone": self.CELERY_TIMEZONE,
            "enable_utc": self.CELERY_ENABLE_UTC,
            "worker_prefetch_multiplier": self.CELERY_WORKER_PREFETCH_MULTIPLIER,
        }

    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration dictionary."""
        return {
            "allow_origins": self.CORS_ORIGINS,
            "allow_credentials": self.CORS_ALLOW_CREDENTIALS,
            "allow_methods": self.CORS_ALLOW_METHODS,
            "allow_headers": self.CORS_ALLOW_HEADERS,
        }

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == Environment.DEVELOPMENT

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == Environment.PRODUCTION

    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.ENVIRONMENT == Environment.TESTING

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration dictionary."""
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": self.LOG_FORMAT,
                    "datefmt": self.LOG_DATE_FORMAT,
                },
                "simple": {
                    "format": "%(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.LOG_LEVEL.value,
                    "formatter": "detailed",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "": {  # Root logger
                    "level": self.LOG_LEVEL.value,
                    "handlers": ["console"],
                },
                "app": {
                    "level": self.LOG_LEVEL.value,
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
        }

        # Add file handler if log file is specified
        if self.LOG_FILE:
            config["handlers"]["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": self.LOG_LEVEL.value,
                "formatter": "detailed",
                "filename": self.LOG_FILE,
                "maxBytes": self.LOG_MAX_SIZE,
                "backupCount": self.LOG_BACKUP_COUNT,
            }
            config["loggers"][""]["handlers"].append("file")
            config["loggers"]["app"]["handlers"].append("file")

        return config
