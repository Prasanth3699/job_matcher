"""
Configuration validation and factory module.
Ensures configuration integrity and provides environment-specific config instances.
"""

import os
import sys
from typing import Type, Dict, Any, List, Optional
from functools import lru_cache
from pathlib import Path

from .base_config import BaseConfig, Environment
from .environments.development import DevelopmentConfig
from .environments.production import ProductionConfig
from .environments.staging import StagingConfig
from .environments.testing import TestingConfig


class ConfigurationError(Exception):
    """Configuration-related error."""
    pass


class ConfigValidator:
    """Configuration validation utility."""
    
    @staticmethod
    def validate_required_env_vars(config: BaseConfig) -> List[str]:
        """
        Validate that required environment variables are set.
        Returns list of missing variables.
        """
        missing_vars = []
        
        # Check database configuration (allow empty for testing environment)
        if not config.POSTGRES_SERVER and config.ENVIRONMENT.value != "testing":
            missing_vars.append("POSTGRES_SERVER")
        if not config.POSTGRES_USER and config.ENVIRONMENT.value != "testing":
            missing_vars.append("POSTGRES_USER")
        if not config.POSTGRES_PASSWORD and config.ENVIRONMENT.value != "testing":
            missing_vars.append("POSTGRES_PASSWORD")
        if not config.POSTGRES_DB and config.ENVIRONMENT.value != "testing":
            missing_vars.append("POSTGRES_DB")
        
        # Check authentication
        if not config.SECRET_KEY or config.SECRET_KEY == "your-secret-key":
            missing_vars.append("SECRET_KEY")
        
        # Check Redis (if not using defaults)
        if config.REDIS_PASSWORD and "${" in config.REDIS_PASSWORD:
            missing_vars.append("REDIS_PASSWORD")
        
        # Production-specific validations
        if config.ENVIRONMENT == Environment.PRODUCTION:
            if config.ENABLE_HTTPS and not config.SSL_CERT_PATH:
                missing_vars.append("SSL_CERT_PATH")
            if config.ENABLE_HTTPS and not config.SSL_KEY_PATH:
                missing_vars.append("SSL_KEY_PATH")
        
        return missing_vars
    
    @staticmethod
    def validate_file_paths(config: BaseConfig) -> List[str]:
        """
        Validate that required file paths exist.
        Returns list of missing files.
        """
        missing_files = []
        
        if config.SSL_CERT_PATH and not Path(config.SSL_CERT_PATH).exists():
            missing_files.append(config.SSL_CERT_PATH)
        
        if config.SSL_KEY_PATH and not Path(config.SSL_KEY_PATH).exists():
            missing_files.append(config.SSL_KEY_PATH)
        
        if config.LOG_FILE:
            log_dir = Path(config.LOG_FILE).parent
            if not log_dir.exists():
                try:
                    log_dir.mkdir(parents=True, exist_ok=True)
                except OSError:
                    missing_files.append(str(log_dir))
        
        # Create upload directory if it doesn't exist
        upload_dir = Path(config.UPLOAD_TEMP_DIR)
        if not upload_dir.exists():
            try:
                upload_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                missing_files.append(str(upload_dir))
        
        return missing_files
    
    @staticmethod
    def validate_network_settings(config: BaseConfig) -> List[str]:
        """
        Validate network-related settings.
        Returns list of validation errors.
        """
        errors = []
        
        # Port validation
        if not (1 <= config.PORT <= 65535):
            errors.append(f"Invalid port number: {config.PORT}")
        
        if not (1 <= config.POSTGRES_PORT <= 65535):
            errors.append(f"Invalid PostgreSQL port: {config.POSTGRES_PORT}")
        
        if not (1 <= config.REDIS_PORT <= 65535):
            errors.append(f"Invalid Redis port: {config.REDIS_PORT}")
        
        # URL validation
        required_urls = [
            ("JOBS_SERVICE_URL", config.JOBS_SERVICE_URL),
            ("AUTH_SERVICE_URL", config.AUTH_SERVICE_URL),
            ("PROFILE_SERVICE_URL", config.PROFILE_SERVICE_URL),
        ]
        
        for name, url in required_urls:
            if not url.startswith(("http://", "https://")):
                errors.append(f"Invalid URL format for {name}: {url}")
        
        return errors
    
    @staticmethod
    def validate_business_rules(config: BaseConfig) -> List[str]:
        """
        Validate business rule settings.
        Returns list of validation errors.
        """
        errors = []
        
        # File size limits
        if config.MAX_FILE_SIZE <= 0:
            errors.append("MAX_FILE_SIZE must be positive")
        
        if config.MAX_RESUME_SIZE_MB <= 0:
            errors.append("MAX_RESUME_SIZE_MB must be positive")
        
        # Score thresholds
        if not (0.0 <= config.MIN_MATCH_SCORE <= 1.0):
            errors.append("MIN_MATCH_SCORE must be between 0.0 and 1.0")
        
        if not (0.0 <= config.ML_SIMILARITY_THRESHOLD <= 1.0):
            errors.append("ML_SIMILARITY_THRESHOLD must be between 0.0 and 1.0")
        
        # Rate limiting
        if config.ENABLE_RATE_LIMITING:
            if config.RATE_LIMIT_PER_MINUTE <= 0:
                errors.append("RATE_LIMIT_PER_MINUTE must be positive when rate limiting is enabled")
        
        # Token expiration
        if config.ACCESS_TOKEN_EXPIRE_MINUTES <= 0:
            errors.append("ACCESS_TOKEN_EXPIRE_MINUTES must be positive")
        
        return errors
    
    @staticmethod
    def validate_security_settings(config: BaseConfig) -> List[str]:
        """
        Validate security-related settings.
        Returns list of validation warnings/errors.
        """
        warnings = []
        
        # Production security checks
        if config.ENVIRONMENT == Environment.PRODUCTION:
            if not config.ENABLE_HTTPS:
                warnings.append("HTTPS should be enabled in production")
            
            if config.DEBUG:
                warnings.append("DEBUG should be disabled in production")
            
            if config.DOCS_URL or config.REDOC_URL:
                warnings.append("API documentation should be disabled in production")
            
            if config.ACCESS_TOKEN_EXPIRE_MINUTES > 60:
                warnings.append("Token expiration time is very long for production")
        
        # Password policy
        if config.PASSWORD_MIN_LENGTH < 8:
            warnings.append("Password minimum length should be at least 8 characters")
        
        # Secret key strength
        if len(config.SECRET_KEY) < 32:
            warnings.append("SECRET_KEY should be at least 32 characters long")
        
        return warnings
    
    @classmethod
    def validate_config(cls, config: BaseConfig) -> Dict[str, List[str]]:
        """
        Perform comprehensive configuration validation.
        Returns dictionary with validation results.
        """
        results = {
            "missing_env_vars": cls.validate_required_env_vars(config),
            "missing_files": cls.validate_file_paths(config),
            "network_errors": cls.validate_network_settings(config),
            "business_rule_errors": cls.validate_business_rules(config),
            "security_warnings": cls.validate_security_settings(config),
        }
        
        return results


class ConfigFactory:
    """Factory for creating environment-specific configurations."""
    
    _config_map: Dict[Environment, Type[BaseConfig]] = {
        Environment.DEVELOPMENT: DevelopmentConfig,
        Environment.PRODUCTION: ProductionConfig,
        Environment.STAGING: StagingConfig,
        Environment.TESTING: TestingConfig,
    }
    
    @classmethod
    def get_environment(cls) -> Environment:
        """
        Determine the current environment from environment variable.
        Defaults to development if not specified.
        """
        env_name = os.getenv("ENVIRONMENT", "development").lower()
        
        try:
            return Environment(env_name)
        except ValueError:
            print(f"Warning: Unknown environment '{env_name}', defaulting to development")
            return Environment.DEVELOPMENT
    
    @classmethod
    def create_config(cls, environment: Optional[Environment] = None) -> BaseConfig:
        """
        Create configuration instance for the specified environment.
        If no environment is specified, detects from environment variable.
        """
        if environment is None:
            environment = cls.get_environment()
        
        config_class = cls._config_map.get(environment)
        if not config_class:
            raise ConfigurationError(f"No configuration found for environment: {environment}")
        
        try:
            config = config_class()
        except Exception as e:
            raise ConfigurationError(f"Failed to create {environment} configuration: {e}")
        
        # Validate configuration
        validation_results = ConfigValidator.validate_config(config)
        
        # Handle missing environment variables
        missing_vars = validation_results["missing_env_vars"]
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            if environment == Environment.PRODUCTION:
                raise ConfigurationError(error_msg)
            else:
                print(f"Warning: {error_msg}")
        
        # Handle missing files
        missing_files = validation_results["missing_files"]
        if missing_files:
            error_msg = f"Missing required files: {', '.join(missing_files)}"
            if environment == Environment.PRODUCTION:
                raise ConfigurationError(error_msg)
            else:
                print(f"Warning: {error_msg}")
        
        # Handle network errors
        network_errors = validation_results["network_errors"]
        if network_errors:
            error_msg = f"Network configuration errors: {', '.join(network_errors)}"
            raise ConfigurationError(error_msg)
        
        # Handle business rule errors
        business_errors = validation_results["business_rule_errors"]
        if business_errors:
            error_msg = f"Business rule configuration errors: {', '.join(business_errors)}"
            raise ConfigurationError(error_msg)
        
        # Handle security warnings
        security_warnings = validation_results["security_warnings"]
        if security_warnings:
            for warning in security_warnings:
                print(f"Security Warning: {warning}")
        
        return config
    
    @classmethod
    def get_available_environments(cls) -> List[Environment]:
        """Get list of available environments."""
        return list(cls._config_map.keys())


@lru_cache()
def get_config() -> BaseConfig:
    """
    Get cached configuration instance for the current environment.
    This is the primary function used throughout the application.
    """
    return ConfigFactory.create_config()


def reload_config() -> BaseConfig:
    """
    Force reload of configuration (clears cache).
    Useful for testing or when environment variables change.
    """
    get_config.cache_clear()
    return get_config()


def validate_current_config() -> Dict[str, Any]:
    """
    Validate the current configuration and return detailed results.
    Useful for health checks and debugging.
    """
    config = get_config()
    validation_results = ConfigValidator.validate_config(config)
    
    return {
        "environment": config.ENVIRONMENT.value,
        "app_name": config.APP_NAME,
        "app_version": config.APP_VERSION,
        "validation_results": validation_results,
        "config_summary": {
            "debug": config.DEBUG,
            "workers": config.WORKERS,
            "database_url": config.DATABASE_URL,
            "redis_host": config.REDIS_HOST,
            "enable_metrics": config.ENABLE_METRICS,
            "enable_tracing": config.ENABLE_TRACING,
        }
    }