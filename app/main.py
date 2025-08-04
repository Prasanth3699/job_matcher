from contextlib import asynccontextmanager
import logging.config

from fastapi import FastAPI, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials

from app.db.session import engine
from app.db.base import Base
from app.api.router import api_router
from app.api.v1.endpoints.core import setup_instrumentation
from app.core.enhanced_ml_config import (
    initialize_enhanced_ml,
    shutdown_enhanced_ml,
    EnhancedMLMode,
)

from app.config.config_validator import get_config
from app.core.monitoring.correlation_tracker import CorrelationMiddleware, get_correlation_tracker
from app.core.monitoring.metrics_collector import initialize_metrics
from app.core.monitoring.business_metrics import initialize_business_metrics
from app.core.monitoring.health_monitor import initialize_health_monitor
from app.core.monitoring.sla_monitor import initialize_sla_monitor
from app.core.security.input_sanitizer import get_input_sanitizer
from app.core.security.rate_limiter import get_rate_limiter, RateLimitRule, RateLimitStrategy
from app.core.security.security_middleware import SecurityMiddleware
from app.core.security.audit_logger import get_audit_logger
from app.utils.logger import logger

config = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    
    # Application startup
    try:
        logging_config = config.get_logging_config()
        logging.config.dictConfig(logging_config)
        logger.info("Configured logging for environment", environment=config.ENVIRONMENT.value)
        
        logger.info("Initializing monitoring systems")
        
        app.state.metrics_collector = initialize_metrics(
            namespace=config.PROMETHEUS_NAMESPACE,
            enable_prometheus=config.ENABLE_METRICS
        )
        logger.info("Metrics collector initialized", prometheus_enabled=config.ENABLE_METRICS)
        
        app.state.business_metrics = initialize_business_metrics(retention_days=30)
        logger.info("Business metrics collector initialized")
        
        app.state.health_monitor = initialize_health_monitor(
            check_interval=config.HEALTH_CHECK_INTERVAL,
            start_monitoring=True
        )
        logger.info("Health monitor initialized", check_interval=config.HEALTH_CHECK_INTERVAL)
        
        app.state.sla_monitor = initialize_sla_monitor()
        logger.info("SLA monitor initialized")
        
        app.state.input_sanitizer = get_input_sanitizer()
        logger.info("Security input sanitizer initialized")
        
        try:
            import redis.asyncio as redis
            redis_client = redis.from_url(config.REDIS_URL)
            rate_limiter = get_rate_limiter(redis_client=redis_client)
            logger.info("Rate limiter initialized with Redis backend")
        except Exception as e:
            logger.warning("Redis not available for rate limiting, using memory backend", error=str(e))
            rate_limiter = get_rate_limiter()
            logger.info("Rate limiter initialized with memory backend")
        
        rate_limiter.add_rule("auth_endpoints", RateLimitRule(
            requests=5,
            window=300,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            block_duration=900
        ))
        
        rate_limiter.add_rule("api_general", RateLimitRule(
            requests=100,
            window=3600,
            strategy=RateLimitStrategy.TOKEN_BUCKET
        ))
        
        rate_limiter.add_rule("file_uploads", RateLimitRule(
            requests=10,
            window=600,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            block_duration=1800
        ))
        
        rate_limiter.add_rule("matching_requests", RateLimitRule(
            requests=20,
            window=3600,
            strategy=RateLimitStrategy.TOKEN_BUCKET
        ))
        
        audit_logger = get_audit_logger()
        await audit_logger.start()
        logger.info("Security audit logger initialized")
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created/verified")

        from app.services.rabbitmq_client import get_rabbitmq_client

        try:
            await get_rabbitmq_client()
            logger.info("RabbitMQ client initialized successfully")
        except Exception as e:
            logger.warning("RabbitMQ client initialization failed", error=str(e))

        try:
            enhanced_ml_mode = (
                EnhancedMLMode.DEVELOPMENT
                if config.DEBUG
                else EnhancedMLMode.FULL_PRODUCTION
            )
            enhanced_ml_success = await initialize_enhanced_ml(mode=enhanced_ml_mode)
            if enhanced_ml_success:
                logger.info("Enhanced ML Pipeline components initialized successfully")
            else:
                logger.warning("Enhanced ML Pipeline components initialization failed")
        except Exception as e:
            logger.warning("Enhanced ML Pipeline initialization error", error=str(e))
        
        logger.info("Application started successfully", 
                   app_name=config.APP_NAME, 
                   version=config.APP_VERSION, 
                   environment=config.ENVIRONMENT.value)

        yield

        # Application shutdown
        logger.info("Shutting down application")
        
        try:
            app.state.health_monitor.stop_monitoring()
            logger.info("Health monitoring stopped")
        except Exception as e:
            logger.warning("Health monitor shutdown error", error=str(e))
        
        try:
            await shutdown_enhanced_ml()
            logger.info("Enhanced ML Pipeline components shutdown completed")
        except Exception as e:
            logger.warning("Enhanced ML Pipeline shutdown error", error=str(e))
        
        try:
            audit_logger = get_audit_logger()
            await audit_logger.stop()
            logger.info("Security audit logger stopped")
        except Exception as e:
            logger.warning("Audit logger shutdown error", error=str(e))

        from app.services.rabbitmq_client import cleanup_rabbitmq_client

        await cleanup_rabbitmq_client()
        engine.dispose()
        logger.info("Application shutdown completed")
        
    except Exception as e:
        logger.error("Application startup/shutdown error", error=str(e))
        raise


app = FastAPI(
    title=config.APP_NAME,
    description=config.APP_DESCRIPTION,
    version=config.APP_VERSION,
    debug=config.DEBUG,
    docs_url=config.DOCS_URL,
    redoc_url=config.REDOC_URL,
    openapi_url=config.OPENAPI_URL,
    lifespan=lifespan,
)

correlation_tracker = get_correlation_tracker()
app.add_middleware(CorrelationMiddleware, tracker=correlation_tracker)

security_config = {
    "ip_whitelist": None,
    "max_request_size": 10 * 1024 * 1024,
    "threat_threshold": 5.0,
    "block_duration": 300,
    "exclude_paths": ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"],
    "security_headers": {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
        "Server": f"{config.APP_NAME}/1.0"
    }
}
app.add_middleware(SecurityMiddleware, **security_config)

cors_config = config.get_cors_config()
app.add_middleware(CORSMiddleware, **cors_config)

app.include_router(api_router, prefix=config.API_V1_STR)

from app.api.v1.endpoints.monitoring import router as monitoring_router
app.include_router(monitoring_router, prefix=config.API_V1_STR)

setup_instrumentation(app)
