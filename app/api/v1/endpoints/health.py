from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.circuit_breaker import circuit_registry
from app.core.cache.redis_cache import cache
from app.db.session import engine
from app.services.rabbitmq_client import get_rabbitmq_client
from app.utils.logger import logger

settings = get_settings()

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    services: Dict[str, Any]
    circuit_breakers: Dict[str, Any]


class ServiceStatus(BaseModel):
    status: str
    message: str
    response_time_ms: float = None


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    start_time = datetime.now(timezone.utc)
    
    services = {}
    
    # Check Database
    services["database"] = await _check_database()
    
    # Check Redis
    services["redis"] = await _check_redis()
    
    # Check RabbitMQ
    services["rabbitmq"] = await _check_rabbitmq()
    
    # Get circuit breaker states
    circuit_breakers = circuit_registry.get_all_states()
    
    # Determine overall status
    all_healthy = all(
        service["status"] == "healthy" 
        for service in services.values()
    )
    
    overall_status = "healthy" if all_healthy else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        timestamp=start_time.isoformat(),
        services=services,
        circuit_breakers=circuit_breakers
    )


@router.get("/health/live")
async def liveness_probe():
    """Kubernetes liveness probe"""
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/health/ready")
async def readiness_probe():
    """Kubernetes readiness probe"""
    services = {
        "database": await _check_database(),
        "redis": await _check_redis(),
    }
    
    all_ready = all(
        service["status"] == "healthy" 
        for service in services.values()
    )
    
    if all_ready:
        return {"status": "ready", "timestamp": datetime.now(timezone.utc).isoformat()}
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Service not ready")


async def _check_database() -> ServiceStatus:
    """Check database connectivity"""
    try:
        start_time = datetime.now()
        
        # Simple connection test
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds() * 1000
        
        return ServiceStatus(
            status="healthy",
            message="Database connection successful",
            response_time_ms=response_time
        )
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return ServiceStatus(
            status="unhealthy",
            message=f"Database connection failed: {str(e)}"
        )


async def _check_redis() -> ServiceStatus:
    """Check Redis connectivity"""
    try:
        if not cache.redis_client:
            return ServiceStatus(
                status="unhealthy",
                message="Redis client not initialized"
            )
        
        start_time = datetime.now()
        
        # Simple ping test
        cache.redis_client.ping()
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds() * 1000
        
        return ServiceStatus(
            status="healthy",
            message="Redis connection successful",
            response_time_ms=response_time
        )
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return ServiceStatus(
            status="unhealthy",
            message=f"Redis connection failed: {str(e)}"
        )


async def _check_rabbitmq() -> ServiceStatus:
    """Check RabbitMQ connectivity"""
    try:
        start_time = datetime.now()
        
        # Get RabbitMQ client and perform health check
        client = await get_rabbitmq_client()
        is_healthy = await client.health_check()
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds() * 1000
        
        if is_healthy:
            return ServiceStatus(
                status="healthy",
                message="RabbitMQ connection active",
                response_time_ms=response_time
            )
        else:
            return ServiceStatus(
                status="unhealthy",
                message="RabbitMQ connection test failed"
            )
            
    except Exception as e:
        logger.error(f"RabbitMQ health check failed: {e}")
        return ServiceStatus(
            status="unhealthy",
            message=f"RabbitMQ connection failed: {str(e)}"
        )
