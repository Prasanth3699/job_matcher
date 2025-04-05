from datetime import datetime, timezone
from pydantic import BaseModel
from fastapi import APIRouter
from fastapi.responses import Response
import prometheus_client
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import REGISTRY

# Create a router for core endpoints
router = APIRouter(
    tags=["core"],
    responses={404: {"description": "Not found"}},
)


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for API monitoring"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=prometheus_client.generate_latest(REGISTRY), media_type="text/plain"
    )


# Create a function to set up instrumentation that can be called from main.py
def setup_instrumentation(app):
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        excluded_handlers=["/metrics", "/docs", "/openapi.json", "/health"],
        env_var_name="ENABLE_METRICS",
    )
    instrumentator.instrument(app).expose(app)
    return instrumentator
