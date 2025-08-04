from fastapi import APIRouter
from app.api.v1.endpoints import core, matching, health, enhanced_ml, security

api_router = APIRouter()

# Include routers from different modules with appropriate prefixes
api_router.include_router(health.router)  # Health endpoints at root level
api_router.include_router(core.router, prefix="/core")
api_router.include_router(matching.router, prefix="/matching")
api_router.include_router(enhanced_ml.router, prefix="/enhanced-ml", tags=["enhanced-ml"])
api_router.include_router(security.router, prefix="/security", tags=["security"])
