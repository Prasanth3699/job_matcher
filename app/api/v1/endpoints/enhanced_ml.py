"""
Enhanced ML Pipeline API endpoints for advanced features.

This module provides API endpoints for accessing Enhanced ML Pipeline components 
including analytics, A/B testing, model monitoring, and dashboard services.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Path, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.utils.logger import logger
from app.core.enhanced_ml_config import get_enhanced_ml_manager, EnhancedMLMode
from app.core.ab_testing.experiment_manager import ExperimentManager


router = APIRouter()


class EnhancedMLStatusResponse(BaseModel):
    """Response model for Enhanced ML Pipeline status."""
    status: str
    health_score: float
    mode: str
    components: Dict[str, str]
    timestamp: str


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    overall_status: str
    health_score: float
    components: Dict[str, Any]
    mode: str
    uptime_seconds: float
    timestamp: str


class AnalyticsEventRequest(BaseModel):
    """Request model for analytics events."""
    event_type: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    data: Dict[str, Any] = {}


class AnalyticsEventResponse(BaseModel):
    """Response model for analytics events."""
    event_id: str
    timestamp: str
    status: str


class DashboardDataRequest(BaseModel):
    """Request model for dashboard data."""
    dashboard_type: str
    time_range_hours: int = 24
    refresh: bool = False


class ExperimentRequest(BaseModel):
    """Request model for creating experiments."""
    experiment_name: str
    description: str
    variant_configs: Dict[str, Any]
    traffic_allocation: Dict[str, float]
    duration_days: int = 14


@router.get("/status", response_model=EnhancedMLStatusResponse)
async def get_enhanced_ml_status():
    """Get comprehensive Enhanced ML Pipeline system status."""
    try:
        manager = get_enhanced_ml_manager()
        if not manager:
            raise HTTPException(
                status_code=503,
                detail="Enhanced ML Pipeline components not initialized"
            )
        
        status_data = await manager.get_status()
        
        if 'error' in status_data:
            raise HTTPException(
                status_code=500,
                detail=f"Enhanced ML Pipeline status error: {status_data['error']}"
            )
        
        enhanced_ml_status = status_data['enhanced_ml_status']
        
        return EnhancedMLStatusResponse(
            status=enhanced_ml_status['components_status'].get('unified_service', 'unknown'),
            health_score=enhanced_ml_status['health_score'],
            mode=enhanced_ml_status['mode'],
            components=enhanced_ml_status['components_status'],
            timestamp=enhanced_ml_status['last_health_check']
        )
        
    except Exception as e:
        logger.error(f"Enhanced ML Pipeline status endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get Enhanced ML Pipeline status: {str(e)}"
        )


@router.get("/health", response_model=HealthCheckResponse)
async def get_health_check():
    """Perform comprehensive health check of Enhanced ML Pipeline components."""
    try:
        manager = get_enhanced_ml_manager()
        if not manager:
            raise HTTPException(
                status_code=503,
                detail="Enhanced ML Pipeline components not initialized"
            )
        
        health_data = await manager.health_check()
        
        return HealthCheckResponse(
            overall_status=health_data['overall_status'],
            health_score=health_data['health_score'],
            components=health_data['components'],
            mode=health_data['mode'],
            uptime_seconds=health_data['uptime_seconds'],
            timestamp=health_data['timestamp']
        )
        
    except Exception as e:
        logger.error(f"Enhanced ML Pipeline health check endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/analytics/summary")
async def get_analytics_summary(
    hours: int = Query(24, ge=1, le=168, description="Time range in hours")
):
    """Get analytics summary for specified time range."""
    try:
        manager = get_enhanced_ml_manager()
        if not manager:
            raise HTTPException(
                status_code=503,
                detail="Enhanced ML Pipeline components not initialized"
            )
        
        analytics_engine = manager.get_analytics_engine()
        if not analytics_engine:
            raise HTTPException(
                status_code=503,
                detail="Analytics engine not available"
            )
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        summary = await analytics_engine.get_analytics_summary(
            start_time=start_time,
            end_time=end_time
        )
        
        return {
            "summary": summary,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Analytics summary endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get analytics summary: {str(e)}"
        )


@router.post("/analytics/events", response_model=AnalyticsEventResponse)
async def track_analytics_event(request: AnalyticsEventRequest):
    """Track an analytics event."""
    try:
        manager = get_enhanced_ml_manager()
        if not manager:
            raise HTTPException(
                status_code=503,
                detail="Enhanced ML Pipeline components not initialized"
            )
        
        analytics_engine = manager.get_analytics_engine()
        if not analytics_engine:
            raise HTTPException(
                status_code=503,
                detail="Analytics engine not available"
            )
        
        event_id = await analytics_engine.track_event(
            event_type=request.event_type,
            user_id=request.user_id,
            session_id=request.session_id,
            data=request.data
        )
        
        return AnalyticsEventResponse(
            event_id=event_id,
            timestamp=datetime.now().isoformat(),
            status="tracked"
        )
        
    except Exception as e:
        logger.error(f"Analytics event tracking error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to track event: {str(e)}"
        )


@router.get("/model-monitor/performance")
async def get_model_performance(
    model_id: Optional[str] = Query(None, description="Specific model ID"),
    hours: int = Query(24, ge=1, le=168, description="Time range in hours")
):
    """Get model performance metrics."""
    try:
        manager = get_enhanced_ml_manager()
        if not manager:
            raise HTTPException(
                status_code=503,
                detail="Enhanced ML Pipeline components not initialized"
            )
        
        model_monitor = manager.get_model_monitor()
        if not model_monitor:
            raise HTTPException(
                status_code=503,
                detail="Model monitor not available"
            )
        
        # Get performance metrics
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        performance_data = await model_monitor.get_performance_summary(
            model_id=model_id,
            start_time=start_time,
            end_time=end_time
        )
        
        return {
            "performance_data": performance_data,
            "model_id": model_id,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model performance endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model performance: {str(e)}"
        )


@router.get("/business-metrics/summary")
async def get_business_metrics_summary(
    hours: int = Query(24, ge=1, le=168, description="Time range in hours")
):
    """Get business metrics summary."""
    try:
        manager = get_enhanced_ml_manager()
        if not manager:
            raise HTTPException(
                status_code=503,
                detail="Enhanced ML Pipeline components not initialized"
            )
        
        business_metrics = manager.get_business_metrics()
        if not business_metrics:
            raise HTTPException(
                status_code=503,
                detail="Business metrics collector not available"
            )
        
        # Get business metrics
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        metrics_data = await business_metrics.get_business_summary(
            start_time=start_time,
            end_time=end_time
        )
        
        return {
            "metrics": metrics_data,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Business metrics endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get business metrics: {str(e)}"
        )


@router.get("/dashboard/{dashboard_type}")
async def get_dashboard_data(
    dashboard_type: str = Path(..., description="Dashboard type"),
    hours: int = Query(24, ge=1, le=168, description="Time range in hours"),
    refresh: bool = Query(False, description="Force refresh data")
):
    """Get dashboard data for specified dashboard type."""
    try:
        manager = get_enhanced_ml_manager()
        if not manager:
            raise HTTPException(
                status_code=503,
                detail="Enhanced ML Pipeline components not initialized"
            )
        
        dashboard_service = manager.get_dashboard_service()
        if not dashboard_service:
            raise HTTPException(
                status_code=503,
                detail="Dashboard service not available"
            )
        
        # Get dashboard data
        dashboard_data = await dashboard_service.get_dashboard_data(
            dashboard_type=dashboard_type,
            time_range_hours=hours,
            force_refresh=refresh
        )
        
        return {
            "dashboard_data": dashboard_data,
            "dashboard_type": dashboard_type,
            "time_range_hours": hours,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Dashboard data endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get dashboard data: {str(e)}"
        )


@router.get("/experiments")
async def get_experiments():
    """Get list of active experiments."""
    try:
        manager = get_enhanced_ml_manager()
        if not manager:
            raise HTTPException(
                status_code=503,
                detail="Enhanced ML Pipeline components not initialized"
            )
        
        experiment_manager = manager.get_experiment_manager()
        if not experiment_manager:
            raise HTTPException(
                status_code=503,
                detail="Experiment manager not available"
            )
        
        experiments = await experiment_manager.get_active_experiments()
        
        return {
            "experiments": experiments,
            "count": len(experiments),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Experiments endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get experiments: {str(e)}"
        )


@router.post("/experiments")
async def create_experiment(request: ExperimentRequest):
    """Create a new A/B test experiment."""
    try:
        manager = get_enhanced_ml_manager()
        if not manager:
            raise HTTPException(
                status_code=503,
                detail="Enhanced ML Pipeline components not initialized"
            )
        
        experiment_manager = manager.get_experiment_manager()
        if not experiment_manager:
            raise HTTPException(
                status_code=503,
                detail="Experiment manager not available"
            )
        
        experiment_id = await experiment_manager.create_experiment(
            name=request.experiment_name,
            description=request.description,
            variant_configs=request.variant_configs,
            traffic_allocation=request.traffic_allocation,
            duration_days=request.duration_days
        )
        
        return {
            "experiment_id": experiment_id,
            "experiment_name": request.experiment_name,
            "status": "created",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Create experiment endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create experiment: {str(e)}"
        )


@router.get("/experiments/{experiment_id}/results")
async def get_experiment_results(
    experiment_id: str = Path(..., description="Experiment ID")
):
    """Get experiment results and statistical analysis."""
    try:
        manager = get_enhanced_ml_manager()
        if not manager:
            raise HTTPException(
                status_code=503,
                detail="Enhanced ML Pipeline components not initialized"
            )
        
        experiment_manager = manager.get_experiment_manager()
        if not experiment_manager:
            raise HTTPException(
                status_code=503,
                detail="Experiment manager not available"
            )
        
        results = await experiment_manager.get_experiment_results(experiment_id)
        
        return {
            "experiment_id": experiment_id,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Experiment results endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get experiment results: {str(e)}"
        )


@router.post("/components/{component_name}/restart")
async def restart_component(
    component_name: str = Path(..., description="Component name to restart")
):
    """Restart a specific Enhanced ML Pipeline component."""
    try:
        manager = get_enhanced_ml_manager()
        if not manager:
            raise HTTPException(
                status_code=503,
                detail="Enhanced ML Pipeline components not initialized"
            )
        
        success = await manager.restart_component(component_name)
        
        if success:
            return {
                "component_name": component_name,
                "status": "restarted",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to restart component: {component_name}"
            )
        
    except Exception as e:
        logger.error(f"Component restart endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Component restart failed: {str(e)}"
        )