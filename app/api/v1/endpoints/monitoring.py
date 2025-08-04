"""
Monitoring and observability endpoints for Phase 4.
Provides health checks, metrics, and SLA monitoring endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Response
from typing import Dict, Any, Optional
import json

from app.core.monitoring.health_monitor import get_health_monitor
from app.core.monitoring.metrics_collector import get_metrics_collector
from app.core.monitoring.business_metrics import get_business_metrics_collector
from app.core.monitoring.sla_monitor import get_sla_monitor
from app.core.monitoring.correlation_tracker import get_correlation_tracker
from app.config.config_validator import get_config, validate_current_config
from app.utils.logger import logger

router = APIRouter(
    prefix="/monitoring",
    tags=["monitoring"],
    responses={404: {"description": "Not found"}},
)


@router.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint.
    Returns overall system health and component status.
    """
    try:
        health_monitor = get_health_monitor()
        
        # Run health checks
        health_results = await health_monitor.run_health_checks()
        overall_health = health_monitor.get_overall_health()
        
        # Get system metrics if available
        system_metrics = health_monitor.get_system_metrics()
        
        response_data = {
            "status": overall_health.status.value,
            "message": overall_health.message,
            "timestamp": overall_health.timestamp.isoformat(),
            "overall_health": overall_health.to_dict(),
            "component_health": {
                name: result.to_dict() for name, result in health_results.items()
            },
            "system_metrics": system_metrics.to_dict() if system_metrics else None,
        }
        
        # Set appropriate HTTP status code
        status_code = 200
        if overall_health.status.value == "unhealthy":
            status_code = 503
        elif overall_health.status.value == "degraded":
            status_code = 200  # Still serving requests
        
        return Response(
            content=json.dumps(response_data),
            status_code=status_code,
            media_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return Response(
            content=json.dumps({
                "status": "unknown",
                "message": f"Health check failed: {str(e)}",
                "timestamp": "unknown"
            }),
            status_code=500,
            media_type="application/json"
        )


@router.get("/health/live")
async def liveness_probe():
    """
    Kubernetes liveness probe endpoint.
    Returns 200 if the application is running.
    """
    return {"status": "alive", "message": "Application is running"}


@router.get("/health/ready")
async def readiness_probe():
    """
    Kubernetes readiness probe endpoint.
    Returns 200 if the application is ready to serve requests.
    """
    try:
        health_monitor = get_health_monitor()
        overall_health = health_monitor.get_overall_health()
        
        if overall_health.status.value in ["healthy", "degraded"]:
            return {"status": "ready", "message": "Application is ready"}
        else:
            raise HTTPException(
                status_code=503,
                detail={"status": "not_ready", "message": overall_health.message}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness probe failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "message": f"Readiness check failed: {str(e)}"}
        )


@router.get("/metrics")
async def get_prometheus_metrics():
    """
    Prometheus metrics endpoint.
    Returns metrics in Prometheus format.
    """
    try:
        metrics_collector = get_metrics_collector()
        prometheus_metrics = metrics_collector.get_prometheus_metrics()
        
        return Response(
            content=prometheus_metrics,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
        
    except Exception as e:
        logger.error(f"Failed to get Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@router.get("/metrics/summary")
async def get_metrics_summary():
    """
    Get summary of collected metrics.
    """
    try:
        metrics_collector = get_metrics_collector()
        return metrics_collector.get_metrics_summary()
        
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics summary")


@router.get("/metrics/business")
async def get_business_metrics():
    """
    Get business metrics and KPIs.
    """
    try:
        business_collector = get_business_metrics_collector()
        return business_collector.export_metrics()
        
    except Exception as e:
        logger.error(f"Failed to get business metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve business metrics")


@router.get("/metrics/performance")
async def get_performance_metrics(hours: int = 24):
    """
    Get performance metrics for the specified time window.
    """
    try:
        business_collector = get_business_metrics_collector()
        return business_collector.get_performance_metrics(hours)
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance metrics")


@router.get("/sla")
async def get_sla_status():
    """
    Get SLA compliance status and reports.
    """
    try:
        sla_monitor = get_sla_monitor()
        return sla_monitor.get_sla_summary()
        
    except Exception as e:
        logger.error(f"Failed to get SLA status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve SLA status")


@router.get("/sla/reports")
async def get_sla_reports(target_name: Optional[str] = None):
    """
    Get detailed SLA compliance reports.
    """
    try:
        sla_monitor = get_sla_monitor()
        reports = sla_monitor.get_sla_report(target_name)
        
        return {
            "reports": [report.to_dict() for report in reports],
            "timestamp": reports[0].report_timestamp.isoformat() if reports else None
        }
        
    except Exception as e:
        logger.error(f"Failed to get SLA reports: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve SLA reports")


@router.get("/sla/alerts")
async def get_sla_alerts(hours: int = 24):
    """
    Get recent SLA alerts.
    """
    try:
        sla_monitor = get_sla_monitor()
        alerts = sla_monitor.get_recent_alerts(hours)
        
        return {
            "alerts": [alert.to_dict() for alert in alerts],
            "count": len(alerts),
            "time_window_hours": hours
        }
        
    except Exception as e:
        logger.error(f"Failed to get SLA alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve SLA alerts")


@router.get("/correlation/active")
async def get_active_correlations():
    """
    Get active correlation contexts.
    """
    try:
        tracker = get_correlation_tracker()
        contexts = tracker.get_active_contexts()
        
        return {
            "active_contexts": [context.to_dict() for context in contexts],
            "count": len(contexts)
        }
        
    except Exception as e:
        logger.error(f"Failed to get active correlations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve correlation data")


@router.get("/config/validate")
async def validate_configuration():
    """
    Validate current configuration and return results.
    """
    try:
        return validate_current_config()
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise HTTPException(status_code=500, detail="Configuration validation failed")


@router.get("/config/current")
async def get_current_config():
    """
    Get current configuration summary (sensitive data redacted).
    """
    try:
        config = get_config()
        
        # Return safe configuration info
        return {
            "environment": config.ENVIRONMENT.value,
            "app_name": config.APP_NAME,
            "app_version": config.APP_VERSION,
            "debug": config.DEBUG,
            "workers": config.WORKERS,
            "features": {
                "metrics_enabled": config.ENABLE_METRICS,
                "tracing_enabled": config.ENABLE_TRACING,
                "ab_testing_enabled": config.ENABLE_AB_TESTING,
                "advanced_ml_enabled": config.ENABLE_ADVANCED_ML,
                "analytics_enabled": config.ENABLE_ANALYTICS,
                "rate_limiting_enabled": config.ENABLE_RATE_LIMITING,
            },
            "api": {
                "docs_enabled": config.DOCS_URL is not None,
                "cors_origins_count": len(config.CORS_ORIGINS),
            },
            "database": {
                "pool_size": config.DATABASE_POOL_SIZE,
                "max_overflow": config.DATABASE_MAX_OVERFLOW,
            },
            "cache": {
                "default_ttl": config.CACHE_DEFAULT_TTL,
                "max_size": config.CACHE_MAX_SIZE,
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve configuration")


@router.get("/system/info")
async def get_system_info():
    """
    Get system information and diagnostics.
    """
    try:
        health_monitor = get_health_monitor()
        metrics_collector = get_metrics_collector()
        business_collector = get_business_metrics_collector()
        
        system_metrics = health_monitor.get_system_metrics()
        metrics_summary = metrics_collector.get_metrics_summary()
        
        return {
            "system_metrics": system_metrics.to_dict() if system_metrics else None,
            "metrics_summary": metrics_summary,
            "active_users_1h": business_collector.get_active_user_count(60),
            "health_check_count": health_monitor.get_context_count() if hasattr(health_monitor, 'get_context_count') else 0,
            "correlation_contexts": get_correlation_tracker().get_context_count(),
        }
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system information")


@router.post("/test/correlation")
async def test_correlation_tracking():
    """
    Test endpoint for correlation ID tracking.
    """
    from app.core.monitoring.correlation_tracker import get_current_correlation_id, generate_correlation_id
    
    current_id = get_current_correlation_id()
    new_id = generate_correlation_id()
    
    return {
        "current_correlation_id": current_id,
        "generated_correlation_id": new_id,
        "message": "Correlation tracking test completed"
    }