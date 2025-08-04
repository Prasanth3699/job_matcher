"""
Security management and monitoring endpoints.
Provides API access to security features and statistics.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.security.rate_limiter import get_rate_limiter, RateLimitRule, RateLimitStrategy
from app.core.security.audit_logger import get_audit_logger, SecurityEventType, SeverityLevel
from app.core.security.input_sanitizer import get_input_sanitizer
from app.utils.logger import logger

router = APIRouter()

class RateLimitRuleRequest(BaseModel):
    """Request model for creating rate limit rules."""
    identifier: str = Field(..., description="Rule identifier")
    requests: int = Field(..., ge=1, description="Number of requests allowed")
    window: int = Field(..., ge=1, description="Time window in seconds")
    strategy: RateLimitStrategy = Field(RateLimitStrategy.SLIDING_WINDOW, description="Rate limiting strategy")
    block_duration: int = Field(300, ge=0, description="Block duration in seconds")

class SecurityEventQuery(BaseModel):
    """Query parameters for security events."""
    start_time: Optional[datetime] = Field(None, description="Start time for query")
    end_time: Optional[datetime] = Field(None, description="End time for query")
    event_types: Optional[List[SecurityEventType]] = Field(None, description="Event types to filter")
    client_ip: Optional[str] = Field(None, description="Client IP to filter")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of events to return")

class InputSanitizeRequest(BaseModel):
    """Request model for input sanitization testing."""
    input_text: str = Field(..., description="Text to sanitize")
    strict: bool = Field(False, description="Use strict sanitization")

@router.get("/rate-limits/stats", summary="Get rate limiting statistics")
async def get_rate_limit_stats(client_ip: Optional[str] = None):
    """Get rate limiting statistics for all clients or a specific client."""
    try:
        rate_limiter = get_rate_limiter()
        
        if client_ip:
            # Get stats for specific client
            stats = await rate_limiter.get_client_stats(client_ip)
            return {
                "client_ip": client_ip,
                "stats": stats
            }
        else:
            # Get stats for all clients (from local state)
            all_stats = {}
            if hasattr(rate_limiter, 'local_state'):
                for client_id, state in rate_limiter.local_state.items():
                    all_stats[client_id] = {
                        "total_requests": state.total_requests,
                        "violations": state.violations,
                        "blocked_until": state.blocked_until,
                        "current_tokens": getattr(state, 'tokens', None),
                        "active_requests": len(state.request_times)
                    }
            
            return {
                "total_clients": len(all_stats),
                "clients": all_stats
            }
            
    except Exception as e:
        logger.error(f"Error getting rate limit stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve rate limit statistics"
        )

@router.post("/rate-limits/rules", summary="Add rate limiting rule")
async def add_rate_limit_rule(rule_request: RateLimitRuleRequest):
    """Add a new rate limiting rule."""
    try:
        rate_limiter = get_rate_limiter()
        
        rule = RateLimitRule(
            requests=rule_request.requests,
            window=rule_request.window,
            strategy=rule_request.strategy,
            block_duration=rule_request.block_duration
        )
        
        rate_limiter.add_rule(rule_request.identifier, rule)
        
        return {
            "message": f"Rate limit rule added for identifier: {rule_request.identifier}",
            "rule": {
                "identifier": rule_request.identifier,
                "requests": rule.requests,
                "window": rule.window,
                "strategy": rule.strategy.value,
                "block_duration": rule.block_duration
            }
        }
        
    except Exception as e:
        logger.error(f"Error adding rate limit rule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add rate limit rule"
        )

@router.delete("/rate-limits/clients/{client_id}", summary="Reset client rate limit")
async def reset_client_rate_limit(client_id: str):
    """Reset rate limiting state for a specific client."""
    try:
        rate_limiter = get_rate_limiter()
        await rate_limiter.reset_client(client_id)
        
        return {
            "message": f"Rate limit reset for client: {client_id}"
        }
        
    except Exception as e:
        logger.error(f"Error resetting client rate limit: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset client rate limit"
        )

@router.get("/audit/events", summary="Get security audit events")
async def get_audit_events(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    event_types: Optional[str] = None,  # Comma-separated event types
    client_ip: Optional[str] = None,
    limit: int = 100
):
    """Retrieve security audit events with optional filtering."""
    try:
        audit_logger = get_audit_logger()
        
        # Parse event types if provided
        parsed_event_types = None
        if event_types:
            try:
                parsed_event_types = [
                    SecurityEventType(event_type.strip()) 
                    for event_type in event_types.split(',')
                ]
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid event type: {e}"
                )
        
        events = await audit_logger.get_events(
            start_time=start_time,
            end_time=end_time,
            event_types=parsed_event_types,
            client_ip=client_ip,
            limit=min(limit, 1000)  # Cap at 1000
        )
        
        # Convert events to serializable format
        serialized_events = []
        for event in events:
            serialized_events.append({
                "event_type": event.event_type.value,
                "severity": event.severity.name,
                "timestamp": event.timestamp.isoformat(),
                "client_ip": event.client_ip,
                "user_id": event.user_id,
                "endpoint": event.endpoint,
                "method": event.method,
                "success": event.success,
                "threat_score": event.threat_score,
                "details": event.details
            })
        
        return {
            "total_events": len(serialized_events),
            "events": serialized_events,
            "query_params": {
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
                "event_types": event_types,
                "client_ip": client_ip,
                "limit": limit
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving audit events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit events"
        )

@router.get("/audit/threat-analysis", summary="Get threat analysis")
async def get_threat_analysis(hours: int = 24):
    """Get comprehensive threat analysis for the specified time window."""
    try:
        if hours <= 0 or hours > 168:  # Max 1 week
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Hours must be between 1 and 168 (1 week)"
            )
        
        audit_logger = get_audit_logger()
        time_window = timedelta(hours=hours)
        
        analysis = await audit_logger.get_threat_analysis(time_window)
        
        return {
            "time_window_hours": hours,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "analysis": analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting threat analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get threat analysis"
        )

@router.get("/audit/compliance-report", summary="Export compliance report")
async def export_compliance_report(
    start_date: datetime,
    end_date: datetime,
    format: str = "json"
):
    """Export compliance report for the specified date range."""
    try:
        if end_date <= start_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="End date must be after start date"
            )
        
        # Limit report range to 1 year
        if (end_date - start_date).days > 365:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Report range cannot exceed 365 days"
            )
        
        audit_logger = get_audit_logger()
        report = await audit_logger.export_compliance_report(
            start_date, end_date, format
        )
        
        if format.lower() == "json":
            import json
            return JSONResponse(
                content=json.loads(report),
                headers={
                    "Content-Disposition": f"attachment; filename=compliance_report_{start_date.date()}_{end_date.date()}.json"
                }
            )
        else:
            return JSONResponse(
                content={"report": report},
                headers={
                    "Content-Disposition": f"attachment; filename=compliance_report_{start_date.date()}_{end_date.date()}.txt"
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting compliance report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export compliance report"
        )

@router.post("/input/sanitize", summary="Test input sanitization")
async def test_input_sanitization(request: InputSanitizeRequest):
    """Test input sanitization functionality."""
    try:
        sanitizer = get_input_sanitizer()
        
        # Perform sanitization
        sanitized_text = sanitizer.sanitize_string(
            request.input_text, 
            strict=request.strict
        )
        
        # Perform security checks
        security_checks = sanitizer.comprehensive_check(request.input_text)
        is_safe = sanitizer.is_safe_input(request.input_text)
        
        return {
            "original_text": request.input_text,
            "sanitized_text": sanitized_text,
            "is_safe": is_safe,
            "security_checks": security_checks,
            "strict_mode": request.strict
        }
        
    except Exception as e:
        logger.error(f"Error testing input sanitization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to test input sanitization"
        )

@router.get("/input/validate/{text}", summary="Validate input text")
async def validate_input_text(text: str):
    """Validate input text for security threats."""
    try:
        sanitizer = get_input_sanitizer()
        
        # Validate and sanitize
        sanitized_text, is_safe = sanitizer.validate_and_sanitize(text)
        security_checks = sanitizer.comprehensive_check(text)
        
        return {
            "original_text": text,
            "sanitized_text": sanitized_text,
            "is_safe": is_safe,
            "security_checks": security_checks,
            "threats_detected": any(security_checks.values())
        }
        
    except Exception as e:
        logger.error(f"Error validating input text: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate input text"
        )

@router.get("/health", summary="Security module health check")
async def security_health_check():
    """Check the health status of all security modules."""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "modules": {}
        }
        
        # Check rate limiter
        try:
            rate_limiter = get_rate_limiter()
            # Test basic functionality
            test_allowed, _ = await rate_limiter.is_allowed("health_check", "test", 1)
            health_status["modules"]["rate_limiter"] = {
                "status": "healthy",
                "has_redis": rate_limiter.redis_client is not None,
                "rules_count": len(rate_limiter.rules)
            }
        except Exception as e:
            health_status["modules"]["rate_limiter"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check input sanitizer
        try:
            sanitizer = get_input_sanitizer()
            # Test basic functionality
            test_result = sanitizer.sanitize_string("<script>test</script>")
            health_status["modules"]["input_sanitizer"] = {
                "status": "healthy",
                "patterns_loaded": len(sanitizer.xss_regex) + len(sanitizer.sql_regex)
            }
        except Exception as e:
            health_status["modules"]["input_sanitizer"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check audit logger
        try:
            audit_logger = get_audit_logger()
            health_status["modules"]["audit_logger"] = {
                "status": "healthy",
                "storage_backends": len(audit_logger.storage_backends),
                "analysis_enabled": audit_logger.enable_analysis
            }
        except Exception as e:
            health_status["modules"]["audit_logger"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error in security health check: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@router.get("/metrics", summary="Security metrics")
async def get_security_metrics():
    """Get comprehensive security metrics."""
    try:
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "rate_limiting": {},
            "threat_detection": {},
            "input_validation": {}
        }
        
        # Rate limiting metrics
        rate_limiter = get_rate_limiter()
        if hasattr(rate_limiter, 'local_state'):
            total_clients = len(rate_limiter.local_state)
            blocked_clients = sum(
                1 for state in rate_limiter.local_state.values()
                if state.blocked_until and state.blocked_until > datetime.utcnow().timestamp()
            )
            total_violations = sum(
                state.violations for state in rate_limiter.local_state.values()
            )
            
            metrics["rate_limiting"] = {
                "total_clients": total_clients,
                "blocked_clients": blocked_clients,
                "total_violations": total_violations,
                "active_rules": len(rate_limiter.rules)
            }
        
        # Get recent threat analysis
        audit_logger = get_audit_logger()
        try:
            threat_analysis = await audit_logger.get_threat_analysis(timedelta(hours=1))
            if "summary" in threat_analysis:
                metrics["threat_detection"] = threat_analysis["summary"]
        except:
            metrics["threat_detection"] = {"error": "Unable to get threat analysis"}
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting security metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get security metrics"
        )