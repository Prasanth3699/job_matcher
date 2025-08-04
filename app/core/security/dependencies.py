"""
Security dependencies for FastAPI endpoints.
Provides reusable security checks and middleware functions.
"""

from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.security.rate_limiter import get_rate_limiter, check_rate_limit
from app.core.security.input_sanitizer import get_input_sanitizer, is_safe_input
from app.core.security.audit_logger import get_audit_logger, SecurityEventType, SeverityLevel
from app.utils.logger import logger

security = HTTPBearer(auto_error=False)

async def check_rate_limit_dependency(
    request: Request,
    rule_identifier: str = "api_general"
) -> None:
    """
    FastAPI dependency for rate limiting.
    
    Args:
        request: FastAPI request object
        rule_identifier: Rate limit rule to apply
        
    Raises:
        HTTPException: When rate limit is exceeded
    """
    # Extract client IP
    client_ip = request.headers.get("X-Real-IP") or \
                request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or \
                request.client.host if request.client else "unknown"
    
    # Check rate limit
    allowed, metadata = await check_rate_limit(
        client_id=client_ip,
        identifier=rule_identifier,
        cost=1
    )
    
    if not allowed:
        # Log rate limit violation
        audit_logger = get_audit_logger()
        await audit_logger.log_security_event({
            "event_type": SecurityEventType.RATE_LIMIT_EXCEEDED.value,
            "client_ip": client_ip,
            "endpoint": str(request.url.path),
            "method": request.method,
            "success": False,
            "details": {
                "rule_identifier": rule_identifier,
                "metadata": metadata
            }
        })
        
        headers = {
            "X-RateLimit-Limit": str(metadata.get("requests_allowed", "unknown")),
            "X-RateLimit-Remaining": str(metadata.get("remaining", 0)),
            "X-RateLimit-Reset": str(int(metadata.get("reset_time", 0))),
            "Retry-After": str(int(metadata.get("remaining_time", 60)))
        }
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please try again later.",
                "details": {
                    "limit": metadata.get("requests_allowed"),
                    "window": metadata.get("window_size"),
                    "reset_time": metadata.get("reset_time"),
                    "blocked": metadata.get("blocked", False)
                }
            },
            headers=headers
        )

async def validate_input_dependency(request: Request) -> None:
    """
    FastAPI dependency for input validation.
    
    Args:
        request: FastAPI request object
        
    Raises:
        HTTPException: When malicious input is detected
    """
    sanitizer = get_input_sanitizer()
    
    # Check URL for malicious patterns
    url_str = str(request.url)
    if not is_safe_input(url_str):
        await _log_malicious_input("url", url_str, request)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Malicious input detected",
                "message": "Request contains potentially harmful content"
            }
        )
    
    # Check query parameters
    for key, value in request.query_params.items():
        if not is_safe_input(value):
            await _log_malicious_input(f"query_param_{key}", value, request)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Malicious input detected",
                    "message": f"Query parameter '{key}' contains potentially harmful content"
                }
            )
    
    # Check headers for suspicious patterns
    suspicious_headers = ['user-agent', 'referer', 'x-forwarded-for']
    for header_name in suspicious_headers:
        header_value = request.headers.get(header_name)
        if header_value and not is_safe_input(header_value):
            await _log_malicious_input(f"header_{header_name}", header_value, request)
            # Don't block on suspicious headers, just log
            logger.warning(f"Suspicious header content detected: {header_name}")

async def _log_malicious_input(source: str, content: str, request: Request) -> None:
    """Log malicious input detection."""
    client_ip = request.headers.get("X-Real-IP") or \
                request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or \
                request.client.host if request.client else "unknown"
    
    audit_logger = get_audit_logger()
    await audit_logger.log_security_event({
        "event_type": SecurityEventType.MALICIOUS_REQUEST.value,
        "client_ip": client_ip,
        "endpoint": str(request.url.path),
        "method": request.method,
        "success": False,
        "details": {
            "source": source,
            "content_preview": content[:100],  # First 100 chars
            "threat_detected": True
        }
    })

# Dependency functions for different security levels
async def standard_security_check(request: Request) -> None:
    """Standard security checks for most endpoints."""
    await check_rate_limit_dependency(request, "api_general")
    await validate_input_dependency(request)

async def strict_security_check(request: Request) -> None:
    """Strict security checks for sensitive endpoints."""
    await check_rate_limit_dependency(request, "auth_endpoints")
    await validate_input_dependency(request)

async def auth_security_check(request: Request) -> None:
    """Security checks specifically for authentication endpoints."""
    await check_rate_limit_dependency(request, "auth_endpoints")
    await validate_input_dependency(request)

# Rate limit dependencies for specific use cases
class RateLimitDependency:
    """Factory for creating rate limit dependencies."""
    
    @staticmethod
    def create(rule_identifier: str = "api_general"):
        """Create a rate limit dependency for a specific rule."""
        async def rate_limit_check(request: Request):
            return await check_rate_limit_dependency(request, rule_identifier)
        return rate_limit_check

# Pre-configured dependencies
api_rate_limit = RateLimitDependency.create("api_general")
auth_rate_limit = RateLimitDependency.create("auth_endpoints")
upload_rate_limit = RateLimitDependency.create("file_uploads")

# Security check dependencies
StandardSecurity = Depends(standard_security_check)
StrictSecurity = Depends(strict_security_check)
AuthSecurity = Depends(auth_security_check)

# Rate limit only dependencies
APIRateLimit = Depends(api_rate_limit)
AuthRateLimit = Depends(auth_rate_limit)
UploadRateLimit = Depends(upload_rate_limit)

async def log_successful_request(request: Request) -> None:
    """Log successful request for audit purposes."""
    client_ip = request.headers.get("X-Real-IP") or \
                request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or \
                request.client.host if request.client else "unknown"
    
    # Only log sensitive operations
    sensitive_paths = ['/auth/', '/admin/', '/security/', '/user/']
    is_sensitive = any(path in str(request.url.path) for path in sensitive_paths)
    
    if is_sensitive:
        audit_logger = get_audit_logger()
        await audit_logger.log_security_event({
            "event_type": SecurityEventType.DATA_ACCESS.value,
            "client_ip": client_ip,
            "endpoint": str(request.url.path),
            "method": request.method,
            "success": True,
            "details": {
                "endpoint_type": "sensitive",
                "user_agent": request.headers.get("user-agent", "unknown")
            }
        })

# Request logging dependency
RequestLogging = Depends(log_successful_request)