"""
Correlation ID tracking system for request tracing across services.
Enables end-to-end request tracking and debugging in distributed systems.
"""

import uuid
import threading
from typing import Optional, Dict, Any, List
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.utils.logger import logger


# Context variable for correlation ID
correlation_id_context: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


@dataclass
class RequestContext:
    """Request context information for correlation tracking."""
    
    correlation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.utcnow)
    method: Optional[str] = None
    path: Optional[str] = None
    user_agent: Optional[str] = None
    client_ip: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        return {
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "start_time": self.start_time.isoformat(),
            "method": self.method,
            "path": self.path,
            "user_agent": self.user_agent,
            "client_ip": self.client_ip,
            "metadata": self.metadata,
        }


class CorrelationTracker:
    """
    Correlation ID tracking and management system.
    Provides utilities for creating, managing, and propagating correlation IDs.
    """
    
    DEFAULT_HEADER_NAME = "X-Correlation-ID"
    REQUEST_ID_HEADER = "X-Request-ID"
    
    def __init__(self, header_name: str = DEFAULT_HEADER_NAME):
        self.header_name = header_name
        self._contexts: Dict[str, RequestContext] = {}
        self._lock = threading.RLock()
    
    @classmethod
    def generate_correlation_id(cls) -> str:
        """Generate a new correlation ID."""
        return str(uuid.uuid4())
    
    @classmethod
    def get_current_correlation_id(cls) -> Optional[str]:
        """Get the current correlation ID from context."""
        return correlation_id_context.get()
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str) -> None:
        """Set the correlation ID in context."""
        correlation_id_context.set(correlation_id)
    
    @classmethod
    def clear_correlation_id(cls) -> None:
        """Clear the correlation ID from context."""
        correlation_id_context.set(None)
    
    def extract_correlation_id(self, request: Request) -> str:
        """
        Extract correlation ID from request headers or generate a new one.
        """
        # Try to get from custom header
        correlation_id = request.headers.get(self.header_name)
        
        # Fallback to other common headers
        if not correlation_id:
            correlation_id = request.headers.get("X-Trace-ID")
        if not correlation_id:
            correlation_id = request.headers.get("X-Request-ID")
        
        # Generate new one if not found
        if not correlation_id:
            correlation_id = self.generate_correlation_id()
        
        return correlation_id
    
    def create_request_context(self, request: Request, correlation_id: str) -> RequestContext:
        """Create request context from HTTP request."""
        
        # Extract user information if available
        user_id = None
        session_id = None
        
        # Try to get user info from auth headers or request state
        if hasattr(request.state, 'user'):
            user_data = getattr(request.state, 'user', {})
            if isinstance(user_data, dict):
                user_id = user_data.get('user_id') or user_data.get('id')
                session_id = user_data.get('session_id')
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        context = RequestContext(
            correlation_id=correlation_id,
            user_id=str(user_id) if user_id else None,
            session_id=session_id,
            method=request.method,
            path=str(request.url.path),
            user_agent=request.headers.get("User-Agent"),
            client_ip=client_ip,
            metadata={
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
            }
        )
        
        return context
    
    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Extract client IP from request headers."""
        # Try various headers in order of preference
        headers_to_check = [
            "X-Forwarded-For",
            "X-Real-IP", 
            "X-Client-IP",
            "CF-Connecting-IP",  # Cloudflare
        ]
        
        for header in headers_to_check:
            ip = request.headers.get(header)
            if ip:
                # X-Forwarded-For can contain multiple IPs
                return ip.split(',')[0].strip()
        
        # Fallback to direct client
        if hasattr(request, 'client') and request.client:
            return request.client.host
        
        return None
    
    def register_context(self, context: RequestContext) -> None:
        """Register a request context for tracking."""
        with self._lock:
            self._contexts[context.correlation_id] = context
            # Set in current context
            self.set_correlation_id(context.correlation_id)
    
    def get_context(self, correlation_id: str) -> Optional[RequestContext]:
        """Get request context by correlation ID."""
        with self._lock:
            return self._contexts.get(correlation_id)
    
    def update_context(self, correlation_id: str, **kwargs) -> None:
        """Update request context metadata."""
        with self._lock:
            if correlation_id in self._contexts:
                context = self._contexts[correlation_id]
                context.metadata.update(kwargs)
    
    def cleanup_context(self, correlation_id: str) -> None:
        """Clean up request context after processing."""
        with self._lock:
            self._contexts.pop(correlation_id, None)
    
    def get_active_contexts(self) -> List[RequestContext]:
        """Get all active request contexts."""
        with self._lock:
            return list(self._contexts.values())
    
    def get_context_count(self) -> int:
        """Get number of active contexts."""
        with self._lock:
            return len(self._contexts)


class CorrelationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic correlation ID tracking.
    """
    
    def __init__(self, app, tracker: Optional[CorrelationTracker] = None):
        super().__init__(app)
        self.tracker = tracker or CorrelationTracker()
    
    async def dispatch(self, request: Request, call_next):
        """Process request with correlation tracking."""
        
        # Extract or generate correlation ID
        correlation_id = self.tracker.extract_correlation_id(request)
        
        # Create request context
        context = self.tracker.create_request_context(request, correlation_id)
        
        # Register context
        self.tracker.register_context(context)
        
        # Log request start
        logger.info(
            "Request started",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": str(request.url.path),
                "client_ip": context.client_ip,
                "user_agent": context.user_agent,
            }
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Add correlation ID to response headers
            response.headers[self.tracker.header_name] = correlation_id
            response.headers["X-Request-ID"] = context.request_id
            
            # Calculate request duration
            duration = (datetime.utcnow() - context.start_time).total_seconds() * 1000
            
            # Log request completion
            logger.info(
                "Request completed",
                extra={
                    "correlation_id": correlation_id,
                    "status_code": response.status_code,
                    "duration_ms": duration,
                }
            )
            
            return response
            
        except Exception as e:
            # Log request error
            duration = (datetime.utcnow() - context.start_time).total_seconds() * 1000
            
            logger.error(
                "Request failed",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e),
                    "duration_ms": duration,
                },
                exc_info=True
            )
            
            raise
            
        finally:
            # Cleanup context
            self.tracker.cleanup_context(correlation_id)
            self.tracker.clear_correlation_id()


# Global tracker instance
_global_tracker = CorrelationTracker()


def get_correlation_tracker() -> CorrelationTracker:
    """Get the global correlation tracker instance."""
    return _global_tracker


def get_current_correlation_id() -> Optional[str]:
    """Get the current correlation ID (convenience function)."""
    return CorrelationTracker.get_current_correlation_id()


def set_correlation_id(correlation_id: str) -> None:
    """Set the current correlation ID (convenience function)."""
    CorrelationTracker.set_correlation_id(correlation_id)


def generate_correlation_id() -> str:
    """Generate a new correlation ID (convenience function)."""
    return CorrelationTracker.generate_correlation_id()


# Decorator for adding correlation ID to function calls
def with_correlation_id(func):
    """Decorator to ensure correlation ID is propagated to function calls."""
    def wrapper(*args, **kwargs):
        correlation_id = get_current_correlation_id()
        if correlation_id:
            # Add correlation_id to keyword arguments if not already present
            if 'correlation_id' not in kwargs:
                kwargs['correlation_id'] = correlation_id
        
        return func(*args, **kwargs)
    
    return wrapper