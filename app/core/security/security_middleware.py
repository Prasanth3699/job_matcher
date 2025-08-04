"""
Comprehensive security middleware for FastAPI applications.
Provides multiple layers of security protection.
"""

import time
import hashlib
import secrets
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime, timedelta
from ipaddress import ip_address, ip_network, AddressValueError

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware

from app.utils.logger import logger
from app.core.security.input_sanitizer import get_input_sanitizer
from app.core.security.audit_logger import get_audit_logger


class SecurityHeaders:
    """Security headers configuration."""
    
    DEFAULT_HEADERS = {
        # Prevent XSS attacks
        "X-XSS-Protection": "1; mode=block",
        "X-Content-Type-Options": "nosniff",
        
        # Prevent clickjacking
        "X-Frame-Options": "DENY",
        
        # Force HTTPS
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
        
        # Content Security Policy
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'"
        ),
        
        # Referrer policy
        "Referrer-Policy": "strict-origin-when-cross-origin",
        
        # Permissions policy
        "Permissions-Policy": (
            "camera=(), microphone=(), geolocation=(), "
            "accelerometer=(), gyroscope=(), magnetometer=(), "
            "payment=(), usb=()"
        ),
        
        # Cache control for sensitive data
        "Cache-Control": "no-store, no-cache, must-revalidate, private",
        "Pragma": "no-cache",
        "Expires": "0",
        
        # Server identification
        "Server": "JobMatcher-API/1.0"
    }


class IPWhitelist:
    """IP address whitelist management."""
    
    def __init__(self, allowed_networks: Optional[List[str]] = None):
        self.allowed_networks = []
        if allowed_networks:
            for network in allowed_networks:
                try:
                    self.allowed_networks.append(ip_network(network, strict=False))
                except AddressValueError as e:
                    logger.error(f"Invalid IP network '{network}': {e}")
    
    def is_allowed(self, ip_addr: str) -> bool:
        """Check if IP address is in whitelist."""
        if not self.allowed_networks:
            return True  # No restrictions if whitelist is empty
        
        try:
            client_ip = ip_address(ip_addr)
            return any(client_ip in network for network in self.allowed_networks)
        except AddressValueError:
            return False
    
    def add_network(self, network: str) -> bool:
        """Add network to whitelist."""
        try:
            self.allowed_networks.append(ip_network(network, strict=False))
            return True
        except AddressValueError as e:
            logger.error(f"Invalid IP network '{network}': {e}")
            return False


class RequestValidator:
    """Request validation and sanitization."""
    
    def __init__(self):
        self.sanitizer = get_input_sanitizer()
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.max_header_size = 8192  # 8KB
        self.max_url_length = 2048
        self.forbidden_patterns = [
            # Common attack patterns
            r'<script[^>]*>',
            r'javascript:',
            r'vbscript:',
            r'data:text/html',
            r'eval\(',
            r'alert\(',
            r'document\.cookie',
            r'window\.location',
        ]
    
    def validate_request_size(self, request: Request) -> bool:
        """Validate request size limits."""
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            return False
        return True
    
    def validate_headers(self, request: Request) -> bool:
        """Validate request headers."""
        total_header_size = sum(
            len(name) + len(value) 
            for name, value in request.headers.items()
        )
        return total_header_size <= self.max_header_size
    
    def validate_url(self, request: Request) -> bool:
        """Validate URL length and format."""
        url = str(request.url)
        return len(url) <= self.max_url_length
    
    def check_malicious_patterns(self, data: str) -> bool:
        """Check for malicious patterns in data."""
        return not self.sanitizer.is_safe_input(data)


class ThreatDetector:
    """Advanced threat detection system."""
    
    def __init__(self):
        self.suspicious_patterns = {
            'sql_injection': [
                'union select', 'drop table', 'insert into',
                'delete from', 'update set', 'exec(',
                'sp_executesql', 'xp_cmdshell'
            ],
            'xss': [
                '<script', 'javascript:', 'onload=',
                'onerror=', 'onclick=', 'alert(',
                'document.write', 'innerHTML'
            ],
            'path_traversal': [
                '../', '..\\', '%2e%2e%2f',
                '%2e%2e\\', '..%2f', '..%5c'
            ],
            'command_injection': [
                '$(', '`', '||', '&&',
                ';rm', ';cat', ';ls', ';pwd',
                'system(', 'exec(', 'eval('
            ]
        }
        
        # Track suspicious activity
        self.threat_scores: Dict[str, float] = {}
        self.last_cleanup = time.time()
    
    def analyze_request(self, request: Request) -> Dict[str, Any]:
        """Analyze request for potential threats."""
        threat_info = {
            'threat_level': 0.0,
            'detected_threats': [],
            'suspicious_patterns': [],
            'risk_score': 0.0
        }
        
        # Analyze URL
        url = str(request.url)
        threat_info.update(self._analyze_text(url, 'url'))
        
        # Analyze headers
        for name, value in request.headers.items():
            header_threats = self._analyze_text(f"{name}: {value}", 'header')
            threat_info['threat_level'] += header_threats['threat_level']
            threat_info['detected_threats'].extend(header_threats['detected_threats'])
        
        # Analyze query parameters
        for key, value in request.query_params.items():
            param_threats = self._analyze_text(f"{key}={value}", 'parameter')
            threat_info['threat_level'] += param_threats['threat_level']
            threat_info['detected_threats'].extend(param_threats['detected_threats'])
        
        # Calculate final risk score
        threat_info['risk_score'] = min(threat_info['threat_level'], 10.0)
        
        return threat_info
    
    def _analyze_text(self, text: str, context: str) -> Dict[str, Any]:
        """Analyze text for threat patterns."""
        threats = {
            'threat_level': 0.0,
            'detected_threats': [],
            'suspicious_patterns': []
        }
        
        text_lower = text.lower()
        
        # Check each threat category
        for threat_type, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    threats['detected_threats'].append({
                        'type': threat_type,
                        'pattern': pattern,
                        'context': context,
                        'severity': self._get_pattern_severity(threat_type, pattern)
                    })
                    threats['threat_level'] += self._get_pattern_severity(threat_type, pattern)
        
        return threats
    
    def _get_pattern_severity(self, threat_type: str, pattern: str) -> float:
        """Get severity score for a threat pattern."""
        severity_map = {
            'sql_injection': {
                'union select': 3.0,
                'drop table': 4.0,
                'delete from': 3.5,
                'exec(': 3.0
            },
            'xss': {
                '<script': 4.0,
                'javascript:': 3.5,
                'alert(': 2.0,
                'document.write': 3.0
            },
            'command_injection': {
                'system(': 4.0,
                'exec(': 3.5,
                ';rm': 4.0,
                '$(': 2.5
            },
            'path_traversal': {
                '../': 2.0,
                '..\\': 2.0,
                '%2e%2e%2f': 3.0
            }
        }
        
        return severity_map.get(threat_type, {}).get(pattern, 1.0)
    
    def update_threat_score(self, client_ip: str, threat_level: float) -> float:
        """Update and return cumulative threat score for client."""
        current_time = time.time()
        
        # Cleanup old scores (older than 1 hour)
        if current_time - self.last_cleanup > 3600:
            self._cleanup_old_scores(current_time)
            self.last_cleanup = current_time
        
        # Update threat score with decay
        if client_ip in self.threat_scores:
            # Apply time decay (reduce score over time)
            time_decay = 0.99 ** (current_time % 3600)  # Decay every hour
            self.threat_scores[client_ip] *= time_decay
        else:
            self.threat_scores[client_ip] = 0.0
        
        self.threat_scores[client_ip] += threat_level
        return self.threat_scores[client_ip]
    
    def _cleanup_old_scores(self, current_time: float) -> None:
        """Remove old threat scores."""
        expired_clients = [
            ip for ip, score in self.threat_scores.items()
            if score < 0.1  # Very low scores considered expired
        ]
        
        for client_ip in expired_clients:
            del self.threat_scores[client_ip]


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive security middleware providing multiple protection layers.
    
    Features:
    - Security headers injection
    - Request validation and sanitization
    - Threat detection and blocking
    - IP whitelisting
    - Request size limits
    - Malicious pattern detection
    - Security event logging
    """
    
    def __init__(
        self,
        app,
        security_headers: Optional[Dict[str, str]] = None,
        ip_whitelist: Optional[List[str]] = None,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        threat_threshold: float = 5.0,
        block_duration: int = 300,  # 5 minutes
        exclude_paths: Optional[List[str]] = None,
        custom_validators: Optional[List[Callable]] = None
    ):
        super().__init__(app)
        
        # Configuration
        self.security_headers = {
            **SecurityHeaders.DEFAULT_HEADERS,
            **(security_headers or {})
        }
        self.ip_whitelist = IPWhitelist(ip_whitelist)
        self.max_request_size = max_request_size
        self.threat_threshold = threat_threshold
        self.block_duration = block_duration
        self.exclude_paths = exclude_paths or [
            "/health", "/metrics", "/docs", "/redoc", "/openapi.json"
        ]
        self.custom_validators = custom_validators or []
        
        # Security components
        self.request_validator = RequestValidator()
        self.threat_detector = ThreatDetector()
        self.audit_logger = get_audit_logger()
        
        # Blocked IPs with expiration times
        self.blocked_ips: Dict[str, float] = {}
        
        logger.info("Security middleware initialized")
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process security checks for incoming requests."""
        start_time = time.time()
        
        # Skip security checks for excluded paths
        if request.url.path in self.exclude_paths:
            response = await call_next(request)
            self._add_security_headers(response)
            return response
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        try:
            # Check if IP is blocked
            if self._is_ip_blocked(client_ip):
                return self._create_blocked_response(client_ip)
            
            # IP whitelist check
            if not self.ip_whitelist.is_allowed(client_ip):
                await self._log_security_event(
                    "ip_not_whitelisted",
                    request,
                    {"client_ip": client_ip}
                )
                return self._create_forbidden_response("IP not whitelisted")
            
            # Validate request
            validation_result = await self._validate_request(request)
            if not validation_result['valid']:
                await self._log_security_event(
                    "request_validation_failed",
                    request,
                    validation_result
                )
                return self._create_bad_request_response(validation_result['reason'])
            
            # Threat detection
            threat_info = self.threat_detector.analyze_request(request)
            total_threat_score = self.threat_detector.update_threat_score(
                client_ip, threat_info['threat_level']
            )
            
            # Block if threat threshold exceeded
            if total_threat_score > self.threat_threshold:
                self._block_ip(client_ip)
                await self._log_security_event(
                    "threat_threshold_exceeded",
                    request,
                    {
                        "client_ip": client_ip,
                        "threat_score": total_threat_score,
                        "threat_info": threat_info
                    }
                )
                return self._create_blocked_response(client_ip)
            
            # Log suspicious activity
            if threat_info['threat_level'] > 1.0:
                await self._log_security_event(
                    "suspicious_activity",
                    request,
                    {
                        "client_ip": client_ip,
                        "threat_info": threat_info
                    }
                )
            
            # Run custom validators
            for validator in self.custom_validators:
                try:
                    if not await validator(request):
                        return self._create_forbidden_response("Custom validation failed")
                except Exception as e:
                    logger.error(f"Custom validator error: {e}")
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response)
            
            # Log successful request
            processing_time = time.time() - start_time
            await self._log_security_event(
                "request_processed",
                request,
                {
                    "client_ip": client_ip,
                    "processing_time": processing_time,
                    "threat_score": total_threat_score
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            await self._log_security_event(
                "middleware_error",
                request,
                {"error": str(e), "client_ip": client_ip}
            )
            
            # Return a safe error response
            return self._create_error_response("Internal security error")
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Try X-Real-IP header first (nginx)
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Try X-Forwarded-For header
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Try X-Forwarded-Host
        forwarded_host = request.headers.get("X-Forwarded-Host")
        if forwarded_host:
            return forwarded_host
        
        # Fallback to direct connection IP
        return request.client.host if request.client else "unknown"
    
    async def _validate_request(self, request: Request) -> Dict[str, Any]:
        """Validate incoming request."""
        # Request size validation
        if not self.request_validator.validate_request_size(request):
            return {
                'valid': False,
                'reason': 'Request too large',
                'details': {'max_size': self.max_request_size}
            }
        
        # Headers validation
        if not self.request_validator.validate_headers(request):
            return {
                'valid': False,
                'reason': 'Headers too large',
                'details': {'max_header_size': self.request_validator.max_header_size}
            }
        
        # URL validation
        if not self.request_validator.validate_url(request):
            return {
                'valid': False,
                'reason': 'URL too long',
                'details': {'max_url_length': self.request_validator.max_url_length}
            }
        
        # Content type validation for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not self._is_allowed_content_type(content_type):
                return {
                    'valid': False,
                    'reason': 'Invalid content type',
                    'details': {'content_type': content_type}
                }
        
        return {'valid': True}
    
    def _is_allowed_content_type(self, content_type: str) -> bool:
        """Check if content type is allowed."""
        allowed_types = [
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain",
            "application/octet-stream"
        ]
        
        return any(
            content_type.startswith(allowed_type) 
            for allowed_type in allowed_types
        )
    
    def _is_ip_blocked(self, client_ip: str) -> bool:
        """Check if IP is currently blocked."""
        if client_ip not in self.blocked_ips:
            return False
        
        # Check if block has expired
        if time.time() > self.blocked_ips[client_ip]:
            del self.blocked_ips[client_ip]
            return False
        
        return True
    
    def _block_ip(self, client_ip: str) -> None:
        """Block IP for specified duration."""
        self.blocked_ips[client_ip] = time.time() + self.block_duration
        logger.warning(f"Blocked IP {client_ip} for {self.block_duration} seconds")
    
    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to response."""
        for header, value in self.security_headers.items():
            response.headers[header] = value
    
    def _create_blocked_response(self, client_ip: str) -> JSONResponse:
        """Create response for blocked IP."""
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": "Access forbidden",
                "message": "Your IP address has been temporarily blocked due to suspicious activity",
                "blocked_until": self.blocked_ips.get(client_ip, 0),
                "client_ip": client_ip
            },
            headers=self.security_headers
        )
    
    def _create_forbidden_response(self, reason: str) -> JSONResponse:
        """Create forbidden response."""
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": "Access forbidden",
                "message": reason
            },
            headers=self.security_headers
        )
    
    def _create_bad_request_response(self, reason: str) -> JSONResponse:
        """Create bad request response."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Bad request",
                "message": reason
            },
            headers=self.security_headers
        )
    
    def _create_error_response(self, message: str) -> JSONResponse:
        """Create generic error response."""
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "message": message
            },
            headers=self.security_headers
        )
    
    async def _log_security_event(
        self,
        event_type: str,
        request: Request,
        details: Dict[str, Any]
    ) -> None:
        """Log security event."""
        try:
            event_data = {
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "client_ip": self._get_client_ip(request),
                "method": request.method,
                "url": str(request.url),
                "user_agent": request.headers.get("user-agent", ""),
                "details": details
            }
            
            await self.audit_logger.log_security_event(event_data)
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")


# Utility functions
def create_security_middleware(
    app,
    config: Optional[Dict[str, Any]] = None
) -> SecurityMiddleware:
    """Create security middleware with configuration."""
    config = config or {}
    
    return SecurityMiddleware(
        app,
        security_headers=config.get("security_headers"),
        ip_whitelist=config.get("ip_whitelist"),
        max_request_size=config.get("max_request_size", 10 * 1024 * 1024),
        threat_threshold=config.get("threat_threshold", 5.0),
        block_duration=config.get("block_duration", 300),
        exclude_paths=config.get("exclude_paths"),
        custom_validators=config.get("custom_validators")
    )


def get_client_ip(request: Request) -> str:
    """Get client IP address from request (utility function)."""
    middleware = SecurityMiddleware(None)
    return middleware._get_client_ip(request)