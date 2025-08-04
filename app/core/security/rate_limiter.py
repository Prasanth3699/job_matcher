"""
Rate limiting middleware for API endpoints.
Provides protection against abuse and DoS attacks.
"""

import time
import asyncio
from typing import Dict, List, Optional, Callable, Any, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import redis.asyncio as redis
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

from app.utils.logger import logger


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitRule:
    """Rate limit rule configuration."""

    requests: int  # Number of requests allowed
    window: int  # Time window in seconds
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_multiplier: float = 1.5  # Allow burst up to this multiplier
    block_duration: int = 300  # Block duration in seconds when limit exceeded

    def __post_init__(self):
        if self.requests <= 0:
            raise ValueError("Requests must be positive")
        if self.window <= 0:
            raise ValueError("Window must be positive")


@dataclass
class ClientState:
    """Client rate limiting state."""

    request_times: deque = field(default_factory=deque)
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.time)
    blocked_until: Optional[float] = None
    total_requests: int = 0
    violations: int = 0


class RateLimiter:
    """
    Advanced rate limiter with multiple strategies and Redis support.

    Features:
    - Multiple rate limiting strategies
    - Per-client tracking
    - Redis-based distributed rate limiting
    - Configurable burst handling
    - Automatic cleanup of expired data
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        default_rule: Optional[RateLimitRule] = None,
        cleanup_interval: int = 3600,
        key_prefix: str = "rate_limit",
    ):
        self.redis_client = redis_client
        self.default_rule = default_rule or RateLimitRule(
            requests=100, window=3600, strategy=RateLimitStrategy.SLIDING_WINDOW
        )
        self.cleanup_interval = cleanup_interval
        self.key_prefix = key_prefix

        # In-memory state for when Redis is not available
        self.local_state: Dict[str, ClientState] = defaultdict(ClientState)
        self.rules: Dict[str, RateLimitRule] = {}

        # Last cleanup time
        self.last_cleanup = time.time()

        logger.info(
            f"Rate limiter initialized with strategy: {self.default_rule.strategy.value}"
        )

    def add_rule(self, identifier: str, rule: RateLimitRule) -> None:
        """Add a custom rate limiting rule for specific endpoints or clients."""
        self.rules[identifier] = rule
        logger.info(
            f"Added rate limit rule for '{identifier}': {rule.requests}/{rule.window}s"
        )

    def get_rule(self, identifier: str) -> RateLimitRule:
        """Get rate limiting rule for identifier, falling back to default."""
        return self.rules.get(identifier, self.default_rule)

    async def is_allowed(
        self, client_id: str, identifier: str = "default", cost: int = 1
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed based on rate limiting rules.

        Args:
            client_id: Unique identifier for the client
            identifier: Rule identifier (endpoint, user type, etc.)
            cost: Request cost (for weighted rate limiting)

        Returns:
            Tuple of (is_allowed, metadata)
        """
        rule = self.get_rule(identifier)
        current_time = time.time()

        # Cleanup expired data periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_expired_data()
            self.last_cleanup = current_time

        if self.redis_client:
            return await self._check_redis_rate_limit(
                client_id, rule, current_time, cost
            )
        else:
            return await self._check_local_rate_limit(
                client_id, rule, current_time, cost
            )

    async def _check_redis_rate_limit(
        self, client_id: str, rule: RateLimitRule, current_time: float, cost: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using Redis backend."""
        key = f"{self.key_prefix}:{client_id}"

        try:
            # Check if client is currently blocked
            blocked_key = f"{key}:blocked"
            blocked_until = await self.redis_client.get(blocked_key)

            if blocked_until and float(blocked_until) > current_time:
                remaining_time = float(blocked_until) - current_time
                return False, {
                    "blocked": True,
                    "blocked_until": float(blocked_until),
                    "remaining_time": remaining_time,
                    "rule": rule,
                }

            # Apply rate limiting strategy
            if rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return await self._redis_sliding_window(key, rule, current_time, cost)
            elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return await self._redis_token_bucket(key, rule, current_time, cost)
            elif rule.strategy == RateLimitStrategy.FIXED_WINDOW:
                return await self._redis_fixed_window(key, rule, current_time, cost)
            else:
                # Default to sliding window
                return await self._redis_sliding_window(key, rule, current_time, cost)

        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            # Fallback to local rate limiting
            return await self._check_local_rate_limit(
                client_id, rule, current_time, cost
            )

    async def _redis_sliding_window(
        self, key: str, rule: RateLimitRule, current_time: float, cost: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Sliding window rate limiting with Redis."""
        window_start = current_time - rule.window

        # Use Redis pipeline for atomic operations
        pipe = self.redis_client.pipeline()

        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)

        # Count current requests
        pipe.zcard(key)

        # Add current request
        pipe.zadd(key, {str(current_time): current_time})

        # Set expiry
        pipe.expire(key, rule.window)

        results = await pipe.execute()
        current_count = results[1] + cost

        allowed = current_count <= rule.requests

        if not allowed:
            # Remove the request we just added since it's not allowed
            await self.redis_client.zrem(key, str(current_time))

            # Block client if too many violations
            violations_key = f"{key}:violations"
            violations = await self.redis_client.incr(violations_key)
            await self.redis_client.expire(violations_key, rule.window)

            if violations > rule.requests * 0.1:  # 10% violation threshold
                blocked_key = f"{key}:blocked"
                blocked_until = current_time + rule.block_duration
                await self.redis_client.setex(
                    blocked_key, rule.block_duration, str(blocked_until)
                )

        # Calculate reset time
        oldest_request = await self.redis_client.zrange(key, 0, 0, withscores=True)
        reset_time = (
            oldest_request[0][1] + rule.window
            if oldest_request
            else current_time + rule.window
        )

        return allowed, {
            "requests_made": current_count,
            "requests_allowed": rule.requests,
            "window_size": rule.window,
            "reset_time": reset_time,
            "remaining": max(0, rule.requests - current_count),
            "rule": rule,
        }

    async def _redis_token_bucket(
        self, key: str, rule: RateLimitRule, current_time: float, cost: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Token bucket rate limiting with Redis."""
        bucket_key = f"{key}:bucket"
        last_refill_key = f"{key}:last_refill"

        # Get current state
        pipe = self.redis_client.pipeline()
        pipe.get(bucket_key)
        pipe.get(last_refill_key)
        results = await pipe.execute()

        current_tokens = float(results[0] or rule.requests)
        last_refill = float(results[1] or current_time)

        # Calculate tokens to add based on time elapsed
        time_elapsed = current_time - last_refill
        tokens_to_add = time_elapsed * (rule.requests / rule.window)
        new_tokens = min(rule.requests, current_tokens + tokens_to_add)

        allowed = new_tokens >= cost

        if allowed:
            new_tokens -= cost

        # Update state
        pipe = self.redis_client.pipeline()
        pipe.setex(bucket_key, rule.window * 2, str(new_tokens))
        pipe.setex(last_refill_key, rule.window * 2, str(current_time))
        await pipe.execute()

        return allowed, {
            "tokens_available": new_tokens,
            "tokens_capacity": rule.requests,
            "refill_rate": rule.requests / rule.window,
            "cost": cost,
            "rule": rule,
        }

    async def _redis_fixed_window(
        self, key: str, rule: RateLimitRule, current_time: float, cost: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Fixed window rate limiting with Redis."""
        window_key = f"{key}:{int(current_time // rule.window)}"

        # Increment counter
        current_count = await self.redis_client.incr(window_key)

        if current_count == 1:
            # Set expiry for new window
            await self.redis_client.expire(window_key, rule.window)

        allowed = current_count <= rule.requests

        if not allowed:
            # Decrement since request is not allowed
            await self.redis_client.decr(window_key)

        reset_time = (int(current_time // rule.window) + 1) * rule.window

        return allowed, {
            "requests_made": current_count,
            "requests_allowed": rule.requests,
            "window_size": rule.window,
            "reset_time": reset_time,
            "remaining": max(0, rule.requests - current_count),
            "rule": rule,
        }

    async def _check_local_rate_limit(
        self, client_id: str, rule: RateLimitRule, current_time: float, cost: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using local memory storage."""
        state = self.local_state[client_id]

        # Check if client is blocked
        if state.blocked_until and state.blocked_until > current_time:
            remaining_time = state.blocked_until - current_time
            return False, {
                "blocked": True,
                "blocked_until": state.blocked_until,
                "remaining_time": remaining_time,
                "rule": rule,
            }

        # Apply rate limiting strategy
        if rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._local_sliding_window(state, rule, current_time, cost)
        elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._local_token_bucket(state, rule, current_time, cost)
        elif rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            return self._local_fixed_window(state, rule, current_time, cost)
        else:
            return self._local_sliding_window(state, rule, current_time, cost)

    def _local_sliding_window(
        self, state: ClientState, rule: RateLimitRule, current_time: float, cost: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Local sliding window implementation."""
        window_start = current_time - rule.window

        # Remove old requests
        while state.request_times and state.request_times[0] < window_start:
            state.request_times.popleft()

        current_count = len(state.request_times) + cost
        allowed = current_count <= rule.requests

        if allowed:
            # Add request times for this cost
            for _ in range(cost):
                state.request_times.append(current_time)
            state.total_requests += cost
        else:
            state.violations += 1
            # Block client if too many violations
            if state.violations > rule.requests * 0.1:
                state.blocked_until = current_time + rule.block_duration

        # Calculate reset time
        reset_time = (
            state.request_times[0] + rule.window
            if state.request_times
            else current_time + rule.window
        )

        return allowed, {
            "requests_made": current_count,
            "requests_allowed": rule.requests,
            "window_size": rule.window,
            "reset_time": reset_time,
            "remaining": max(0, rule.requests - current_count),
            "rule": rule,
        }

    def _local_token_bucket(
        self, state: ClientState, rule: RateLimitRule, current_time: float, cost: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Local token bucket implementation."""
        # Initialize tokens if first request
        if state.tokens == 0:
            state.tokens = rule.requests
            state.last_refill = current_time

        # Calculate tokens to add
        time_elapsed = current_time - state.last_refill
        tokens_to_add = time_elapsed * (rule.requests / rule.window)
        state.tokens = min(rule.requests, state.tokens + tokens_to_add)
        state.last_refill = current_time

        allowed = state.tokens >= cost

        if allowed:
            state.tokens -= cost
            state.total_requests += cost
        else:
            state.violations += 1

        return allowed, {
            "tokens_available": state.tokens,
            "tokens_capacity": rule.requests,
            "refill_rate": rule.requests / rule.window,
            "cost": cost,
            "rule": rule,
        }

    def _local_fixed_window(
        self, state: ClientState, rule: RateLimitRule, current_time: float, cost: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Local fixed window implementation."""
        window_start = int(current_time // rule.window) * rule.window

        # Reset counter if new window
        if not state.request_times or state.request_times[-1] < window_start:
            state.request_times.clear()

        current_count = len(state.request_times) + cost
        allowed = current_count <= rule.requests

        if allowed:
            for _ in range(cost):
                state.request_times.append(current_time)
            state.total_requests += cost
        else:
            state.violations += 1

        reset_time = window_start + rule.window

        return allowed, {
            "requests_made": current_count,
            "requests_allowed": rule.requests,
            "window_size": rule.window,
            "reset_time": reset_time,
            "remaining": max(0, rule.requests - current_count),
            "rule": rule,
        }

    async def _cleanup_expired_data(self) -> None:
        """Clean up expired local state data."""
        current_time = time.time()
        expired_clients = []

        for client_id, state in self.local_state.items():
            # Remove clients with no recent activity
            if (
                state.request_times
                and current_time - state.request_times[-1] > self.cleanup_interval
            ):
                expired_clients.append(client_id)
            elif state.blocked_until and state.blocked_until < current_time:
                state.blocked_until = None
                state.violations = 0

        for client_id in expired_clients:
            del self.local_state[client_id]

        if expired_clients:
            logger.info(f"Cleaned up {len(expired_clients)} expired rate limit entries")

    async def reset_client(self, client_id: str) -> None:
        """Reset rate limiting state for a specific client."""
        if self.redis_client:
            # Remove all Redis keys for this client
            pattern = f"{self.key_prefix}:{client_id}*"
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
        else:
            # Reset local state
            if client_id in self.local_state:
                del self.local_state[client_id]

        logger.info(f"Reset rate limiting state for client: {client_id}")

    async def get_client_stats(self, client_id: str) -> Dict[str, Any]:
        """Get rate limiting statistics for a client."""
        if client_id in self.local_state:
            state = self.local_state[client_id]
            return {
                "total_requests": state.total_requests,
                "violations": state.violations,
                "blocked_until": state.blocked_until,
                "current_tokens": getattr(state, "tokens", None),
                "active_requests": len(state.request_times),
            }
        return {"total_requests": 0, "violations": 0, "blocked_until": None}


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting."""

    def __init__(
        self,
        rate_limiter: RateLimiter,
        client_id_extractor: Optional[Callable[[Request], str]] = None,
        identifier_extractor: Optional[Callable[[Request], str]] = None,
        cost_extractor: Optional[Callable[[Request], int]] = None,
        exclude_paths: Optional[List[str]] = None,
    ):
        self.rate_limiter = rate_limiter
        self.client_id_extractor = client_id_extractor or self._default_client_id
        self.identifier_extractor = identifier_extractor or self._default_identifier
        self.cost_extractor = cost_extractor or self._default_cost
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/redoc"]

    def _default_client_id(self, request: Request) -> str:
        """Default client ID extraction from IP address."""
        # Try to get real IP from headers (reverse proxy)
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Fallback to direct IP
        return request.client.host if request.client else "unknown"

    def _default_identifier(self, request: Request) -> str:
        """Default identifier extraction from endpoint."""
        return f"{request.method}:{request.url.path}"

    def _default_cost(self, request: Request) -> int:
        """Default cost calculation (always 1)."""
        return 1

    async def __call__(self, request: Request, call_next) -> JSONResponse:
        """Process rate limiting for incoming requests."""
        # Skip rate limiting for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Extract request information
        client_id = self.client_id_extractor(request)
        identifier = self.identifier_extractor(request)
        cost = self.cost_extractor(request)

        # Check rate limit
        allowed, metadata = await self.rate_limiter.is_allowed(
            client_id, identifier, cost
        )

        if not allowed:
            # Add rate limit headers
            headers = {
                "X-RateLimit-Limit": str(metadata.get("requests_allowed", "unknown")),
                "X-RateLimit-Remaining": str(metadata.get("remaining", 0)),
                "X-RateLimit-Reset": str(int(metadata.get("reset_time", 0))),
                "Retry-After": str(int(metadata.get("remaining_time", 60))),
            }

            error_response = {
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please try again later.",
                "details": {
                    "limit": metadata.get("requests_allowed"),
                    "window": metadata.get("window_size"),
                    "reset_time": metadata.get("reset_time"),
                    "blocked": metadata.get("blocked", False),
                },
            }

            status_code = (
                status.HTTP_429_TOO_MANY_REQUESTS
                if not metadata.get("blocked")
                else status.HTTP_403_FORBIDDEN
            )

            logger.warning(
                f"Rate limit exceeded for client {client_id} on {identifier}",
                extra={
                    "client_id": client_id,
                    "identifier": identifier,
                    "metadata": metadata,
                },
            )

            return JSONResponse(
                status_code=status_code, content=error_response, headers=headers
            )

        # Add rate limit info headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(
            metadata.get("requests_allowed", "unknown")
        )
        response.headers["X-RateLimit-Remaining"] = str(metadata.get("remaining", 0))
        response.headers["X-RateLimit-Reset"] = str(int(metadata.get("reset_time", 0)))

        return response


# Global rate limiter instance
_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(redis_client: Optional[redis.Redis] = None) -> RateLimiter:
    """Get the global rate limiter instance."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter(redis_client=redis_client)
    return _global_rate_limiter


# Convenience functions
async def check_rate_limit(
    client_id: str, identifier: str = "default", cost: int = 1
) -> Tuple[bool, Dict[str, Any]]:
    """Check rate limit (convenience function)."""
    rate_limiter = get_rate_limiter()
    return await rate_limiter.is_allowed(client_id, identifier, cost)


async def reset_client_rate_limit(client_id: str) -> None:
    """Reset client rate limit (convenience function)."""
    rate_limiter = get_rate_limiter()
    await rate_limiter.reset_client(client_id)
