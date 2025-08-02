import asyncio
import time
from enum import Enum
from typing import Dict, Any, Callable, Optional
from app.utils.logger import logger


class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Circuit breaker implementation for service resilience"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: Exception = Exception,
        name: str = "CircuitBreaker"
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
    def __call__(self, func):
        """Decorator implementation"""
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker {self.name}: Moving to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.timeout
        )
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info(f"Circuit breaker {self.name}: Moving to CLOSED")
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker {self.name}: Moving to OPEN after {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "timeout": self.timeout
        }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers"""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: Exception = Exception
    ) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                timeout=timeout,
                expected_exception=expected_exception,
                name=name
            )
        return self._breakers[name]
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all circuit breakers"""
        return {name: breaker.get_state() for name, breaker in self._breakers.items()}
    
    def reset_breaker(self, name: str):
        """Reset a specific circuit breaker"""
        if name in self._breakers:
            breaker = self._breakers[name]
            breaker.failure_count = 0
            breaker.state = CircuitState.CLOSED
            breaker.last_failure_time = None
            logger.info(f"Circuit breaker {name} has been reset")


# Global registry instance
circuit_registry = CircuitBreakerRegistry()


# Predefined circuit breakers for common services
auth_service_breaker = circuit_registry.get_breaker(
    "auth_service",
    failure_threshold=3,
    timeout=30,
    expected_exception=Exception
)

rabbitmq_breaker = circuit_registry.get_breaker(
    "rabbitmq_service",
    failure_threshold=5,
    timeout=60,
    expected_exception=Exception
)

database_breaker = circuit_registry.get_breaker(
    "database",
    failure_threshold=10,
    timeout=30,
    expected_exception=Exception
)

redis_breaker = circuit_registry.get_breaker(
    "redis_cache",
    failure_threshold=5,
    timeout=30,
    expected_exception=Exception
)