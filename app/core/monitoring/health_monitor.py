"""
Comprehensive health monitoring system for the application and its dependencies.
Provides health checks, dependency monitoring, and automated recovery mechanisms.
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Callable, NamedTuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import socket

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from app.utils.logger import logger
from .metrics_collector import get_metrics_collector


class HealthStatus(str, Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    response_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "response_time_ms": self.response_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    load_average: Optional[List[float]] = None
    network_io: Optional[Dict[str, int]] = None
    process_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "disk_percent": self.disk_percent,
            "load_average": self.load_average,
            "network_io": self.network_io,
            "process_count": self.process_count,
        }


class HealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, timeout: float = 5.0):
        self.name = name
        self.timeout = timeout
    
    async def check(self) -> HealthCheckResult:
        """Perform the health check."""
        raise NotImplementedError
    
    def __str__(self):
        return f"HealthCheck(name={self.name})"


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity."""
    
    def __init__(self, name: str = "database", timeout: float = 5.0):
        super().__init__(name, timeout)
    
    async def check(self) -> HealthCheckResult:
        """Check database health."""
        start_time = time.time()
        
        try:
            from app.db.session import get_db
            
            # Get database session and test connectivity
            db = next(get_db())
            
            # Simple query to test connectivity
            result = db.execute("SELECT 1").scalar()
            
            if result == 1:
                response_time = (time.time() - start_time) * 1000
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Database connection successful",
                    response_time_ms=response_time,
                    metadata={"query_result": result}
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Database query returned unexpected result",
                    response_time_ms=(time.time() - start_time) * 1000,
                    metadata={"query_result": result}
                )
                
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e), "error_type": type(e).__name__}
            )


class RedisHealthCheck(HealthCheck):
    """Health check for Redis connectivity."""
    
    def __init__(self, name: str = "redis", timeout: float = 3.0):
        super().__init__(name, timeout)
    
    async def check(self) -> HealthCheckResult:
        """Check Redis health."""
        start_time = time.time()
        
        try:
            import redis
            from app.config.config_validator import get_config
            
            config = get_config()
            redis_client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                password=config.REDIS_PASSWORD,
                db=config.REDIS_DB,
                socket_timeout=self.timeout
            )
            
            # Test connectivity with ping
            result = redis_client.ping()
            
            if result:
                response_time = (time.time() - start_time) * 1000
                
                # Get additional Redis info
                info = redis_client.info()
                
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Redis connection successful",
                    response_time_ms=response_time,
                    metadata={
                        "redis_version": info.get("redis_version"),
                        "connected_clients": info.get("connected_clients"),
                        "used_memory": info.get("used_memory"),
                        "used_memory_human": info.get("used_memory_human"),
                    }
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Redis ping failed",
                    response_time_ms=(time.time() - start_time) * 1000
                )
                
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e), "error_type": type(e).__name__}
            )


class ServiceHealthCheck(HealthCheck):
    """Health check for external service connectivity."""
    
    def __init__(self, name: str, url: str, timeout: float = 10.0):
        super().__init__(name, timeout)
        self.url = url
    
    async def check(self) -> HealthCheckResult:
        """Check service health via HTTP."""
        start_time = time.time()
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(f"{self.url}/health") as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        try:
                            data = await response.json()
                        except:
                            data = {"status": "ok"}
                        
                        return HealthCheckResult(
                            name=self.name,
                            status=HealthStatus.HEALTHY,
                            message=f"Service {self.name} is healthy",
                            response_time_ms=response_time,
                            metadata={
                                "url": self.url,
                                "status_code": response.status,
                                "response_data": data
                            }
                        )
                    else:
                        return HealthCheckResult(
                            name=self.name,
                            status=HealthStatus.UNHEALTHY,
                            message=f"Service {self.name} returned status {response.status}",
                            response_time_ms=response_time,
                            metadata={
                                "url": self.url,
                                "status_code": response.status
                            }
                        )
                        
        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Service {self.name} timeout",
                response_time_ms=(time.time() - start_time) * 1000,
                metadata={"url": self.url, "error": "timeout"}
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Service {self.name} connection failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    "url": self.url,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )


class CeleryHealthCheck(HealthCheck):
    """Health check for Celery task queue."""
    
    def __init__(self, name: str = "celery", timeout: float = 5.0):
        super().__init__(name, timeout)
    
    async def check(self) -> HealthCheckResult:
        """Check Celery health."""
        start_time = time.time()
        
        try:
            from app.services.task_queue_domain import app as celery_app
            
            # Inspect active workers
            inspect = celery_app.control.inspect()
            
            # Get worker statistics
            stats = inspect.stats()
            active_tasks = inspect.active()
            
            if stats:
                worker_count = len(stats)
                total_active_tasks = sum(len(tasks) for tasks in active_tasks.values()) if active_tasks else 0
                
                response_time = (time.time() - start_time) * 1000
                
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message=f"Celery is healthy with {worker_count} workers",
                    response_time_ms=response_time,
                    metadata={
                        "worker_count": worker_count,
                        "active_tasks": total_active_tasks,
                        "workers": list(stats.keys()) if stats else []
                    }
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="No Celery workers found",
                    response_time_ms=(time.time() - start_time) * 1000,
                    metadata={"worker_count": 0}
                )
                
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Celery health check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e), "error_type": type(e).__name__}
            )


class HealthMonitor:
    """
    Comprehensive health monitoring system.
    Orchestrates multiple health checks and provides aggregated health status.
    """
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self._health_checks: List[HealthCheck] = []
        self._last_results: Dict[str, HealthCheckResult] = {}
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Metrics collector
        self.metrics_collector = get_metrics_collector()
        
        # Add default health checks
        self._add_default_health_checks()
        
        logger.info("Health monitor initialized")
    
    def _add_default_health_checks(self):
        """Add default health checks."""
        try:
            # Database health check
            self.add_health_check(DatabaseHealthCheck())
            
            # Redis health check
            self.add_health_check(RedisHealthCheck())
            
            # Celery health check
            self.add_health_check(CeleryHealthCheck())
            
            # External service health checks
            from app.config.config_validator import get_config
            config = get_config()
            
            if config.JOBS_SERVICE_URL:
                self.add_health_check(ServiceHealthCheck("jobs_service", config.JOBS_SERVICE_URL))
            
            if config.AUTH_SERVICE_URL:
                self.add_health_check(ServiceHealthCheck("auth_service", config.AUTH_SERVICE_URL))
            
        except Exception as e:
            logger.warning(f"Failed to add some default health checks: {e}")
    
    def add_health_check(self, health_check: HealthCheck):
        """Add a health check to the monitor."""
        with self._lock:
            self._health_checks.append(health_check)
        logger.info(f"Added health check: {health_check.name}")
    
    def remove_health_check(self, name: str):
        """Remove a health check by name."""
        with self._lock:
            self._health_checks = [hc for hc in self._health_checks if hc.name != name]
            self._last_results.pop(name, None)
        logger.info(f"Removed health check: {name}")
    
    async def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks and return results."""
        results = {}
        
        # Run health checks concurrently
        tasks = []
        for health_check in self._health_checks:
            task = asyncio.create_task(health_check.check())
            tasks.append((health_check.name, task))
        
        # Wait for all tasks to complete
        for name, task in tasks:
            try:
                result = await task
                results[name] = result
                
                # Update metrics
                self.metrics_collector.observe_histogram(
                    "health_check_duration_seconds",
                    (result.response_time_ms or 0) / 1000,
                    labels={"check_name": name}
                )
                
                self.metrics_collector.increment_counter(
                    "health_check_total",
                    labels={
                        "check_name": name,
                        "status": result.status.value
                    }
                )
                
            except Exception as e:
                logger.error(f"Health check {name} failed with exception: {e}")
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check failed: {str(e)}",
                    metadata={"error": str(e), "error_type": type(e).__name__}
                )
        
        # Update last results
        with self._lock:
            self._last_results.update(results)
        
        return results
    
    def get_overall_health(self) -> HealthCheckResult:
        """Get overall system health status."""
        with self._lock:
            if not self._last_results:
                return HealthCheckResult(
                    name="overall",
                    status=HealthStatus.UNKNOWN,
                    message="No health check results available"
                )
            
            # Analyze overall health
            healthy_count = sum(1 for result in self._last_results.values() if result.status == HealthStatus.HEALTHY)
            degraded_count = sum(1 for result in self._last_results.values() if result.status == HealthStatus.DEGRADED)
            unhealthy_count = sum(1 for result in self._last_results.values() if result.status == HealthStatus.UNHEALTHY)
            total_checks = len(self._last_results)
            
            # Determine overall status
            if unhealthy_count > 0:
                status = HealthStatus.UNHEALTHY
                message = f"{unhealthy_count} components unhealthy"
            elif degraded_count > 0:
                status = HealthStatus.DEGRADED
                message = f"{degraded_count} components degraded"
            elif healthy_count == total_checks:
                status = HealthStatus.HEALTHY
                message = "All components healthy"
            else:
                status = HealthStatus.UNKNOWN
                message = "Unknown health status"
            
            return HealthCheckResult(
                name="overall",
                status=status,
                message=message,
                metadata={
                    "total_checks": total_checks,
                    "healthy": healthy_count,
                    "degraded": degraded_count,
                    "unhealthy": unhealthy_count,
                    "health_percentage": (healthy_count / total_checks) * 100 if total_checks > 0 else 0
                }
            )
    
    def get_system_metrics(self) -> Optional[SystemMetrics]:
        """Get system resource metrics."""
        if not PSUTIL_AVAILABLE:
            return None
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Load average (Unix systems only)
            load_avg = None
            try:
                load_avg = list(psutil.getloadavg())
            except AttributeError:
                pass  # Windows doesn't have load average
            
            # Network I/O
            network_io = None
            try:
                net_io = psutil.net_io_counters()
                network_io = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                }
            except Exception:
                pass
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                load_average=load_avg,
                network_io=network_io,
                process_count=len(psutil.pids())
            )
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return None
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._monitoring:
            logger.warning("Health monitoring is already running")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info(f"Started health monitoring with {self.check_interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        logger.info("Stopped health monitoring")
    
    def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while self._monitoring:
            try:
                # Run health checks
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                results = loop.run_until_complete(self.run_health_checks())
                overall_health = self.get_overall_health()
                
                # Log overall health status
                logger.info(
                    f"Health check completed: {overall_health.status.value}",
                    extra={
                        "overall_status": overall_health.status.value,
                        "healthy_checks": sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY),
                        "total_checks": len(results)
                    }
                )
                
                # Update system metrics if available
                system_metrics = self.get_system_metrics()
                if system_metrics:
                    self.metrics_collector.set_gauge("system_cpu_percent", system_metrics.cpu_percent)
                    self.metrics_collector.set_gauge("system_memory_percent", system_metrics.memory_percent)
                    self.metrics_collector.set_gauge("system_disk_percent", system_metrics.disk_percent)
                
                loop.close()
                
            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}", exc_info=True)
            
            # Wait for next check
            time.sleep(self.check_interval)
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        overall_health = self.get_overall_health()
        system_metrics = self.get_system_metrics()
        
        with self._lock:
            component_health = {
                name: result.to_dict()
                for name, result in self._last_results.items()
            }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_health": overall_health.to_dict(),
            "component_health": component_health,
            "system_metrics": system_metrics.to_dict() if system_metrics else None,
            "monitoring_active": self._monitoring,
            "check_interval": self.check_interval,
        }


# Global health monitor instance
_global_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = HealthMonitor()
    return _global_health_monitor


def initialize_health_monitor(check_interval: int = 30, start_monitoring: bool = True) -> HealthMonitor:
    """Initialize the global health monitor."""
    global _global_health_monitor
    _global_health_monitor = HealthMonitor(check_interval=check_interval)
    
    if start_monitoring:
        _global_health_monitor.start_monitoring()
    
    return _global_health_monitor