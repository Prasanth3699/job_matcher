"""
Enterprise metrics collection system using Prometheus metrics.
Provides comprehensive application monitoring and observability.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from contextlib import contextmanager

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from app.utils.logger import logger
from .correlation_tracker import get_current_correlation_id


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: datetime
    correlation_id: Optional[str] = None


class MetricsCollector:
    """
    Enterprise metrics collection system.
    Collects and exposes application metrics for monitoring and alerting.
    """
    
    def __init__(self, namespace: str = "resume_matcher", enable_prometheus: bool = True):
        self.namespace = namespace
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self._lock = threading.RLock()
        
        # Initialize Prometheus registry if available
        if self.enable_prometheus:
            self.registry = CollectorRegistry()
            self._init_prometheus_metrics()
        
        # Fallback in-memory metrics storage
        self._metrics_buffer: deque = deque(maxlen=10000)
        self._counters: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._gauges: Dict[str, float] = {}
        
        logger.info(f"MetricsCollector initialized with namespace '{namespace}', Prometheus: {self.enable_prometheus}")
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        if not self.enable_prometheus:
            return
        
        # Request metrics
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # Business metrics
        self.match_requests = Counter(
            'match_requests_total',
            'Total match requests',
            ['user_type', 'match_type'],
            registry=self.registry
        )
        
        self.match_duration = Histogram(
            'match_processing_duration_seconds',
            'Match processing duration in seconds',
            ['match_type', 'status'],
            buckets=[1, 5, 10, 30, 60, 300, 600],
            registry=self.registry
        )
        
        self.match_quality = Histogram(
            'match_quality_score',
            'Match quality scores',
            ['match_type'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        # ML model metrics
        self.ml_model_predictions = Counter(
            'ml_model_predictions_total',
            'Total ML model predictions',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        self.ml_model_latency = Histogram(
            'ml_model_prediction_duration_seconds',
            'ML model prediction duration',
            ['model_name'],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        self.ml_model_accuracy = Gauge(
            'ml_model_accuracy_score',
            'ML model accuracy score',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        # System metrics
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.queue_size = Gauge(
            'task_queue_size',
            'Task queue size',
            ['queue_name'],
            registry=self.registry
        )
        
        self.database_connections = Gauge(
            'database_connections_active',
            'Active database connections',
            ['pool_name'],
            registry=self.registry
        )
        
        # Error metrics
        self.error_count = Counter(
            'errors_total',
            'Total errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_name'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_name'],
            registry=self.registry
        )
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        labels = labels or {}
        correlation_id = get_current_correlation_id()
        
        with self._lock:
            # Store in buffer
            self._metrics_buffer.append(MetricPoint(
                name=name,
                value=value,
                labels=labels,
                timestamp=datetime.utcnow(),
                correlation_id=correlation_id
            ))
            
            # Update in-memory counter
            key = f"{name}:{':'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
            self._counters[key] += value
        
        # Update Prometheus if available
        if self.enable_prometheus and hasattr(self, name):
            metric = getattr(self, name)
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value in a histogram metric."""
        labels = labels or {}
        correlation_id = get_current_correlation_id()
        
        with self._lock:
            # Store in buffer
            self._metrics_buffer.append(MetricPoint(
                name=name,
                value=value,
                labels=labels,
                timestamp=datetime.utcnow(),
                correlation_id=correlation_id
            ))
            
            # Update in-memory histogram
            key = f"{name}:{':'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
            self._histograms[key].append(value)
            
            # Keep only last 1000 values
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]
        
        # Update Prometheus if available
        if self.enable_prometheus and hasattr(self, name):
            metric = getattr(self, name)
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        labels = labels or {}
        correlation_id = get_current_correlation_id()
        
        with self._lock:
            # Store in buffer
            self._metrics_buffer.append(MetricPoint(
                name=name,
                value=value,
                labels=labels,
                timestamp=datetime.utcnow(),
                correlation_id=correlation_id
            ))
            
            # Update in-memory gauge
            key = f"{name}:{':'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
            self._gauges[key] = value
        
        # Update Prometheus if available
        if self.enable_prometheus and hasattr(self, name):
            metric = getattr(self, name)
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
    
    @contextmanager
    def time_operation(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        
        try:
            yield
        except Exception as e:
            # Record error
            error_labels = (labels or {}).copy()
            error_labels.update({"operation": operation_name, "error_type": type(e).__name__})
            self.increment_counter("operation_errors", labels=error_labels)
            raise
        finally:
            # Record duration
            duration = time.time() - start_time
            duration_labels = (labels or {}).copy()
            duration_labels["operation"] = operation_name
            self.observe_histogram("operation_duration_seconds", duration, duration_labels)
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics."""
        if not self.enable_prometheus:
            return "# Prometheus not available\n"
        
        return generate_latest(self.registry).decode('utf-8')
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        with self._lock:
            return {
                "metrics_collected": len(self._metrics_buffer),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histogram_counts": {k: len(v) for k, v in self._histograms.items()},
                "prometheus_enabled": self.enable_prometheus,
                "buffer_size": len(self._metrics_buffer),
            }
    
    def get_recent_metrics(self, minutes: int = 5) -> List[MetricPoint]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        with self._lock:
            return [
                metric for metric in self._metrics_buffer
                if metric.timestamp >= cutoff_time
            ]
    
    def clear_metrics(self):
        """Clear all metrics (useful for testing)."""
        with self._lock:
            self._metrics_buffer.clear()
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()


# Decorator for automatic method timing
def timed_method(metric_name: str = None, labels_func: Callable = None):
    """Decorator to automatically time method execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Determine metric name
            name = metric_name or f"{func.__module__}.{func.__name__}_duration"
            
            # Get labels if function provided
            labels = {}
            if labels_func:
                try:
                    labels = labels_func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Failed to get labels for {func.__name__}: {e}")
            
            # Time the operation
            collector = get_metrics_collector()
            with collector.time_operation(name, labels):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def track_counter(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to increment a counter when function is called."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            collector.increment_counter(metric_name, labels=labels)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def initialize_metrics(namespace: str = "resume_matcher", enable_prometheus: bool = True) -> MetricsCollector:
    """Initialize the global metrics collector."""
    global _global_collector
    _global_collector = MetricsCollector(namespace=namespace, enable_prometheus=enable_prometheus)
    return _global_collector