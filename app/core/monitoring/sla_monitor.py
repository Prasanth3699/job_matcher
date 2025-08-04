"""
SLA (Service Level Agreement) monitoring system.
Tracks performance metrics against defined SLA targets and triggers alerts.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import statistics

from app.utils.logger import logger
from .metrics_collector import get_metrics_collector
from .correlation_tracker import get_current_correlation_id


class SLAMetricType(str, Enum):
    """Types of SLA metrics."""
    AVAILABILITY = "availability"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    SUCCESS_RATE = "success_rate"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SLATarget:
    """SLA target definition."""
    name: str
    metric_type: SLAMetricType
    target_value: float
    comparison: str  # ">=", "<=", ">", "<", "=="
    time_window_minutes: int = 60
    description: str = ""
    alert_threshold: float = 0.05  # Alert when within 5% of breach
    
    def is_breached(self, current_value: float) -> bool:
        """Check if current value breaches the SLA target."""
        if self.comparison == ">=":
            return current_value < self.target_value
        elif self.comparison == "<=":
            return current_value > self.target_value
        elif self.comparison == ">":
            return current_value <= self.target_value
        elif self.comparison == "<":
            return current_value >= self.target_value
        elif self.comparison == "==":
            return current_value != self.target_value
        return False
    
    def is_at_risk(self, current_value: float) -> bool:
        """Check if current value is at risk of breaching SLA."""
        if self.comparison in [">=", ">"]:
            threshold = self.target_value * (1 + self.alert_threshold)
            return current_value < threshold
        elif self.comparison in ["<=", "<"]:
            threshold = self.target_value * (1 - self.alert_threshold)
            return current_value > threshold
        return False


@dataclass
class SLAMeasurement:
    """Single SLA measurement."""
    target_name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SLAAlert:
    """SLA alert."""
    target_name: str
    severity: AlertSeverity
    message: str
    current_value: float
    target_value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "target_name": self.target_name,
            "severity": self.severity.value,
            "message": self.message,
            "current_value": self.current_value,
            "target_value": self.target_value,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }


@dataclass
class SLAReport:
    """SLA compliance report."""
    target_name: str
    metric_type: SLAMetricType
    target_value: float
    current_value: float
    compliance_percentage: float
    is_breached: bool
    is_at_risk: bool
    measurements_count: int
    time_window_minutes: int
    report_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "target_name": self.target_name,
            "metric_type": self.metric_type.value,
            "target_value": self.target_value,
            "current_value": self.current_value,
            "compliance_percentage": self.compliance_percentage,
            "is_breached": self.is_breached,
            "is_at_risk": self.is_at_risk,
            "measurements_count": self.measurements_count,
            "time_window_minutes": self.time_window_minutes,
            "report_timestamp": self.report_timestamp.isoformat(),
        }


class SLAMonitor:
    """
    Service Level Agreement monitoring system.
    Tracks performance metrics against SLA targets and provides alerting.
    """
    
    def __init__(self, max_measurements: int = 100000):
        self.max_measurements = max_measurements
        self._lock = threading.RLock()
        
        # SLA targets and measurements
        self._sla_targets: Dict[str, SLATarget] = {}
        self._measurements: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_measurements))
        
        # Alerts
        self._alerts: deque = deque(maxlen=10000)
        self._alert_callbacks: List[Callable[[SLAAlert], None]] = []
        
        # Compliance tracking
        self._compliance_history: Dict[str, List[float]] = defaultdict(list)
        
        # Metrics collector
        self.metrics_collector = get_metrics_collector()
        
        # Add default SLA targets
        self._add_default_sla_targets()
        
        logger.info("SLA monitor initialized")
    
    def _add_default_sla_targets(self):
        """Add default SLA targets based on common requirements."""
        try:
            from app.config.config_validator import get_config
            config = get_config()
            
            # Default SLA targets
            default_targets = [
                SLATarget(
                    name="api_availability",
                    metric_type=SLAMetricType.AVAILABILITY,
                    target_value=99.9,  # 99.9% availability
                    comparison=">=",
                    time_window_minutes=60,
                    description="API availability SLA"
                ),
                SLATarget(
                    name="api_response_time",
                    metric_type=SLAMetricType.RESPONSE_TIME,
                    target_value=200.0,  # 200ms average response time
                    comparison="<=",
                    time_window_minutes=15,
                    description="API response time SLA"
                ),
                SLATarget(
                    name="api_error_rate",
                    metric_type=SLAMetricType.ERROR_RATE,
                    target_value=1.0,  # 1% error rate
                    comparison="<=",
                    time_window_minutes=30,
                    description="API error rate SLA"
                ),
                SLATarget(
                    name="match_processing_time",
                    metric_type=SLAMetricType.RESPONSE_TIME,
                    target_value=30.0,  # 30 seconds for match processing
                    comparison="<=",
                    time_window_minutes=60,
                    description="Match processing time SLA"
                ),
                SLATarget(
                    name="match_success_rate",
                    metric_type=SLAMetricType.SUCCESS_RATE,
                    target_value=95.0,  # 95% success rate
                    comparison=">=",
                    time_window_minutes=60,
                    description="Match processing success rate SLA"
                ),
            ]
            
            # Add production-specific targets if in production
            if config.is_production():
                default_targets.extend([
                    SLATarget(
                        name="production_availability",
                        metric_type=SLAMetricType.AVAILABILITY,
                        target_value=99.95,  # Higher availability for production
                        comparison=">=",
                        time_window_minutes=60,
                        description="Production availability SLA"
                    ),
                    SLATarget(
                        name="production_response_time",
                        metric_type=SLAMetricType.RESPONSE_TIME,
                        target_value=150.0,  # Stricter response time
                        comparison="<=",
                        time_window_minutes=15,
                        description="Production response time SLA"
                    ),
                ])
            
            for target in default_targets:
                self.add_sla_target(target)
                
        except Exception as e:
            logger.warning(f"Failed to add some default SLA targets: {e}")
    
    def add_sla_target(self, target: SLATarget):
        """Add an SLA target to monitor."""
        with self._lock:
            self._sla_targets[target.name] = target
        
        logger.info(f"Added SLA target: {target.name} ({target.metric_type.value} {target.comparison} {target.target_value})")
    
    def remove_sla_target(self, name: str):
        """Remove an SLA target."""
        with self._lock:
            self._sla_targets.pop(name, None)
            self._measurements.pop(name, None)
            self._compliance_history.pop(name, None)
        
        logger.info(f"Removed SLA target: {name}")
    
    def record_measurement(
        self,
        target_name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a measurement for an SLA target."""
        
        if target_name not in self._sla_targets:
            logger.warning(f"Unknown SLA target: {target_name}")
            return
        
        measurement = SLAMeasurement(
            target_name=target_name,
            value=value,
            correlation_id=get_current_correlation_id(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self._measurements[target_name].append(measurement)
        
        # Update Prometheus metrics
        self.metrics_collector.observe_histogram(
            "sla_measurement",
            value,
            labels={"target_name": target_name}
        )
        
        # Check for SLA breaches
        self._check_sla_compliance(target_name)
    
    def _check_sla_compliance(self, target_name: str):
        """Check SLA compliance for a target."""
        target = self._sla_targets.get(target_name)
        if not target:
            return
        
        # Calculate current metric value
        current_value = self._calculate_current_value(target_name, target.time_window_minutes)
        if current_value is None:
            return
        
        # Check for breach
        is_breached = target.is_breached(current_value)
        is_at_risk = target.is_at_risk(current_value)
        
        # Generate alerts
        if is_breached:
            alert = SLAAlert(
                target_name=target_name,
                severity=AlertSeverity.CRITICAL,
                message=f"SLA breach: {target.description} - Current: {current_value:.2f}, Target: {target.comparison} {target.target_value}",
                current_value=current_value,
                target_value=target.target_value,
                correlation_id=get_current_correlation_id(),
                metadata={"metric_type": target.metric_type.value}
            )
            self._trigger_alert(alert)
            
        elif is_at_risk:
            alert = SLAAlert(
                target_name=target_name,
                severity=AlertSeverity.HIGH,
                message=f"SLA at risk: {target.description} - Current: {current_value:.2f}, Target: {target.comparison} {target.target_value}",
                current_value=current_value,
                target_value=target.target_value,
                correlation_id=get_current_correlation_id(),
                metadata={"metric_type": target.metric_type.value}
            )
            self._trigger_alert(alert)
        
        # Update compliance metrics
        compliance_percentage = self._calculate_compliance_percentage(target_name, target.time_window_minutes)
        
        self.metrics_collector.set_gauge(
            "sla_compliance_percentage",
            compliance_percentage,
            labels={"target_name": target_name}
        )
        
        self.metrics_collector.set_gauge(
            "sla_current_value",
            current_value,
            labels={"target_name": target_name}
        )
        
        # Track compliance history
        self._compliance_history[target_name].append(compliance_percentage)
        if len(self._compliance_history[target_name]) > 1000:
            self._compliance_history[target_name] = self._compliance_history[target_name][-1000:]
    
    def _calculate_current_value(self, target_name: str, time_window_minutes: int) -> Optional[float]:
        """Calculate current metric value for the time window."""
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        with self._lock:
            measurements = self._measurements.get(target_name, deque())
            recent_measurements = [
                m for m in measurements
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_measurements:
            return None
        
        values = [m.value for m in recent_measurements]
        target = self._sla_targets[target_name]
        
        # Calculate based on metric type
        if target.metric_type == SLAMetricType.AVAILABILITY:
            # Availability: percentage of successful measurements
            successful = sum(1 for v in values if v == 1.0)  # 1.0 = success, 0.0 = failure
            return (successful / len(values)) * 100.0
            
        elif target.metric_type == SLAMetricType.RESPONSE_TIME:
            # Response time: average or percentile
            return statistics.mean(values)
            
        elif target.metric_type == SLAMetricType.ERROR_RATE:
            # Error rate: percentage of error measurements
            errors = sum(1 for v in values if v == 1.0)  # 1.0 = error, 0.0 = success
            return (errors / len(values)) * 100.0
            
        elif target.metric_type == SLAMetricType.THROUGHPUT:
            # Throughput: requests per minute
            return len(values) / time_window_minutes
            
        elif target.metric_type == SLAMetricType.SUCCESS_RATE:
            # Success rate: percentage of successful measurements
            successful = sum(1 for v in values if v == 1.0)  # 1.0 = success, 0.0 = failure
            return (successful / len(values)) * 100.0
        
        # Default: average
        return statistics.mean(values)
    
    def _calculate_compliance_percentage(self, target_name: str, time_window_minutes: int) -> float:
        """Calculate compliance percentage for the time window."""
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        target = self._sla_targets[target_name]
        
        with self._lock:
            measurements = self._measurements.get(target_name, deque())
            recent_measurements = [
                m for m in measurements
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_measurements:
            return 100.0  # No data = compliant
        
        # Check compliance for each measurement
        compliant_count = 0
        for measurement in recent_measurements:
            if not target.is_breached(measurement.value):
                compliant_count += 1
        
        return (compliant_count / len(recent_measurements)) * 100.0
    
    def _trigger_alert(self, alert: SLAAlert):
        """Trigger an SLA alert."""
        
        with self._lock:
            self._alerts.append(alert)
        
        # Log alert
        logger.error(
            f"SLA Alert: {alert.message}",
            extra={
                "alert_severity": alert.severity.value,
                "target_name": alert.target_name,
                "current_value": alert.current_value,
                "target_value": alert.target_value,
                "correlation_id": alert.correlation_id
            }
        )
        
        # Update alert metrics
        self.metrics_collector.increment_counter(
            "sla_alerts_total",
            labels={
                "target_name": alert.target_name,
                "severity": alert.severity.value
            }
        )
        
        # Call alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[SLAAlert], None]):
        """Add a callback function for alerts."""
        self._alert_callbacks.append(callback)
        logger.info(f"Added SLA alert callback: {callback.__name__}")
    
    def get_sla_report(self, target_name: Optional[str] = None) -> List[SLAReport]:
        """Get SLA compliance report."""
        reports = []
        
        targets_to_check = [target_name] if target_name else list(self._sla_targets.keys())
        
        for name in targets_to_check:
            if name not in self._sla_targets:
                continue
            
            target = self._sla_targets[name]
            current_value = self._calculate_current_value(name, target.time_window_minutes)
            
            if current_value is None:
                continue
            
            compliance_percentage = self._calculate_compliance_percentage(name, target.time_window_minutes)
            
            with self._lock:
                measurements_count = len([
                    m for m in self._measurements.get(name, deque())
                    if m.timestamp >= datetime.utcnow() - timedelta(minutes=target.time_window_minutes)
                ])
            
            report = SLAReport(
                target_name=name,
                metric_type=target.metric_type,
                target_value=target.target_value,
                current_value=current_value,
                compliance_percentage=compliance_percentage,
                is_breached=target.is_breached(current_value),
                is_at_risk=target.is_at_risk(current_value),
                measurements_count=measurements_count,
                time_window_minutes=target.time_window_minutes
            )
            
            reports.append(report)
        
        return reports
    
    def get_recent_alerts(self, hours: int = 24) -> List[SLAAlert]:
        """Get recent alerts within the specified time window."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self._lock:
            return [
                alert for alert in self._alerts
                if alert.timestamp >= cutoff_time
            ]
    
    def get_sla_summary(self) -> Dict[str, Any]:
        """Get overall SLA summary."""
        reports = self.get_sla_report()
        recent_alerts = self.get_recent_alerts(24)
        
        # Calculate overall compliance
        if reports:
            overall_compliance = statistics.mean([r.compliance_percentage for r in reports])
            breaches = sum(1 for r in reports if r.is_breached)
            at_risk = sum(1 for r in reports if r.is_at_risk)
        else:
            overall_compliance = 100.0
            breaches = 0
            at_risk = 0
        
        # Alert counts by severity
        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[alert.severity.value] += 1
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_compliance_percentage": overall_compliance,
            "total_sla_targets": len(self._sla_targets),
            "targets_breached": breaches,
            "targets_at_risk": at_risk,
            "alerts_24h": len(recent_alerts),
            "alert_counts_24h": dict(alert_counts),
            "sla_targets": [
                {
                    "name": target.name,
                    "metric_type": target.metric_type.value,
                    "target_value": target.target_value,
                    "comparison": target.comparison,
                    "description": target.description
                }
                for target in self._sla_targets.values()
            ]
        }


# Convenience functions for common SLA measurements
def record_api_availability(is_successful: bool):
    """Record API availability measurement."""
    monitor = get_sla_monitor()
    monitor.record_measurement("api_availability", 1.0 if is_successful else 0.0)


def record_api_response_time(response_time_ms: float):
    """Record API response time measurement."""
    monitor = get_sla_monitor()
    monitor.record_measurement("api_response_time", response_time_ms)


def record_api_error(is_error: bool):
    """Record API error measurement."""
    monitor = get_sla_monitor()
    monitor.record_measurement("api_error_rate", 1.0 if is_error else 0.0)


def record_match_processing_time(processing_time_seconds: float):
    """Record match processing time measurement."""
    monitor = get_sla_monitor()
    monitor.record_measurement("match_processing_time", processing_time_seconds)


def record_match_success(is_successful: bool):
    """Record match processing success measurement."""
    monitor = get_sla_monitor()
    monitor.record_measurement("match_success_rate", 1.0 if is_successful else 0.0)


# Global SLA monitor instance
_global_sla_monitor: Optional[SLAMonitor] = None


def get_sla_monitor() -> SLAMonitor:
    """Get the global SLA monitor instance."""
    global _global_sla_monitor
    if _global_sla_monitor is None:
        _global_sla_monitor = SLAMonitor()
    return _global_sla_monitor


def initialize_sla_monitor(max_measurements: int = 100000) -> SLAMonitor:
    """Initialize the global SLA monitor."""
    global _global_sla_monitor
    _global_sla_monitor = SLAMonitor(max_measurements=max_measurements)
    return _global_sla_monitor