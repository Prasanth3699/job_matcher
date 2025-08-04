"""
Resource Manager for system resource monitoring and optimization.

This module provides intelligent resource management including memory, CPU,
GPU monitoring, automatic scaling, and resource allocation optimization.
"""

import asyncio
import psutil
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import deque

from app.utils.logger import logger


class ResourceType(str, Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"


class ResourceStatus(str, Enum):
    """Resource status levels."""
    OPTIMAL = "optimal"
    WARNING = "warning"
    CRITICAL = "critical"
    OVERLOADED = "overloaded"


class ScalingAction(str, Enum):
    """Scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    THROTTLE = "throttle"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    gpu_utilization: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    load_average: Optional[Tuple[float, float, float]] = None
    active_connections: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ResourceThreshold:
    """Resource threshold configuration."""
    resource_type: ResourceType
    warning_threshold: float
    critical_threshold: float
    overload_threshold: float
    scaling_threshold: float
    cooldown_seconds: int = 300


@dataclass
class ResourceAlert:
    """Resource alert information."""
    resource_type: ResourceType
    status: ResourceStatus
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    acknowledged: bool = False


@dataclass
class ScalingEvent:
    """Resource scaling event."""
    action: ScalingAction
    resource_type: ResourceType
    trigger_value: float
    target_value: float
    timestamp: datetime
    success: bool
    details: Dict[str, Any]


class ResourceManager:
    """
    Advanced resource manager for monitoring system resources,
    detecting bottlenecks, and implementing automatic scaling strategies.
    """
    
    def __init__(
        self,
        monitoring_interval_seconds: float = 10.0,
        history_retention_minutes: int = 60,
        enable_auto_scaling: bool = True,
        enable_gpu_monitoring: bool = False
    ):
        """Initialize resource manager."""
        self.monitoring_interval_seconds = monitoring_interval_seconds
        self.history_retention_minutes = history_retention_minutes
        self.enable_auto_scaling = enable_auto_scaling
        self.enable_gpu_monitoring = enable_gpu_monitoring
        
        # Resource monitoring
        self.current_metrics: Optional[ResourceMetrics] = None
        self.metrics_history: deque = deque(maxlen=int(history_retention_minutes * 60 / monitoring_interval_seconds))
        
        # Thresholds and configuration
        self.thresholds: Dict[ResourceType, ResourceThreshold] = {}
        self._setup_default_thresholds()
        
        # Alerts and events
        self.active_alerts: Dict[str, ResourceAlert] = {}
        self.alert_history: List[ResourceAlert] = []
        self.scaling_events: List[ScalingEvent] = []
        
        # Monitoring tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.scaling_task: Optional[asyncio.Task] = None
        
        # Scaling callbacks
        self.scaling_callbacks: Dict[ScalingAction, List[Callable]] = {
            action: [] for action in ScalingAction
        }
        
        # Performance tracking
        self.performance_stats = {
            'monitoring_cycles': 0,
            'alerts_triggered': 0,
            'scaling_actions': 0,
            'average_cpu_usage': 0.0,
            'average_memory_usage': 0.0,
            'peak_usage': {
                'cpu': 0.0,
                'memory': 0.0,
                'disk': 0.0
            },
            'uptime_seconds': 0.0
        }
        
        # GPU monitoring setup
        self.gpu_available = False
        if self.enable_gpu_monitoring:
            self._setup_gpu_monitoring()
        
        self.start_time = time.time()
        
        logger.info("ResourceManager initialized")
    
    def _setup_default_thresholds(self):
        """Setup default resource thresholds."""
        try:
            self.thresholds = {
                ResourceType.CPU: ResourceThreshold(
                    resource_type=ResourceType.CPU,
                    warning_threshold=70.0,
                    critical_threshold=85.0,
                    overload_threshold=95.0,
                    scaling_threshold=80.0
                ),
                ResourceType.MEMORY: ResourceThreshold(
                    resource_type=ResourceType.MEMORY,
                    warning_threshold=75.0,
                    critical_threshold=90.0,
                    overload_threshold=95.0,
                    scaling_threshold=85.0
                ),
                ResourceType.DISK: ResourceThreshold(
                    resource_type=ResourceType.DISK,
                    warning_threshold=80.0,
                    critical_threshold=90.0,
                    overload_threshold=95.0,
                    scaling_threshold=85.0
                ),
                ResourceType.NETWORK: ResourceThreshold(
                    resource_type=ResourceType.NETWORK,
                    warning_threshold=70.0,
                    critical_threshold=85.0,
                    overload_threshold=95.0,
                    scaling_threshold=80.0
                )
            }
            
            if self.gpu_available:
                self.thresholds[ResourceType.GPU] = ResourceThreshold(
                    resource_type=ResourceType.GPU,
                    warning_threshold=75.0,
                    critical_threshold=90.0,
                    overload_threshold=95.0,
                    scaling_threshold=85.0
                )
            
        except Exception as e:
            logger.error(f"Failed to setup default thresholds: {str(e)}")
    
    def _setup_gpu_monitoring(self):
        """Setup GPU monitoring if available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_available = True
            logger.info("GPU monitoring enabled")
        except ImportError:
            logger.warning("pynvml not available, GPU monitoring disabled")
        except Exception as e:
            logger.warning(f"GPU monitoring setup failed: {str(e)}")
    
    async def initialize(self):
        """Initialize resource manager with monitoring tasks."""
        try:
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Start scaling task if enabled
            if self.enable_auto_scaling:
                self.scaling_task = asyncio.create_task(self._scaling_loop())
            
            logger.info("ResourceManager monitoring started")
            
        except Exception as e:
            logger.error(f"ResourceManager initialization failed: {str(e)}")
            raise
    
    async def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get current resource metrics."""
        try:
            metrics = await self._collect_metrics()
            self.current_metrics = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get current metrics: {str(e)}")
            return None
    
    async def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Load average (Unix systems)
            load_average = None
            try:
                load_average = psutil.getloadavg()
            except AttributeError:
                pass  # Not available on Windows
            
            # Active connections
            active_connections = len(psutil.net_connections())
            
            # GPU metrics
            gpu_utilization = None
            gpu_memory_percent = None
            
            if self.gpu_available:
                try:
                    gpu_utilization, gpu_memory_percent = await self._get_gpu_metrics()
                except Exception as e:
                    logger.warning(f"GPU metrics collection failed: {str(e)}")
            
            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_gb=memory_available_gb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                gpu_utilization=gpu_utilization,
                gpu_memory_percent=gpu_memory_percent,
                load_average=load_average,
                active_connections=active_connections
            )
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {str(e)}")
            raise
    
    async def _get_gpu_metrics(self) -> Tuple[Optional[float], Optional[float]]:
        """Get GPU utilization and memory metrics."""
        try:
            if not self.gpu_available:
                return None, None
            
            import pynvml
            
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                return None, None
            
            # Get metrics for first GPU
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = utilization.gpu
            
            # GPU memory
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_percent = (memory_info.used / memory_info.total) * 100
            
            return gpu_utilization, gpu_memory_percent
            
        except Exception as e:
            logger.error(f"GPU metrics collection failed: {str(e)}")
            return None, None
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        try:
            while True:
                await asyncio.sleep(self.monitoring_interval_seconds)
                
                try:
                    # Collect metrics
                    metrics = await self._collect_metrics()
                    self.current_metrics = metrics
                    
                    # Add to history
                    timestamp = datetime.now()
                    self.metrics_history.append((timestamp, metrics))
                    
                    # Check for alerts
                    await self._check_resource_alerts(metrics)
                    
                    # Update performance stats
                    await self._update_performance_stats(metrics)
                    
                    self.performance_stats['monitoring_cycles'] += 1
                    
                except Exception as e:
                    logger.error(f"Monitoring cycle error: {str(e)}")
                
        except asyncio.CancelledError:
            logger.info("Resource monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Monitoring loop error: {str(e)}")
    
    async def _scaling_loop(self):
        """Background scaling decision loop."""
        try:
            while True:
                await asyncio.sleep(30)  # Check scaling every 30 seconds
                
                try:
                    if self.current_metrics:
                        await self._evaluate_scaling_decisions(self.current_metrics)
                        
                except Exception as e:
                    logger.error(f"Scaling cycle error: {str(e)}")
                
        except asyncio.CancelledError:
            logger.info("Resource scaling loop cancelled")
        except Exception as e:
            logger.error(f"Scaling loop error: {str(e)}")
    
    async def _check_resource_alerts(self, metrics: ResourceMetrics):
        """Check for resource threshold violations and create alerts."""
        try:
            current_time = datetime.now()
            
            # Check each resource type
            resource_values = {
                ResourceType.CPU: metrics.cpu_percent,
                ResourceType.MEMORY: metrics.memory_percent,
                ResourceType.DISK: metrics.disk_usage_percent
            }
            
            if metrics.gpu_utilization is not None:
                resource_values[ResourceType.GPU] = metrics.gpu_utilization
            
            for resource_type, current_value in resource_values.items():
                threshold = self.thresholds.get(resource_type)
                if not threshold:
                    continue
                
                # Determine status
                status = ResourceStatus.OPTIMAL
                threshold_value = 0.0
                
                if current_value >= threshold.overload_threshold:
                    status = ResourceStatus.OVERLOADED
                    threshold_value = threshold.overload_threshold
                elif current_value >= threshold.critical_threshold:
                    status = ResourceStatus.CRITICAL
                    threshold_value = threshold.critical_threshold
                elif current_value >= threshold.warning_threshold:
                    status = ResourceStatus.WARNING
                    threshold_value = threshold.warning_threshold
                
                # Create or update alert
                alert_key = f"{resource_type.value}_{status.value}"
                
                if status != ResourceStatus.OPTIMAL:
                    if alert_key not in self.active_alerts:
                        alert = ResourceAlert(
                            resource_type=resource_type,
                            status=status,
                            current_value=current_value,
                            threshold_value=threshold_value,
                            message=f"{resource_type.value.upper()} usage at {current_value:.1f}% (threshold: {threshold_value:.1f}%)",
                            timestamp=current_time
                        )
                        
                        self.active_alerts[alert_key] = alert
                        self.alert_history.append(alert)
                        self.performance_stats['alerts_triggered'] += 1
                        
                        logger.warning(f"Resource alert: {alert.message}")
                        
                        # Trigger alert callbacks
                        await self._trigger_alert_callbacks(alert)
                    else:
                        # Update existing alert
                        self.active_alerts[alert_key].current_value = current_value
                        self.active_alerts[alert_key].timestamp = current_time
                
                else:
                    # Remove resolved alerts
                    resolved_alerts = [
                        key for key in self.active_alerts.keys()
                        if key.startswith(f"{resource_type.value}_")
                    ]
                    
                    for key in resolved_alerts:
                        resolved_alert = self.active_alerts.pop(key)
                        logger.info(f"Resource alert resolved: {resolved_alert.message}")
            
        except Exception as e:
            logger.error(f"Alert checking failed: {str(e)}")
    
    async def _evaluate_scaling_decisions(self, metrics: ResourceMetrics):
        """Evaluate if scaling actions are needed."""
        try:
            current_time = datetime.now()
            
            # Check each resource for scaling needs
            resource_values = {
                ResourceType.CPU: metrics.cpu_percent,
                ResourceType.MEMORY: metrics.memory_percent
            }
            
            if metrics.gpu_utilization is not None:
                resource_values[ResourceType.GPU] = metrics.gpu_utilization
            
            for resource_type, current_value in resource_values.items():
                threshold = self.thresholds.get(resource_type)
                if not threshold:
                    continue
                
                # Check cooldown period
                recent_scaling = [
                    event for event in self.scaling_events[-10:]
                    if (event.resource_type == resource_type and
                        (current_time - event.timestamp).total_seconds() < threshold.cooldown_seconds)
                ]
                
                if recent_scaling:
                    continue  # Still in cooldown
                
                # Determine scaling action
                action = ScalingAction.MAINTAIN
                
                if current_value >= threshold.scaling_threshold:
                    # Get trend from recent history
                    trend = await self._calculate_resource_trend(resource_type, minutes=5)
                    
                    if trend > 0:  # Increasing trend
                        action = ScalingAction.SCALE_UP
                    elif current_value >= threshold.critical_threshold:
                        action = ScalingAction.THROTTLE
                
                elif current_value < threshold.scaling_threshold * 0.5:
                    # Check if we can scale down
                    trend = await self._calculate_resource_trend(resource_type, minutes=10)
                    
                    if trend <= 0:  # Stable or decreasing trend
                        action = ScalingAction.SCALE_DOWN
                
                # Execute scaling action
                if action != ScalingAction.MAINTAIN:
                    await self._execute_scaling_action(action, resource_type, current_value, threshold)
            
        except Exception as e:
            logger.error(f"Scaling evaluation failed: {str(e)}")
    
    async def _calculate_resource_trend(self, resource_type: ResourceType, minutes: int = 5) -> float:
        """Calculate resource usage trend over specified time period."""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            # Get relevant metrics from history
            relevant_metrics = [
                (timestamp, metrics) for timestamp, metrics in self.metrics_history
                if timestamp >= cutoff_time
            ]
            
            if len(relevant_metrics) < 2:
                return 0.0
            
            # Extract values for the resource type
            values = []
            for timestamp, metrics in relevant_metrics:
                if resource_type == ResourceType.CPU:
                    values.append(metrics.cpu_percent)
                elif resource_type == ResourceType.MEMORY:
                    values.append(metrics.memory_percent)
                elif resource_type == ResourceType.GPU and metrics.gpu_utilization is not None:
                    values.append(metrics.gpu_utilization)
                else:
                    continue
            
            if len(values) < 2:
                return 0.0
            
            # Calculate simple linear trend
            n = len(values)
            x_values = list(range(n))
            
            # Linear regression slope
            x_mean = sum(x_values) / n
            y_mean = sum(values) / n
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
            denominator = sum((x - x_mean) ** 2 for x in x_values)
            
            if denominator == 0:
                return 0.0
            
            slope = numerator / denominator
            return slope
            
        except Exception as e:
            logger.error(f"Trend calculation failed: {str(e)}")
            return 0.0
    
    async def _execute_scaling_action(
        self,
        action: ScalingAction,
        resource_type: ResourceType,
        current_value: float,
        threshold: ResourceThreshold
    ):
        """Execute a scaling action."""
        try:
            scaling_event = ScalingEvent(
                action=action,
                resource_type=resource_type,
                trigger_value=current_value,
                target_value=threshold.scaling_threshold,
                timestamp=datetime.now(),
                success=False,
                details={}
            )
            
            # Execute scaling callbacks
            callbacks = self.scaling_callbacks.get(action, [])
            success_count = 0
            
            for callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        result = await callback(resource_type, current_value, threshold)
                    else:
                        result = callback(resource_type, current_value, threshold)
                    
                    if result:
                        success_count += 1
                        
                except Exception as callback_error:
                    logger.error(f"Scaling callback failed: {str(callback_error)}")
            
            scaling_event.success = success_count > 0
            scaling_event.details = {
                'callbacks_executed': len(callbacks),
                'successful_callbacks': success_count
            }
            
            self.scaling_events.append(scaling_event)
            self.performance_stats['scaling_actions'] += 1
            
            logger.info(f"Scaling action executed: {action.value} for {resource_type.value}")
            
        except Exception as e:
            logger.error(f"Scaling action execution failed: {str(e)}")
    
    async def _trigger_alert_callbacks(self, alert: ResourceAlert):
        """Trigger alert notification callbacks."""
        try:
            # This would typically send notifications to monitoring systems,
            # email alerts, Slack notifications, etc.
            logger.info(f"Alert triggered: {alert.message}")
            
        except Exception as e:
            logger.error(f"Alert callback failed: {str(e)}")
    
    async def _update_performance_stats(self, metrics: ResourceMetrics):
        """Update performance statistics."""
        try:
            # Update averages
            cycle_count = self.performance_stats['monitoring_cycles']
            
            if cycle_count > 0:
                # Update running averages
                self.performance_stats['average_cpu_usage'] = (
                    (self.performance_stats['average_cpu_usage'] * cycle_count + metrics.cpu_percent) / 
                    (cycle_count + 1)
                )
                
                self.performance_stats['average_memory_usage'] = (
                    (self.performance_stats['average_memory_usage'] * cycle_count + metrics.memory_percent) / 
                    (cycle_count + 1)
                )
            
            # Update peaks
            self.performance_stats['peak_usage']['cpu'] = max(
                self.performance_stats['peak_usage']['cpu'], 
                metrics.cpu_percent
            )
            
            self.performance_stats['peak_usage']['memory'] = max(
                self.performance_stats['peak_usage']['memory'], 
                metrics.memory_percent
            )
            
            self.performance_stats['peak_usage']['disk'] = max(
                self.performance_stats['peak_usage']['disk'], 
                metrics.disk_usage_percent
            )
            
            # Update uptime
            self.performance_stats['uptime_seconds'] = time.time() - self.start_time
            
        except Exception as e:
            logger.error(f"Performance stats update failed: {str(e)}")
    
    def register_scaling_callback(
        self,
        action: ScalingAction,
        callback: Callable
    ) -> bool:
        """Register a callback for scaling actions."""
        try:
            if action not in self.scaling_callbacks:
                self.scaling_callbacks[action] = []
            
            self.scaling_callbacks[action].append(callback)
            
            logger.info(f"Registered scaling callback for {action.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register scaling callback: {str(e)}")
            return False
    
    def update_threshold(
        self,
        resource_type: ResourceType,
        threshold: ResourceThreshold
    ) -> bool:
        """Update resource threshold configuration."""
        try:
            self.thresholds[resource_type] = threshold
            
            logger.info(f"Updated threshold for {resource_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update threshold: {str(e)}")
            return False
    
    async def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource summary."""
        try:
            current_metrics = await self.get_current_metrics()
            
            if not current_metrics:
                return {'error': 'Unable to collect current metrics'}
            
            # Calculate historical averages
            historical_stats = {}
            if self.metrics_history:
                recent_metrics = list(self.metrics_history)[-60:]  # Last 60 data points
                
                if recent_metrics:
                    avg_cpu = sum(m[1].cpu_percent for m in recent_metrics) / len(recent_metrics)
                    avg_memory = sum(m[1].memory_percent for m in recent_metrics) / len(recent_metrics)
                    avg_disk = sum(m[1].disk_usage_percent for m in recent_metrics) / len(recent_metrics)
                    
                    historical_stats = {
                        'average_cpu_percent': avg_cpu,
                        'average_memory_percent': avg_memory,
                        'average_disk_percent': avg_disk,
                        'data_points': len(recent_metrics)
                    }
            
            return {
                'current_metrics': current_metrics.to_dict(),
                'historical_stats': historical_stats,
                'active_alerts': {
                    alert_id: {
                        'resource_type': alert.resource_type.value,
                        'status': alert.status.value,
                        'current_value': alert.current_value,
                        'threshold_value': alert.threshold_value,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert_id, alert in self.active_alerts.items()
                },
                'recent_scaling_events': [
                    {
                        'action': event.action.value,
                        'resource_type': event.resource_type.value,
                        'trigger_value': event.trigger_value,
                        'success': event.success,
                        'timestamp': event.timestamp.isoformat()
                    }
                    for event in self.scaling_events[-10:]
                ],
                'performance_stats': self.performance_stats.copy(),
                'thresholds': {
                    resource_type.value: {
                        'warning': threshold.warning_threshold,
                        'critical': threshold.critical_threshold,
                        'overload': threshold.overload_threshold,
                        'scaling': threshold.scaling_threshold
                    }
                    for resource_type, threshold in self.thresholds.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get resource summary: {str(e)}")
            return {'error': str(e)}
    
    async def get_resource_trends(self, hours: int = 1) -> Dict[str, List[Dict[str, Any]]]:
        """Get resource usage trends over specified time period."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            relevant_metrics = [
                (timestamp, metrics) for timestamp, metrics in self.metrics_history
                if timestamp >= cutoff_time
            ]
            
            trends = {
                'cpu': [],
                'memory': [],
                'disk': []
            }
            
            for timestamp, metrics in relevant_metrics:
                data_point = {
                    'timestamp': timestamp.isoformat(),
                    'value': 0.0
                }
                
                # CPU trend
                cpu_point = data_point.copy()
                cpu_point['value'] = metrics.cpu_percent
                trends['cpu'].append(cpu_point)
                
                # Memory trend
                memory_point = data_point.copy()
                memory_point['value'] = metrics.memory_percent
                trends['memory'].append(memory_point)
                
                # Disk trend
                disk_point = data_point.copy()
                disk_point['value'] = metrics.disk_usage_percent
                trends['disk'].append(disk_point)
            
            # Add GPU trend if available
            if any(m[1].gpu_utilization is not None for m in relevant_metrics):
                trends['gpu'] = []
                for timestamp, metrics in relevant_metrics:
                    if metrics.gpu_utilization is not None:
                        gpu_point = {
                            'timestamp': timestamp.isoformat(),
                            'value': metrics.gpu_utilization
                        }
                        trends['gpu'].append(gpu_point)
            
            return trends
            
        except Exception as e:
            logger.error(f"Failed to get resource trends: {str(e)}")
            return {}
    
    async def acknowledge_alert(self, alert_key: str) -> bool:
        """Acknowledge an active alert."""
        try:
            if alert_key in self.active_alerts:
                self.active_alerts[alert_key].acknowledged = True
                logger.info(f"Alert acknowledged: {alert_key}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {str(e)}")
            return False
    
    async def cleanup(self):
        """Clean up resource manager."""
        try:
            # Cancel monitoring tasks
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            if self.scaling_task:
                self.scaling_task.cancel()
                try:
                    await self.scaling_task
                except asyncio.CancelledError:
                    pass
            
            # Clear data structures
            self.metrics_history.clear()
            self.active_alerts.clear()
            self.alert_history.clear()
            self.scaling_events.clear()
            self.scaling_callbacks.clear()
            
            logger.info("ResourceManager cleanup completed")
            
        except Exception as e:
            logger.error(f"ResourceManager cleanup failed: {str(e)}")