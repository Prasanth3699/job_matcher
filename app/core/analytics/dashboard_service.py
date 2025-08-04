"""
Dashboard Service for comprehensive monitoring and visualization.

This module provides a unified dashboard service that aggregates data from all
monitoring components and presents real-time analytics and insights.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from app.utils.logger import logger
from app.core.analytics.analytics_engine import AnalyticsEngine
from app.core.analytics.model_monitor import ModelPerformanceMonitor
from app.core.analytics.business_metrics import BusinessMetricsCollector


class DashboardType(str, Enum):
    """Types of dashboards."""
    EXECUTIVE = "executive"
    OPERATIONAL = "operational"
    TECHNICAL = "technical"
    BUSINESS = "business"
    MODEL_PERFORMANCE = "model_performance"
    REAL_TIME = "real_time"


class WidgetType(str, Enum):
    """Types of dashboard widgets."""
    METRIC = "metric"
    CHART = "chart"
    TABLE = "table"
    ALERT = "alert"
    GAUGE = "gauge"
    TREND = "trend"
    MAP = "map"
    TEXT = "text"


class ChartType(str, Enum):
    """Types of charts."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    AREA = "area"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    widget_id: str
    title: str
    widget_type: WidgetType
    data_source: str
    config: Dict[str, Any]
    position: Dict[str, int]  # x, y, width, height
    refresh_interval: int = 30  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'widget_id': self.widget_id,
            'title': self.title,
            'widget_type': self.widget_type.value,
            'data_source': self.data_source,
            'config': self.config,
            'position': self.position,
            'refresh_interval': self.refresh_interval
        }


@dataclass
class Dashboard:
    """Dashboard configuration."""
    dashboard_id: str
    name: str
    description: str
    dashboard_type: DashboardType
    widgets: List[DashboardWidget]
    layout: Dict[str, Any]
    permissions: List[str]
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'dashboard_id': self.dashboard_id,
            'name': self.name,
            'description': self.description,
            'dashboard_type': self.dashboard_type.value,
            'widgets': [w.to_dict() for w in self.widgets],
            'layout': self.layout,
            'permissions': self.permissions,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class DashboardData:
    """Dashboard data response."""
    dashboard_id: str
    timestamp: datetime
    widgets_data: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class DashboardService:
    """
    Comprehensive dashboard service that aggregates monitoring data
    and provides real-time visualization capabilities.
    """
    
    def __init__(
        self,
        analytics_engine: Optional[AnalyticsEngine] = None,
        model_monitor: Optional[ModelPerformanceMonitor] = None,
        business_metrics: Optional[BusinessMetricsCollector] = None,
        update_interval_seconds: int = 30
    ):
        """Initialize dashboard service."""
        self.analytics_engine = analytics_engine
        self.model_monitor = model_monitor
        self.business_metrics = business_metrics
        self.update_interval_seconds = update_interval_seconds
        
        # Dashboard storage
        self.dashboards: Dict[str, Dashboard] = {}
        self.dashboard_data: Dict[str, DashboardData] = {}
        
        # Data sources
        self.data_sources: Dict[str, Callable] = {}
        self._setup_default_data_sources()
        
        # Real-time updates
        self.update_task: Optional[asyncio.Task] = None
        self.subscribers: Dict[str, List[Callable]] = {}
        
        # Cache
        self.data_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.cache_ttl_seconds = 60
        
        # Statistics
        self.service_stats = {
            'dashboards_created': 0,
            'data_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'subscribers_count': 0,
            'updates_sent': 0
        }
        
        # Setup default dashboards
        self._setup_default_dashboards()
        
        logger.info("DashboardService initialized")
    
    def _setup_default_data_sources(self):
        """Setup default data source functions."""
        try:
            # Analytics data sources
            self.data_sources['analytics_summary'] = self._get_analytics_summary
            self.data_sources['analytics_insights'] = self._get_analytics_insights
            self.data_sources['analytics_patterns'] = self._get_analytics_patterns
            
            # Model performance data sources
            self.data_sources['model_performance'] = self._get_model_performance
            self.data_sources['model_alerts'] = self._get_model_alerts
            self.data_sources['drift_detection'] = self._get_drift_detection
            
            # Business metrics data sources
            self.data_sources['business_metrics'] = self._get_business_metrics
            self.data_sources['revenue_metrics'] = self._get_revenue_metrics
            self.data_sources['user_metrics'] = self._get_user_metrics
            self.data_sources['conversion_metrics'] = self._get_conversion_metrics
            
            # System metrics data sources
            self.data_sources['system_health'] = self._get_system_health
            self.data_sources['performance_trends'] = self._get_performance_trends
            
            logger.info(f"Setup {len(self.data_sources)} default data sources")
            
        except Exception as e:
            logger.error(f"Failed to setup data sources: {str(e)}")
    
    def _setup_default_dashboards(self):
        """Setup default dashboard configurations."""
        try:
            # Executive Dashboard
            executive_dashboard = self._create_executive_dashboard()
            self.dashboards[executive_dashboard.dashboard_id] = executive_dashboard
            
            # Operational Dashboard
            operational_dashboard = self._create_operational_dashboard()
            self.dashboards[operational_dashboard.dashboard_id] = operational_dashboard
            
            # Technical Dashboard
            technical_dashboard = self._create_technical_dashboard()
            self.dashboards[technical_dashboard.dashboard_id] = technical_dashboard
            
            # Business Dashboard
            business_dashboard = self._create_business_dashboard()
            self.dashboards[business_dashboard.dashboard_id] = business_dashboard
            
            # Model Performance Dashboard
            model_dashboard = self._create_model_performance_dashboard()
            self.dashboards[model_dashboard.dashboard_id] = model_dashboard
            
            # Real-time Dashboard
            realtime_dashboard = self._create_realtime_dashboard()
            self.dashboards[realtime_dashboard.dashboard_id] = realtime_dashboard
            
            self.service_stats['dashboards_created'] = len(self.dashboards)
            
            logger.info(f"Setup {len(self.dashboards)} default dashboards")
            
        except Exception as e:
            logger.error(f"Failed to setup default dashboards: {str(e)}")
    
    def _create_executive_dashboard(self) -> Dashboard:
        """Create executive dashboard with high-level KPIs."""
        widgets = [
            DashboardWidget(
                widget_id="total_revenue",
                title="Total Revenue",
                widget_type=WidgetType.METRIC,
                data_source="revenue_metrics",
                config={
                    "metric": "total_revenue",
                    "format": "currency",
                    "trend": True
                },
                position={"x": 0, "y": 0, "width": 3, "height": 2}
            ),
            DashboardWidget(
                widget_id="active_users",
                title="Active Users",
                widget_type=WidgetType.METRIC,
                data_source="user_metrics",
                config={
                    "metric": "daily_active_users",
                    "format": "number",
                    "trend": True
                },
                position={"x": 3, "y": 0, "width": 3, "height": 2}
            ),
            DashboardWidget(
                widget_id="conversion_rate",
                title="Conversion Rate",
                widget_type=WidgetType.GAUGE,
                data_source="conversion_metrics",
                config={
                    "metric": "overall_conversion_rate",
                    "min": 0,
                    "max": 1,
                    "target": 0.15
                },
                position={"x": 6, "y": 0, "width": 3, "height": 2}
            ),
            DashboardWidget(
                widget_id="revenue_trend",
                title="Revenue Trend (30 Days)",
                widget_type=WidgetType.CHART,
                data_source="revenue_metrics",
                config={
                    "chart_type": ChartType.LINE.value,
                    "metric": "daily_revenue",
                    "time_range": "30d"
                },
                position={"x": 0, "y": 2, "width": 6, "height": 4}
            ),
            DashboardWidget(
                widget_id="key_metrics_table",
                title="Key Performance Indicators",
                widget_type=WidgetType.TABLE,
                data_source="business_metrics",
                config={
                    "metrics": [
                        "daily_active_users",
                        "job_match_success_rate",
                        "revenue_per_user",
                        "user_satisfaction_score"
                    ]
                },
                position={"x": 6, "y": 2, "width": 6, "height": 4}
            )
        ]
        
        return Dashboard(
            dashboard_id="executive",
            name="Executive Dashboard",
            description="High-level business metrics and KPIs for executives",
            dashboard_type=DashboardType.EXECUTIVE,
            widgets=widgets,
            layout={"grid_columns": 12, "grid_rows": 8},
            permissions=["executive", "admin"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def _create_operational_dashboard(self) -> Dashboard:
        """Create operational dashboard for day-to-day operations."""
        widgets = [
            DashboardWidget(
                widget_id="system_health",
                title="System Health",
                widget_type=WidgetType.GAUGE,
                data_source="system_health",
                config={
                    "metric": "overall_health_score",
                    "min": 0,
                    "max": 100,
                    "thresholds": [50, 80]
                },
                position={"x": 0, "y": 0, "width": 3, "height": 2}
            ),
            DashboardWidget(
                widget_id="active_alerts",
                title="Active Alerts",
                widget_type=WidgetType.ALERT,
                data_source="model_alerts",
                config={
                    "severity_filter": ["high", "critical"],
                    "max_alerts": 10
                },
                position={"x": 3, "y": 0, "width": 3, "height": 2}
            ),
            DashboardWidget(
                widget_id="request_volume",
                title="Request Volume",
                widget_type=WidgetType.CHART,
                data_source="analytics_summary",
                config={
                    "chart_type": ChartType.LINE.value,
                    "metric": "requests_per_minute",
                    "time_range": "24h"
                },
                position={"x": 6, "y": 0, "width": 6, "height": 3}
            ),
            DashboardWidget(
                widget_id="model_performance",
                title="Model Performance Overview",
                widget_type=WidgetType.TABLE,
                data_source="model_performance",
                config={
                    "models": "all",
                    "metrics": ["accuracy", "latency", "error_rate"]
                },
                position={"x": 0, "y": 3, "width": 12, "height": 4}
            )
        ]
        
        return Dashboard(
            dashboard_id="operational",
            name="Operational Dashboard",
            description="Day-to-day operational metrics and system health",
            dashboard_type=DashboardType.OPERATIONAL,
            widgets=widgets,
            layout={"grid_columns": 12, "grid_rows": 8},
            permissions=["operator", "admin"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def _create_technical_dashboard(self) -> Dashboard:
        """Create technical dashboard for engineering teams."""
        widgets = [
            DashboardWidget(
                widget_id="model_drift",
                title="Model Drift Detection",
                widget_type=WidgetType.CHART,
                data_source="drift_detection",
                config={
                    "chart_type": ChartType.HEATMAP.value,
                    "time_range": "7d"
                },
                position={"x": 0, "y": 0, "width": 6, "height": 4}
            ),
            DashboardWidget(
                widget_id="performance_trends",
                title="Performance Trends",
                widget_type=WidgetType.CHART,
                data_source="performance_trends",
                config={
                    "chart_type": ChartType.LINE.value,
                    "metrics": ["response_time", "throughput", "error_rate"],
                    "time_range": "24h"
                },
                position={"x": 6, "y": 0, "width": 6, "height": 4}
            ),
            DashboardWidget(
                widget_id="analytics_insights",
                title="Latest Analytics Insights",
                widget_type=WidgetType.TEXT,
                data_source="analytics_insights",
                config={
                    "max_insights": 5,
                    "insight_types": ["anomaly", "trend", "pattern"]
                },
                position={"x": 0, "y": 4, "width": 6, "height": 4}
            ),
            DashboardWidget(
                widget_id="model_comparison",
                title="Model Performance Comparison",
                widget_type=WidgetType.CHART,
                data_source="model_performance",
                config={
                    "chart_type": ChartType.BAR.value,
                    "comparison_metric": "accuracy",
                    "models": "all"
                },
                position={"x": 6, "y": 4, "width": 6, "height": 4}
            )
        ]
        
        return Dashboard(
            dashboard_id="technical",
            name="Technical Dashboard",
            description="Technical metrics and insights for engineering teams",
            dashboard_type=DashboardType.TECHNICAL,
            widgets=widgets,
            layout={"grid_columns": 12, "grid_rows": 8},
            permissions=["engineer", "admin"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def _create_business_dashboard(self) -> Dashboard:
        """Create business dashboard for business analysts."""
        widgets = [
            DashboardWidget(
                widget_id="user_acquisition",
                title="User Acquisition",
                widget_type=WidgetType.CHART,
                data_source="user_metrics",
                config={
                    "chart_type": ChartType.AREA.value,
                    "metrics": ["new_registrations", "activated_users"],
                    "time_range": "30d"
                },
                position={"x": 0, "y": 0, "width": 6, "height": 3}
            ),
            DashboardWidget(
                widget_id="conversion_funnel",
                title="Conversion Funnel",
                widget_type=WidgetType.CHART,
                data_source="conversion_metrics",
                config={
                    "chart_type": ChartType.BAR.value,
                    "funnel_steps": ["registration", "activation", "first_match", "application"]
                },
                position={"x": 6, "y": 0, "width": 6, "height": 3}
            ),
            DashboardWidget(
                widget_id="revenue_breakdown",
                title="Revenue Breakdown",
                widget_type=WidgetType.CHART,
                data_source="revenue_metrics",
                config={
                    "chart_type": ChartType.PIE.value,
                    "breakdown_by": "user_segment"
                },
                position={"x": 0, "y": 3, "width": 4, "height": 3}
            ),
            DashboardWidget(
                widget_id="user_segments",
                title="User Segments Performance",
                widget_type=WidgetType.TABLE,
                data_source="user_metrics",
                config={
                    "segments": ["new", "active", "power_user", "enterprise"],
                    "metrics": ["count", "conversion_rate", "revenue_per_user"]
                },
                position={"x": 4, "y": 3, "width": 8, "height": 3}
            )
        ]
        
        return Dashboard(
            dashboard_id="business",
            name="Business Dashboard",
            description="Business analytics and user behavior insights",
            dashboard_type=DashboardType.BUSINESS,
            widgets=widgets,
            layout={"grid_columns": 12, "grid_rows": 8},
            permissions=["analyst", "business", "admin"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def _create_model_performance_dashboard(self) -> Dashboard:
        """Create model performance dashboard for ML teams."""
        widgets = [
            DashboardWidget(
                widget_id="model_accuracy_trends",
                title="Model Accuracy Trends",
                widget_type=WidgetType.CHART,
                data_source="model_performance",
                config={
                    "chart_type": ChartType.LINE.value,
                    "metric": "accuracy",
                    "time_range": "7d",
                    "models": "all"
                },
                position={"x": 0, "y": 0, "width": 6, "height": 3}
            ),
            DashboardWidget(
                widget_id="model_latency",
                title="Model Latency Distribution",
                widget_type=WidgetType.CHART,
                data_source="model_performance",
                config={
                    "chart_type": ChartType.HISTOGRAM.value,
                    "metric": "latency_p95",
                    "models": "all"
                },
                position={"x": 6, "y": 0, "width": 6, "height": 3}
            ),
            DashboardWidget(
                widget_id="drift_alerts",
                title="Recent Drift Alerts",
                widget_type=WidgetType.ALERT,
                data_source="drift_detection",
                config={
                    "drift_types": ["data_drift", "concept_drift"],
                    "time_range": "24h"
                },
                position={"x": 0, "y": 3, "width": 6, "height": 2}
            ),
            DashboardWidget(
                widget_id="model_health_status",
                title="Model Health Status",
                widget_type=WidgetType.TABLE,
                data_source="model_performance",
                config={
                    "models": "all",
                    "health_indicators": ["status", "last_update", "prediction_count", "error_rate"]
                },
                position={"x": 6, "y": 3, "width": 6, "height": 2}
            )
        ]
        
        return Dashboard(
            dashboard_id="model_performance",
            name="Model Performance Dashboard",
            description="ML model performance monitoring and drift detection",
            dashboard_type=DashboardType.MODEL_PERFORMANCE,
            widgets=widgets,
            layout={"grid_columns": 12, "grid_rows": 8},
            permissions=["ml_engineer", "data_scientist", "admin"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def _create_realtime_dashboard(self) -> Dashboard:
        """Create real-time dashboard for live monitoring."""
        widgets = [
            DashboardWidget(
                widget_id="live_requests",
                title="Live Request Count",
                widget_type=WidgetType.METRIC,
                data_source="analytics_summary",
                config={
                    "metric": "requests_per_second",
                    "format": "number",
                    "refresh_rate": 5
                },
                position={"x": 0, "y": 0, "width": 2, "height": 2},
                refresh_interval=5
            ),
            DashboardWidget(
                widget_id="live_users",
                title="Live Users",
                widget_type=WidgetType.METRIC,
                data_source="user_metrics",
                config={
                    "metric": "concurrent_users",
                    "format": "number",
                    "refresh_rate": 5
                },
                position={"x": 2, "y": 0, "width": 2, "height": 2},
                refresh_interval=5
            ),
            DashboardWidget(
                widget_id="live_response_time",
                title="Response Time",
                widget_type=WidgetType.GAUGE,
                data_source="performance_trends",
                config={
                    "metric": "avg_response_time",
                    "min": 0,
                    "max": 2000,
                    "unit": "ms",
                    "thresholds": [500, 1000]
                },
                position={"x": 4, "y": 0, "width": 2, "height": 2},
                refresh_interval=5
            ),
            DashboardWidget(
                widget_id="live_alerts",
                title="Live System Alerts",
                widget_type=WidgetType.ALERT,
                data_source="model_alerts",
                config={
                    "max_alerts": 5,
                    "auto_refresh": True
                },
                position={"x": 6, "y": 0, "width": 6, "height": 4},
                refresh_interval=10
            ),
            DashboardWidget(
                widget_id="live_activity_feed",
                title="Live Activity Feed",
                widget_type=WidgetType.TEXT,
                data_source="analytics_summary",
                config={
                    "feed_type": "activity",
                    "max_items": 10,
                    "auto_scroll": True
                },
                position={"x": 0, "y": 4, "width": 12, "height": 4},
                refresh_interval=15
            )
        ]
        
        return Dashboard(
            dashboard_id="realtime",
            name="Real-time Dashboard",
            description="Live system monitoring and real-time metrics",
            dashboard_type=DashboardType.REAL_TIME,
            widgets=widgets,
            layout={"grid_columns": 12, "grid_rows": 8},
            permissions=["operator", "admin"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    async def initialize(self):
        """Initialize dashboard service."""
        try:
            # Start update task
            self.update_task = asyncio.create_task(self._update_loop())
            
            logger.info("DashboardService started")
            
        except Exception as e:
            logger.error(f"DashboardService initialization failed: {str(e)}")
            raise
    
    async def _update_loop(self):
        """Background update loop for real-time dashboards."""
        try:
            while True:
                await asyncio.sleep(self.update_interval_seconds)
                
                try:
                    await self._update_dashboard_data()
                    await self._notify_subscribers()
                    
                except Exception as e:
                    logger.error(f"Dashboard update cycle error: {str(e)}")
                
        except asyncio.CancelledError:
            logger.info("Dashboard update loop cancelled")
        except Exception as e:
            logger.error(f"Dashboard update loop error: {str(e)}")
    
    async def _update_dashboard_data(self):
        """Update data for all dashboards."""
        try:
            for dashboard_id, dashboard in self.dashboards.items():
                dashboard_data = await self._collect_dashboard_data(dashboard)
                self.dashboard_data[dashboard_id] = dashboard_data
                
        except Exception as e:
            logger.error(f"Dashboard data update failed: {str(e)}")
    
    async def _collect_dashboard_data(self, dashboard: Dashboard) -> DashboardData:
        """Collect data for a specific dashboard."""
        try:
            widgets_data = {}
            alerts = []
            
            for widget in dashboard.widgets:
                try:
                    # Check cache first
                    cache_key = f"{widget.data_source}_{widget.widget_id}"
                    
                    if (cache_key in self.data_cache and 
                        cache_key in self.cache_timestamps and
                        (datetime.now() - self.cache_timestamps[cache_key]).total_seconds() < self.cache_ttl_seconds):
                        
                        widgets_data[widget.widget_id] = self.data_cache[cache_key]
                        self.service_stats['cache_hits'] += 1
                        continue
                    
                    # Fetch fresh data
                    if widget.data_source in self.data_sources:
                        data_source_func = self.data_sources[widget.data_source]
                        widget_data = await data_source_func(widget.config)
                        
                        widgets_data[widget.widget_id] = widget_data
                        
                        # Cache the data
                        self.data_cache[cache_key] = widget_data
                        self.cache_timestamps[cache_key] = datetime.now()
                        self.service_stats['cache_misses'] += 1
                    
                except Exception as widget_error:
                    logger.error(f"Widget data collection failed for {widget.widget_id}: {str(widget_error)}")
                    widgets_data[widget.widget_id] = {'error': str(widget_error)}
            
            # Collect alerts
            try:
                alerts = await self._collect_dashboard_alerts(dashboard)
            except Exception as alert_error:
                logger.error(f"Alert collection failed: {str(alert_error)}")
            
            return DashboardData(
                dashboard_id=dashboard.dashboard_id,
                timestamp=datetime.now(),
                widgets_data=widgets_data,
                alerts=alerts,
                metadata={
                    'dashboard_type': dashboard.dashboard_type.value,
                    'widget_count': len(dashboard.widgets),
                    'last_updated': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Dashboard data collection failed: {str(e)}")
            raise
    
    async def _collect_dashboard_alerts(self, dashboard: Dashboard) -> List[Dict[str, Any]]:
        """Collect alerts relevant to the dashboard."""
        try:
            alerts = []
            
            # Collect model alerts if model monitor is available
            if self.model_monitor:
                try:
                    monitor_summary = await self.model_monitor.get_monitoring_summary()
                    if 'recent_alerts' in monitor_summary:
                        for alert in monitor_summary['recent_alerts'][-5:]:  # Last 5 alerts
                            alerts.append({
                                'type': 'model_alert',
                                'severity': alert.get('severity', 'medium'),
                                'message': alert.get('type', 'Model alert'),
                                'timestamp': alert.get('timestamp'),
                                'source': 'model_monitor'
                            })
                except Exception as e:
                    logger.error(f"Model alerts collection failed: {str(e)}")
            
            # Collect business metric alerts if available
            if self.business_metrics:
                try:
                    # This would collect business metric threshold violations
                    # Implementation depends on business metrics structure
                    pass
                except Exception as e:
                    logger.error(f"Business alerts collection failed: {str(e)}")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Dashboard alerts collection failed: {str(e)}")
            return []
    
    # Data source implementations
    async def _get_analytics_summary(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get analytics summary data."""
        try:
            if not self.analytics_engine:
                return {'error': 'Analytics engine not available'}
            
            summary = await self.analytics_engine.get_analytics_summary()
            return {
                'data': summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Analytics summary data source failed: {str(e)}")
            return {'error': str(e)}
    
    async def _get_analytics_insights(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get analytics insights data."""
        try:
            if not self.analytics_engine:
                return {'error': 'Analytics engine not available'}
            
            summary = await self.analytics_engine.get_analytics_summary()
            insights = summary.get('recent_insights', [])
            
            # Filter by insight type if specified
            insight_types = config.get('insight_types', [])
            if insight_types:
                insights = [
                    insight for insight in insights
                    if insight.get('insight_type') in insight_types
                ]
            
            # Limit number of insights
            max_insights = config.get('max_insights', 10)
            insights = insights[:max_insights]
            
            return {
                'insights': insights,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Analytics insights data source failed: {str(e)}")
            return {'error': str(e)}
    
    async def _get_analytics_patterns(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get analytics patterns data."""
        try:
            if not self.analytics_engine:
                return {'error': 'Analytics engine not available'}
            
            summary = await self.analytics_engine.get_analytics_summary()
            patterns = summary.get('top_patterns', [])
            
            return {
                'patterns': patterns,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Analytics patterns data source failed: {str(e)}")
            return {'error': str(e)}
    
    async def _get_model_performance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get model performance data."""
        try:
            if not self.model_monitor:
                return {'error': 'Model monitor not available'}
            
            summary = await self.model_monitor.get_monitoring_summary()
            
            # Filter by specific models if requested
            models = config.get('models', 'all')
            if models != 'all':
                # Filter the summary for specific models
                # Implementation would depend on summary structure
                pass
            
            return {
                'data': summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model performance data source failed: {str(e)}")
            return {'error': str(e)}
    
    async def _get_model_alerts(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get model alerts data."""
        try:
            if not self.model_monitor:
                return {'error': 'Model monitor not available'}
            
            summary = await self.model_monitor.get_monitoring_summary()
            alerts = summary.get('recent_alerts', [])
            
            # Filter by severity if specified
            severity_filter = config.get('severity_filter', [])
            if severity_filter:
                alerts = [
                    alert for alert in alerts
                    if alert.get('severity') in severity_filter
                ]
            
            # Limit number of alerts
            max_alerts = config.get('max_alerts', 10)
            alerts = alerts[:max_alerts]
            
            return {
                'alerts': alerts,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model alerts data source failed: {str(e)}")
            return {'error': str(e)}
    
    async def _get_drift_detection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get drift detection data."""
        try:
            if not self.model_monitor:
                return {'error': 'Model monitor not available'}
            
            summary = await self.model_monitor.get_monitoring_summary()
            
            return {
                'drift_summary': summary.get('drift_summary', {}),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Drift detection data source failed: {str(e)}")
            return {'error': str(e)}
    
    async def _get_business_metrics(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get business metrics data."""
        try:
            if not self.business_metrics:
                return {'error': 'Business metrics not available'}
            
            summary = await self.business_metrics.get_metrics_summary()
            
            # Filter by specific metrics if requested
            metrics = config.get('metrics', [])
            if metrics:
                current_metrics = summary.get('current_metrics', {})
                filtered_metrics = {
                    metric_id: metric_data
                    for metric_id, metric_data in current_metrics.items()
                    if metric_id in metrics
                }
                summary['current_metrics'] = filtered_metrics
            
            return {
                'data': summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Business metrics data source failed: {str(e)}")
            return {'error': str(e)}
    
    async def _get_revenue_metrics(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get revenue metrics data."""
        try:
            if not self.business_metrics:
                return {'error': 'Business metrics not available'}
            
            summary = await self.business_metrics.get_metrics_summary()
            
            return {
                'total_revenue': summary.get('total_revenue', 0),
                'revenue_tracking': summary.get('collector_stats', {}).get('revenue_tracked', 0),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Revenue metrics data source failed: {str(e)}")
            return {'error': str(e)}
    
    async def _get_user_metrics(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get user metrics data."""
        try:
            if not self.business_metrics:
                return {'error': 'Business metrics not available'}
            
            summary = await self.business_metrics.get_metrics_summary()
            
            return {
                'active_users': summary.get('active_users', 0),
                'real_time_metrics': summary.get('real_time_metrics', {}),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"User metrics data source failed: {str(e)}")
            return {'error': str(e)}
    
    async def _get_conversion_metrics(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get conversion metrics data."""
        try:
            if not self.business_metrics:
                return {'error': 'Business metrics not available'}
            
            summary = await self.business_metrics.get_metrics_summary()
            conversion_data = {}
            
            # Extract conversion-related metrics
            current_metrics = summary.get('current_metrics', {})
            for metric_id, metric_data in current_metrics.items():
                if 'conversion' in metric_id or 'rate' in metric_id:
                    conversion_data[metric_id] = metric_data
            
            return {
                'conversions': conversion_data,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Conversion metrics data source failed: {str(e)}")
            return {'error': str(e)}
    
    async def _get_system_health(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get system health data."""
        try:
            # This would integrate with system monitoring
            # For now, return mock data
            return {
                'overall_health_score': 85,
                'components': {
                    'api': 'healthy',
                    'database': 'healthy',
                    'cache': 'warning',
                    'ml_models': 'healthy'
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"System health data source failed: {str(e)}")
            return {'error': str(e)}
    
    async def _get_performance_trends(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get performance trends data."""
        try:
            # This would collect performance metrics over time
            # For now, return mock data
            return {
                'response_time': {
                    'current': 150,
                    'trend': 'stable'
                },
                'throughput': {
                    'current': 1200,
                    'trend': 'increasing'
                },
                'error_rate': {
                    'current': 0.02,
                    'trend': 'decreasing'
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Performance trends data source failed: {str(e)}")
            return {'error': str(e)}
    
    async def _notify_subscribers(self):
        """Notify subscribers of dashboard updates."""
        try:
            for dashboard_id, callbacks in self.subscribers.items():
                if dashboard_id in self.dashboard_data:
                    dashboard_data = self.dashboard_data[dashboard_id]
                    
                    for callback in callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(dashboard_data)
                            else:
                                callback(dashboard_data)
                                
                            self.service_stats['updates_sent'] += 1
                            
                        except Exception as callback_error:
                            logger.error(f"Subscriber notification failed: {str(callback_error)}")
            
        except Exception as e:
            logger.error(f"Subscriber notification failed: {str(e)}")
    
    async def get_dashboard(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Get dashboard configuration."""
        try:
            if dashboard_id not in self.dashboards:
                return None
            
            dashboard = self.dashboards[dashboard_id]
            return dashboard.to_dict()
            
        except Exception as e:
            logger.error(f"Failed to get dashboard {dashboard_id}: {str(e)}")
            return None
    
    async def get_dashboard_data(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Get dashboard data."""
        try:
            self.service_stats['data_requests'] += 1
            
            if dashboard_id not in self.dashboards:
                return None
            
            # Get fresh data
            dashboard = self.dashboards[dashboard_id]
            dashboard_data = await self._collect_dashboard_data(dashboard)
            
            return {
                'dashboard_id': dashboard_data.dashboard_id,
                'timestamp': dashboard_data.timestamp.isoformat(),
                'widgets_data': dashboard_data.widgets_data,
                'alerts': dashboard_data.alerts,
                'metadata': dashboard_data.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data for {dashboard_id}: {str(e)}")
            return None
    
    def subscribe_to_dashboard(self, dashboard_id: str, callback: Callable) -> bool:
        """Subscribe to dashboard updates."""
        try:
            if dashboard_id not in self.dashboards:
                return False
            
            if dashboard_id not in self.subscribers:
                self.subscribers[dashboard_id] = []
            
            self.subscribers[dashboard_id].append(callback)
            self.service_stats['subscribers_count'] += 1
            
            logger.info(f"Added subscriber to dashboard {dashboard_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to dashboard {dashboard_id}: {str(e)}")
            return False
    
    def unsubscribe_from_dashboard(self, dashboard_id: str, callback: Callable) -> bool:
        """Unsubscribe from dashboard updates."""
        try:
            if dashboard_id in self.subscribers and callback in self.subscribers[dashboard_id]:
                self.subscribers[dashboard_id].remove(callback)
                self.service_stats['subscribers_count'] -= 1
                
                logger.info(f"Removed subscriber from dashboard {dashboard_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from dashboard {dashboard_id}: {str(e)}")
            return False
    
    async def list_dashboards(self) -> List[Dict[str, Any]]:
        """List all available dashboards."""
        try:
            return [
                {
                    'dashboard_id': dashboard.dashboard_id,
                    'name': dashboard.name,
                    'description': dashboard.description,
                    'dashboard_type': dashboard.dashboard_type.value,
                    'widget_count': len(dashboard.widgets)
                }
                for dashboard in self.dashboards.values()
            ]
            
        except Exception as e:
            logger.error(f"Failed to list dashboards: {str(e)}")
            return []
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get dashboard service statistics."""
        try:
            return {
                'service_stats': self.service_stats.copy(),
                'dashboards_count': len(self.dashboards),
                'active_subscribers': sum(len(subs) for subs in self.subscribers.values()),
                'cache_size': len(self.data_cache),
                'data_sources_count': len(self.data_sources)
            }
            
        except Exception as e:
            logger.error(f"Failed to get service stats: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Clean up dashboard service resources."""
        try:
            # Cancel update task
            if self.update_task:
                self.update_task.cancel()
                try:
                    await self.update_task
                except asyncio.CancelledError:
                    pass
            
            # Clear data structures
            self.dashboards.clear()
            self.dashboard_data.clear()
            self.data_sources.clear()
            self.subscribers.clear()
            self.data_cache.clear()
            self.cache_timestamps.clear()
            
            logger.info("DashboardService cleanup completed")
            
        except Exception as e:
            logger.error(f"DashboardService cleanup failed: {str(e)}")