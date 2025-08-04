"""
Business Metrics Collector for tracking business KPIs and ROI analytics.

This module provides comprehensive business metrics tracking including user
engagement, conversion rates, revenue impact, and business intelligence analytics.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from decimal import Decimal

from app.utils.logger import logger


class MetricType(str, Enum):
    """Types of business metrics."""
    ENGAGEMENT = "engagement"
    CONVERSION = "conversion"
    REVENUE = "revenue"
    RETENTION = "retention"
    SATISFACTION = "satisfaction"
    PERFORMANCE = "performance"
    GROWTH = "growth"


class MetricCategory(str, Enum):
    """Business metric categories."""
    USER_ACQUISITION = "user_acquisition"
    USER_ENGAGEMENT = "user_engagement"
    MATCHING_PERFORMANCE = "matching_performance"
    BUSINESS_IMPACT = "business_impact"
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    CUSTOMER_SUCCESS = "customer_success"


class TimeGranularity(str, Enum):
    """Time granularity for metrics."""
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class BusinessEvent:
    """Business event for metrics tracking."""
    event_id: str
    event_type: str
    user_id: Optional[str]
    session_id: Optional[str]
    timestamp: datetime
    properties: Dict[str, Any]
    revenue_impact: Optional[Decimal] = None
    conversion_value: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MetricDefinition:
    """Definition of a business metric."""
    metric_id: str
    name: str
    description: str
    metric_type: MetricType
    category: MetricCategory
    calculation_method: str
    target_value: Optional[float] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    unit: str = "count"
    is_higher_better: bool = True


@dataclass
class MetricValue:
    """Calculated metric value."""
    metric_id: str
    timestamp: datetime
    value: float
    granularity: TimeGranularity
    dimensions: Dict[str, str]
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class BusinessReport:
    """Business analytics report."""
    report_id: str
    report_type: str
    timestamp: datetime
    time_range: Tuple[datetime, datetime]
    metrics: Dict[str, MetricValue]
    insights: List[str]
    recommendations: List[str]
    executive_summary: str


class BusinessMetricsCollector:
    """
    Comprehensive business metrics collector for tracking KPIs,
    conversion rates, revenue impact, and business intelligence.
    """
    
    def __init__(
        self,
        collection_interval_seconds: int = 300,
        retention_days: int = 365,
        enable_real_time: bool = True
    ):
        """Initialize business metrics collector."""
        self.collection_interval_seconds = collection_interval_seconds
        self.retention_days = retention_days
        self.enable_real_time = enable_real_time
        
        # Event storage
        self.events: deque = deque()
        self.processed_events: Dict[str, List[BusinessEvent]] = defaultdict(list)
        
        # Metric definitions
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self._setup_default_metrics()
        
        # Calculated metrics
        self.current_metrics: Dict[str, MetricValue] = {}
        self.metric_history: Dict[str, List[MetricValue]] = defaultdict(list)
        self.real_time_metrics: Dict[str, float] = defaultdict(float)
        
        # User tracking
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.user_journeys: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.cohort_data: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Business intelligence
        self.segments: Dict[str, Dict[str, Any]] = {}
        self.conversion_funnels: Dict[str, List[Dict[str, Any]]] = {}
        self.revenue_tracking: Dict[str, Decimal] = defaultdict(Decimal)
        
        # Processing tasks
        self.collection_task: Optional[asyncio.Task] = None
        self.reporting_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.metric_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.alert_callbacks: List[Callable] = []
        
        # Performance stats
        self.collector_stats = {
            'total_events_processed': 0,
            'metrics_calculated': 0,
            'reports_generated': 0,
            'alerts_triggered': 0,
            'active_users_tracked': 0,
            'revenue_tracked': Decimal('0.00')
        }
        
        logger.info("BusinessMetricsCollector initialized")
    
    def _setup_default_metrics(self):
        """Setup default business metrics definitions."""
        try:
            default_metrics = [
                # User Acquisition Metrics
                MetricDefinition(
                    metric_id="new_user_registrations",
                    name="New User Registrations",
                    description="Number of new users registered",
                    metric_type=MetricType.ENGAGEMENT,
                    category=MetricCategory.USER_ACQUISITION,
                    calculation_method="count",
                    target_value=100.0,
                    unit="users"
                ),
                MetricDefinition(
                    metric_id="user_activation_rate",
                    name="User Activation Rate",
                    description="Percentage of users who complete first meaningful action",
                    metric_type=MetricType.CONVERSION,
                    category=MetricCategory.USER_ACQUISITION,
                    calculation_method="percentage",
                    target_value=0.8,
                    threshold_warning=0.6,
                    threshold_critical=0.4,
                    unit="percentage"
                ),
                
                # Engagement Metrics
                MetricDefinition(
                    metric_id="daily_active_users",
                    name="Daily Active Users",
                    description="Number of unique users active per day",
                    metric_type=MetricType.ENGAGEMENT,
                    category=MetricCategory.USER_ENGAGEMENT,
                    calculation_method="distinct_count",
                    target_value=1000.0,
                    unit="users"
                ),
                MetricDefinition(
                    metric_id="session_duration_avg",
                    name="Average Session Duration",
                    description="Average time users spend per session",
                    metric_type=MetricType.ENGAGEMENT,
                    category=MetricCategory.USER_ENGAGEMENT,
                    calculation_method="average",
                    target_value=600.0,  # 10 minutes
                    unit="seconds"
                ),
                MetricDefinition(
                    metric_id="pages_per_session",
                    name="Pages Per Session",
                    description="Average number of pages viewed per session",
                    metric_type=MetricType.ENGAGEMENT,
                    category=MetricCategory.USER_ENGAGEMENT,
                    calculation_method="average",
                    target_value=5.0,
                    unit="pages"
                ),
                
                # Matching Performance Metrics
                MetricDefinition(
                    metric_id="job_match_success_rate",
                    name="Job Match Success Rate",
                    description="Percentage of job matches that result in applications",
                    metric_type=MetricType.CONVERSION,
                    category=MetricCategory.MATCHING_PERFORMANCE,
                    calculation_method="conversion_rate",
                    target_value=0.25,
                    threshold_warning=0.15,
                    threshold_critical=0.10,
                    unit="percentage"
                ),
                MetricDefinition(
                    metric_id="match_quality_score",
                    name="Match Quality Score",
                    description="Average quality score of job matches",
                    metric_type=MetricType.PERFORMANCE,
                    category=MetricCategory.MATCHING_PERFORMANCE,
                    calculation_method="average",
                    target_value=0.8,
                    threshold_warning=0.6,
                    threshold_critical=0.4,
                    unit="score"
                ),
                MetricDefinition(
                    metric_id="time_to_first_match",
                    name="Time to First Match",
                    description="Average time from registration to first job match",
                    metric_type=MetricType.PERFORMANCE,
                    category=MetricCategory.MATCHING_PERFORMANCE,
                    calculation_method="average",
                    target_value=300.0,  # 5 minutes
                    is_higher_better=False,
                    unit="seconds"
                ),
                
                # Business Impact Metrics
                MetricDefinition(
                    metric_id="application_conversion_rate",
                    name="Application Conversion Rate",
                    description="Percentage of job views that result in applications",
                    metric_type=MetricType.CONVERSION,
                    category=MetricCategory.BUSINESS_IMPACT,
                    calculation_method="conversion_rate",
                    target_value=0.15,
                    threshold_warning=0.10,
                    threshold_critical=0.05,
                    unit="percentage"
                ),
                MetricDefinition(
                    metric_id="hire_success_rate",
                    name="Hire Success Rate",
                    description="Percentage of applications that result in hires",
                    metric_type=MetricType.CONVERSION,
                    category=MetricCategory.BUSINESS_IMPACT,
                    calculation_method="conversion_rate",
                    target_value=0.05,
                    threshold_warning=0.03,
                    threshold_critical=0.01,
                    unit="percentage"
                ),
                MetricDefinition(
                    metric_id="revenue_per_user",
                    name="Revenue Per User",
                    description="Average revenue generated per active user",
                    metric_type=MetricType.REVENUE,
                    category=MetricCategory.BUSINESS_IMPACT,
                    calculation_method="average",
                    target_value=50.0,
                    unit="currency"
                ),
                
                # Customer Success Metrics
                MetricDefinition(
                    metric_id="user_satisfaction_score",
                    name="User Satisfaction Score",
                    description="Average user satisfaction rating",
                    metric_type=MetricType.SATISFACTION,
                    category=MetricCategory.CUSTOMER_SUCCESS,
                    calculation_method="average",
                    target_value=4.0,
                    threshold_warning=3.5,
                    threshold_critical=3.0,
                    unit="rating"
                ),
                MetricDefinition(
                    metric_id="user_retention_rate",
                    name="User Retention Rate",
                    description="Percentage of users retained after 30 days",
                    metric_type=MetricType.RETENTION,
                    category=MetricCategory.CUSTOMER_SUCCESS,
                    calculation_method="retention_rate",
                    target_value=0.7,
                    threshold_warning=0.5,
                    threshold_critical=0.3,
                    unit="percentage"
                )
            ]
            
            for metric_def in default_metrics:
                self.metric_definitions[metric_def.metric_id] = metric_def
            
            logger.info(f"Setup {len(default_metrics)} default business metrics")
            
        except Exception as e:
            logger.error(f"Failed to setup default metrics: {str(e)}")
    
    async def initialize(self):
        """Initialize metrics collection tasks."""
        try:
            # Start collection task
            self.collection_task = asyncio.create_task(self._collection_loop())
            
            # Start reporting task
            self.reporting_task = asyncio.create_task(self._reporting_loop())
            
            logger.info("BusinessMetricsCollector started")
            
        except Exception as e:
            logger.error(f"BusinessMetricsCollector initialization failed: {str(e)}")
            raise
    
    async def track_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        revenue_impact: Optional[Union[float, Decimal]] = None,
        conversion_value: Optional[float] = None
    ) -> str:
        """
        Track a business event.
        
        Args:
            event_type: Type of business event
            user_id: User identifier
            session_id: Session identifier
            properties: Event properties
            revenue_impact: Revenue impact of the event
            conversion_value: Conversion value
            
        Returns:
            Event ID
        """
        try:
            event_id = f"{event_type}_{int(time.time() * 1000)}_{len(self.events)}"
            
            # Convert revenue impact to Decimal
            if revenue_impact is not None and not isinstance(revenue_impact, Decimal):
                revenue_impact = Decimal(str(revenue_impact))
            
            event = BusinessEvent(
                event_id=event_id,
                event_type=event_type,
                user_id=user_id,
                session_id=session_id,
                timestamp=datetime.now(),
                properties=properties or {},
                revenue_impact=revenue_impact,
                conversion_value=conversion_value
            )
            
            # Add to processing queue
            self.events.append(event)
            
            # Real-time processing
            if self.enable_real_time:
                await self._process_event_real_time(event)
            
            return event_id
            
        except Exception as e:
            logger.error(f"Event tracking failed: {str(e)}")
            return ""
    
    async def _process_event_real_time(self, event: BusinessEvent):
        """Process event in real-time for immediate metrics updates."""
        try:
            # Store event
            date_key = event.timestamp.date().isoformat()
            self.processed_events[date_key].append(event)
            
            # Update real-time metrics
            await self._update_real_time_metrics(event)
            
            # Track user session
            await self._track_user_session(event)
            
            # Track user journey
            await self._track_user_journey(event)
            
            # Update revenue tracking
            if event.revenue_impact:
                self.revenue_tracking[date_key] += event.revenue_impact
                self.collector_stats['revenue_tracked'] += event.revenue_impact
            
            self.collector_stats['total_events_processed'] += 1
            
        except Exception as e:
            logger.error(f"Real-time event processing failed: {str(e)}")
    
    async def _update_real_time_metrics(self, event: BusinessEvent):
        """Update real-time metrics based on event."""
        try:
            event_type = event.event_type
            
            # Basic event counting
            self.real_time_metrics[f"{event_type}_count"] += 1
            self.real_time_metrics['total_events'] += 1
            
            # User metrics
            if event.user_id:
                today_key = f"active_users_{datetime.now().date()}"
                # Track unique users (simplified)
                user_key = f"user_{event.user_id}_{today_key}"
                if user_key not in self.real_time_metrics:
                    self.real_time_metrics[today_key] += 1
                    self.real_time_metrics[user_key] = 1
            
            # Conversion tracking
            if event_type in ['job_view', 'job_application', 'user_registration', 'profile_complete']:
                await self._update_conversion_metrics(event)
            
            # Revenue tracking
            if event.revenue_impact:
                self.real_time_metrics['total_revenue'] += float(event.revenue_impact)
            
            # Performance metrics
            if 'response_time' in event.properties:
                response_time = event.properties['response_time']
                self._update_running_average('avg_response_time', response_time)
            
            if 'match_score' in event.properties:
                match_score = event.properties['match_score']
                self._update_running_average('avg_match_score', match_score)
            
        except Exception as e:
            logger.error(f"Real-time metrics update failed: {str(e)}")
    
    def _update_running_average(self, metric_key: str, new_value: float):
        """Update running average for a metric."""
        try:
            count_key = f"{metric_key}_count"
            
            current_avg = self.real_time_metrics.get(metric_key, 0.0)
            current_count = self.real_time_metrics.get(count_key, 0)
            
            new_count = current_count + 1
            new_avg = ((current_avg * current_count) + new_value) / new_count
            
            self.real_time_metrics[metric_key] = new_avg
            self.real_time_metrics[count_key] = new_count
            
        except Exception as e:
            logger.error(f"Running average update failed for {metric_key}: {str(e)}")
    
    async def _update_conversion_metrics(self, event: BusinessEvent):
        """Update conversion-related metrics."""
        try:
            event_type = event.event_type
            user_id = event.user_id
            
            if not user_id:
                return
            
            # Track conversion funnel
            funnel_key = f"conversion_funnel_{user_id}"
            
            if funnel_key not in self.conversion_funnels:
                self.conversion_funnels[funnel_key] = []
            
            funnel_step = {
                'event_type': event_type,
                'timestamp': event.timestamp,
                'properties': event.properties
            }
            
            self.conversion_funnels[funnel_key].append(funnel_step)
            
            # Calculate conversion rates
            if event_type == 'job_application':
                # Check if user had job views
                user_funnel = self.conversion_funnels[funnel_key]
                job_views = [step for step in user_funnel if step['event_type'] == 'job_view']
                
                if job_views:
                    self.real_time_metrics['job_view_to_application_conversions'] += 1
            
            elif event_type == 'user_registration':
                self.real_time_metrics['new_registrations'] += 1
            
            elif event_type == 'profile_complete':
                # Check if user registered recently
                user_funnel = self.conversion_funnels[funnel_key]
                registrations = [step for step in user_funnel if step['event_type'] == 'user_registration']
                
                if registrations:
                    self.real_time_metrics['registration_to_activation_conversions'] += 1
            
        except Exception as e:
            logger.error(f"Conversion metrics update failed: {str(e)}")
    
    async def _track_user_session(self, event: BusinessEvent):
        """Track user session data."""
        try:
            if not event.user_id or not event.session_id:
                return
            
            session_key = f"{event.user_id}_{event.session_id}"
            
            if session_key not in self.user_sessions:
                self.user_sessions[session_key] = {
                    'user_id': event.user_id,
                    'session_id': event.session_id,
                    'start_time': event.timestamp,
                    'last_activity': event.timestamp,
                    'event_count': 0,
                    'page_views': 0,
                    'events': []
                }
            
            session = self.user_sessions[session_key]
            session['last_activity'] = event.timestamp
            session['event_count'] += 1
            session['events'].append({
                'event_type': event.event_type,
                'timestamp': event.timestamp,
                'properties': event.properties
            })
            
            if event.event_type == 'page_view':
                session['page_views'] += 1
            
            # Update session duration
            session['duration_seconds'] = (
                session['last_activity'] - session['start_time']
            ).total_seconds()
            
        except Exception as e:
            logger.error(f"User session tracking failed: {str(e)}")
    
    async def _track_user_journey(self, event: BusinessEvent):
        """Track user journey for behavior analysis."""
        try:
            if not event.user_id:
                return
            
            journey_event = {
                'event_type': event.event_type,
                'timestamp': event.timestamp,
                'session_id': event.session_id,
                'properties': event.properties,
                'revenue_impact': float(event.revenue_impact) if event.revenue_impact else None
            }
            
            self.user_journeys[event.user_id].append(journey_event)
            
            # Keep only recent journey events (last 100)
            if len(self.user_journeys[event.user_id]) > 100:
                self.user_journeys[event.user_id] = self.user_journeys[event.user_id][-100:]
            
        except Exception as e:
            logger.error(f"User journey tracking failed: {str(e)}")
    
    async def _collection_loop(self):
        """Background metrics collection loop."""
        try:
            while True:
                await asyncio.sleep(self.collection_interval_seconds)
                
                try:
                    await self._calculate_metrics()
                    await self._check_metric_thresholds()
                    
                except Exception as e:
                    logger.error(f"Collection cycle error: {str(e)}")
                
        except asyncio.CancelledError:
            logger.info("Metrics collection loop cancelled")
        except Exception as e:
            logger.error(f"Collection loop error: {str(e)}")
    
    async def _calculate_metrics(self):
        """Calculate all defined business metrics."""
        try:
            current_time = datetime.now()
            
            for metric_id, metric_def in self.metric_definitions.items():
                try:
                    # Calculate metric value
                    value = await self._calculate_metric_value(metric_def, current_time)
                    
                    if value is not None:
                        metric_value = MetricValue(
                            metric_id=metric_id,
                            timestamp=current_time,
                            value=value,
                            granularity=TimeGranularity.REAL_TIME,
                            dimensions={}
                        )
                        
                        # Store metric
                        self.current_metrics[metric_id] = metric_value
                        self.metric_history[metric_id].append(metric_value)
                        
                        # Keep history manageable
                        if len(self.metric_history[metric_id]) > 1000:
                            self.metric_history[metric_id] = self.metric_history[metric_id][-1000:]
                        
                        # Trigger callbacks
                        for callback in self.metric_callbacks[metric_id]:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(metric_value)
                                else:
                                    callback(metric_value)
                            except Exception as callback_error:
                                logger.error(f"Metric callback failed: {str(callback_error)}")
                
                except Exception as metric_error:
                    logger.error(f"Failed to calculate metric {metric_id}: {str(metric_error)}")
            
            self.collector_stats['metrics_calculated'] += len(self.metric_definitions)
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {str(e)}")
    
    async def _calculate_metric_value(
        self,
        metric_def: MetricDefinition,
        timestamp: datetime
    ) -> Optional[float]:
        """Calculate the value for a specific metric."""
        try:
            metric_id = metric_def.metric_id
            calculation_method = metric_def.calculation_method
            
            # Get relevant data based on metric
            if metric_id == "daily_active_users":
                today_key = f"active_users_{timestamp.date()}"
                return self.real_time_metrics.get(today_key, 0.0)
            
            elif metric_id == "new_user_registrations":
                return self.real_time_metrics.get("user_registration_count", 0.0)
            
            elif metric_id == "session_duration_avg":
                # Calculate from active sessions
                if self.user_sessions:
                    durations = [
                        session['duration_seconds'] for session in self.user_sessions.values()
                        if session.get('duration_seconds', 0) > 0
                    ]
                    return np.mean(durations) if durations else 0.0
                return 0.0
            
            elif metric_id == "pages_per_session":
                if self.user_sessions:
                    page_counts = [
                        session['page_views'] for session in self.user_sessions.values()
                        if session['page_views'] > 0
                    ]
                    return np.mean(page_counts) if page_counts else 0.0
                return 0.0
            
            elif metric_id == "job_match_success_rate":
                job_views = self.real_time_metrics.get("job_view_count", 0)
                applications = self.real_time_metrics.get("job_application_count", 0)
                return (applications / job_views) if job_views > 0 else 0.0
            
            elif metric_id == "match_quality_score":
                return self.real_time_metrics.get("avg_match_score", 0.0)
            
            elif metric_id == "application_conversion_rate":
                job_views = self.real_time_metrics.get("job_view_count", 0)
                applications = self.real_time_metrics.get("job_application_count", 0)
                return (applications / job_views) if job_views > 0 else 0.0
            
            elif metric_id == "user_activation_rate":
                registrations = self.real_time_metrics.get("new_registrations", 0)
                activations = self.real_time_metrics.get("registration_to_activation_conversions", 0)
                return (activations / registrations) if registrations > 0 else 0.0
            
            elif metric_id == "revenue_per_user":
                total_revenue = self.real_time_metrics.get("total_revenue", 0.0)
                active_users = self.real_time_metrics.get(f"active_users_{timestamp.date()}", 1)
                return total_revenue / max(active_users, 1)
            
            elif metric_id == "user_satisfaction_score":
                # This would come from feedback/rating events
                return self.real_time_metrics.get("avg_satisfaction_score", 0.0)
            
            elif metric_id == "time_to_first_match":
                return self.real_time_metrics.get("avg_time_to_first_match", 0.0)
            
            # Default calculation methods
            elif calculation_method == "count":
                return self.real_time_metrics.get(f"{metric_id}_count", 0.0)
            
            elif calculation_method == "average":
                return self.real_time_metrics.get(f"avg_{metric_id}", 0.0)
            
            else:
                # Generic calculation
                return self.real_time_metrics.get(metric_id, 0.0)
            
        except Exception as e:
            logger.error(f"Metric value calculation failed for {metric_def.metric_id}: {str(e)}")
            return None
    
    async def _check_metric_thresholds(self):
        """Check metric values against thresholds and trigger alerts."""
        try:
            for metric_id, metric_value in self.current_metrics.items():
                metric_def = self.metric_definitions.get(metric_id)
                
                if not metric_def:
                    continue
                
                value = metric_value.value
                
                # Check thresholds
                if metric_def.threshold_critical is not None:
                    if (metric_def.is_higher_better and value < metric_def.threshold_critical) or \
                       (not metric_def.is_higher_better and value > metric_def.threshold_critical):
                        await self._trigger_metric_alert(metric_def, metric_value, "critical")
                
                elif metric_def.threshold_warning is not None:
                    if (metric_def.is_higher_better and value < metric_def.threshold_warning) or \
                       (not metric_def.is_higher_better and value > metric_def.threshold_warning):
                        await self._trigger_metric_alert(metric_def, metric_value, "warning")
            
        except Exception as e:
            logger.error(f"Threshold checking failed: {str(e)}")
    
    async def _trigger_metric_alert(
        self,
        metric_def: MetricDefinition,
        metric_value: MetricValue,
        severity: str
    ):
        """Trigger alert for metric threshold violation."""
        try:
            alert_data = {
                'metric_id': metric_def.metric_id,
                'metric_name': metric_def.name,
                'current_value': metric_value.value,
                'threshold_type': severity,
                'threshold_value': (
                    metric_def.threshold_critical if severity == "critical" 
                    else metric_def.threshold_warning
                ),
                'is_higher_better': metric_def.is_higher_better,
                'timestamp': metric_value.timestamp.isoformat(),
                'unit': metric_def.unit
            }
            
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert_data)
                    else:
                        callback(alert_data)
                except Exception as callback_error:
                    logger.error(f"Alert callback failed: {str(callback_error)}")
            
            self.collector_stats['alerts_triggered'] += 1
            
            logger.warning(
                f"Metric alert ({severity}): {metric_def.name} = {metric_value.value} "
                f"(threshold: {alert_data['threshold_value']})"
            )
            
        except Exception as e:
            logger.error(f"Alert triggering failed: {str(e)}")
    
    async def _reporting_loop(self):
        """Background reporting loop."""
        try:
            while True:
                await asyncio.sleep(3600)  # Generate reports every hour
                
                try:
                    await self._generate_automated_reports()
                    
                except Exception as e:
                    logger.error(f"Reporting cycle error: {str(e)}")
                
        except asyncio.CancelledError:
            logger.info("Reporting loop cancelled")
        except Exception as e:
            logger.error(f"Reporting loop error: {str(e)}")
    
    async def _generate_automated_reports(self):
        """Generate automated business reports."""
        try:
            current_time = datetime.now()
            
            # Daily report
            if current_time.hour == 9:  # Generate at 9 AM
                await self.generate_report("daily_summary", timedelta(days=1))
            
            # Weekly report
            if current_time.weekday() == 0 and current_time.hour == 9:  # Monday 9 AM
                await self.generate_report("weekly_summary", timedelta(weeks=1))
            
            # Monthly report
            if current_time.day == 1 and current_time.hour == 9:  # First day of month
                await self.generate_report("monthly_summary", timedelta(days=30))
            
        except Exception as e:
            logger.error(f"Automated report generation failed: {str(e)}")
    
    async def generate_report(
        self,
        report_type: str,
        time_range: timedelta,
        custom_metrics: Optional[List[str]] = None
    ) -> BusinessReport:
        """
        Generate a comprehensive business report.
        
        Args:
            report_type: Type of report to generate
            time_range: Time range for the report
            custom_metrics: Optional list of specific metrics to include
            
        Returns:
            Generated business report
        """
        try:
            end_time = datetime.now()
            start_time = end_time - time_range
            
            # Collect metrics for the time range
            report_metrics = {}
            metrics_to_include = custom_metrics or list(self.metric_definitions.keys())
            
            for metric_id in metrics_to_include:
                if metric_id in self.current_metrics:
                    report_metrics[metric_id] = self.current_metrics[metric_id]
            
            # Generate insights
            insights = await self._generate_report_insights(report_metrics, time_range)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(report_metrics, insights)
            
            # Generate executive summary
            executive_summary = await self._generate_executive_summary(
                report_metrics, insights, time_range
            )
            
            report = BusinessReport(
                report_id=f"{report_type}_{int(time.time())}",
                report_type=report_type,
                timestamp=end_time,
                time_range=(start_time, end_time),
                metrics=report_metrics,
                insights=insights,
                recommendations=recommendations,
                executive_summary=executive_summary
            )
            
            self.collector_stats['reports_generated'] += 1
            
            logger.info(f"Generated {report_type} report with {len(report_metrics)} metrics")
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise
    
    async def _generate_report_insights(
        self,
        metrics: Dict[str, MetricValue],
        time_range: timedelta
    ) -> List[str]:
        """Generate insights from metrics data."""
        try:
            insights = []
            
            # User engagement insights
            if "daily_active_users" in metrics:
                dau = metrics["daily_active_users"].value
                if dau > 1000:
                    insights.append(f"Strong user engagement with {dau:.0f} daily active users")
                elif dau < 100:
                    insights.append("User engagement is below target - consider activation campaigns")
            
            # Conversion insights
            if "job_match_success_rate" in metrics:
                success_rate = metrics["job_match_success_rate"].value
                if success_rate > 0.3:
                    insights.append(f"Excellent job matching performance with {success_rate:.1%} success rate")
                elif success_rate < 0.15:
                    insights.append("Job matching success rate is below target - review matching algorithm")
            
            # Revenue insights
            if "revenue_per_user" in metrics:
                rpu = metrics["revenue_per_user"].value
                if rpu > 75:
                    insights.append(f"Strong monetization with ${rpu:.2f} revenue per user")
                elif rpu < 25:
                    insights.append("Revenue per user is below target - explore monetization opportunities")
            
            # Performance insights
            if "match_quality_score" in metrics:
                quality_score = metrics["match_quality_score"].value
                if quality_score > 0.8:
                    insights.append(f"High-quality matches with average score of {quality_score:.2f}")
                elif quality_score < 0.6:
                    insights.append("Match quality needs improvement - consider model optimization")
            
            return insights
            
        except Exception as e:
            logger.error(f"Insight generation failed: {str(e)}")
            return []
    
    async def _generate_recommendations(
        self,
        metrics: Dict[str, MetricValue],
        insights: List[str]
    ) -> List[str]:
        """Generate actionable recommendations."""
        try:
            recommendations = []
            
            # Check each metric against its target
            for metric_id, metric_value in metrics.items():
                metric_def = self.metric_definitions.get(metric_id)
                if not metric_def or not metric_def.target_value:
                    continue
                
                value = metric_value.value
                target = metric_def.target_value
                
                if metric_def.is_higher_better and value < target * 0.8:
                    recommendations.append(
                        f"Improve {metric_def.name}: currently {value:.2f}, target {target:.2f}"
                    )
                elif not metric_def.is_higher_better and value > target * 1.2:
                    recommendations.append(
                        f"Optimize {metric_def.name}: currently {value:.2f}, target {target:.2f}"
                    )
            
            # Strategic recommendations based on patterns
            if any("below target" in insight for insight in insights):
                recommendations.append("Conduct comprehensive performance review and optimization")
            
            if any("Strong" in insight or "Excellent" in insight for insight in insights):
                recommendations.append("Scale successful strategies and share best practices")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            return []
    
    async def _generate_executive_summary(
        self,
        metrics: Dict[str, MetricValue],
        insights: List[str],
        time_range: timedelta
    ) -> str:
        """Generate executive summary."""
        try:
            # Key metrics summary
            key_metrics = []
            
            if "daily_active_users" in metrics:
                dau = metrics["daily_active_users"].value
                key_metrics.append(f"{dau:.0f} daily active users")
            
            if "job_match_success_rate" in metrics:
                success_rate = metrics["job_match_success_rate"].value
                key_metrics.append(f"{success_rate:.1%} job match success rate")
            
            if "revenue_per_user" in metrics:
                rpu = metrics["revenue_per_user"].value
                key_metrics.append(f"${rpu:.2f} revenue per user")
            
            summary_parts = []
            
            # Period summary
            period_str = f"{time_range.days} days" if time_range.days > 0 else f"{time_range.seconds//3600} hours"
            summary_parts.append(f"Business performance summary for the past {period_str}.")
            
            # Key metrics
            if key_metrics:
                summary_parts.append(f"Key metrics: {', '.join(key_metrics)}.")
            
            # Top insights
            if insights:
                top_insights = insights[:2]  # Top 2 insights
                summary_parts.append(f"Key insights: {' '.join(top_insights)}")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Executive summary generation failed: {str(e)}")
            return f"Business performance summary for the specified period with {len(metrics)} metrics tracked."
    
    def register_metric_callback(self, metric_id: str, callback: Callable) -> bool:
        """Register callback for specific metric updates."""
        try:
            self.metric_callbacks[metric_id].append(callback)
            logger.info(f"Registered callback for metric {metric_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register metric callback: {str(e)}")
            return False
    
    def register_alert_callback(self, callback: Callable) -> bool:
        """Register callback for metric alerts."""
        try:
            self.alert_callbacks.append(callback)
            logger.info("Registered alert callback")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register alert callback: {str(e)}")
            return False
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        try:
            return {
                'collector_stats': {
                    k: float(v) if isinstance(v, Decimal) else v
                    for k, v in self.collector_stats.items()
                },
                'current_metrics': {
                    metric_id: {
                        'value': metric.value,
                        'timestamp': metric.timestamp.isoformat(),
                        'unit': self.metric_definitions[metric_id].unit
                    }
                    for metric_id, metric in self.current_metrics.items()
                },
                'real_time_metrics': dict(self.real_time_metrics),
                'active_users': len(self.user_sessions),
                'total_revenue': float(sum(self.revenue_tracking.values())),
                'metrics_definitions_count': len(self.metric_definitions),
                'conversion_funnels_tracked': len(self.conversion_funnels)
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Clean up business metrics collector resources."""
        try:
            # Cancel tasks
            if self.collection_task:
                self.collection_task.cancel()
                try:
                    await self.collection_task
                except asyncio.CancelledError:
                    pass
            
            if self.reporting_task:
                self.reporting_task.cancel()
                try:
                    await self.reporting_task
                except asyncio.CancelledError:
                    pass
            
            # Clear data structures
            self.events.clear()
            self.processed_events.clear()
            self.current_metrics.clear()
            self.metric_history.clear()
            self.real_time_metrics.clear()
            self.user_sessions.clear()
            self.user_journeys.clear()
            self.cohort_data.clear()
            self.segments.clear()
            self.conversion_funnels.clear()
            self.revenue_tracking.clear()
            self.metric_callbacks.clear()
            self.alert_callbacks.clear()
            
            logger.info("BusinessMetricsCollector cleanup completed")
            
        except Exception as e:
            logger.error(f"BusinessMetricsCollector cleanup failed: {str(e)}")