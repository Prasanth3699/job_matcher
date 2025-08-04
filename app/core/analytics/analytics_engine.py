"""
Analytics Engine for real-time data processing and insights generation.

This module provides advanced analytics capabilities including real-time event
processing, pattern detection, trend analysis, and automated insights.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from scipy import stats
import heapq

from app.utils.logger import logger


class EventType(str, Enum):
    """Types of analytics events."""
    USER_ACTION = "user_action"
    MODEL_PREDICTION = "model_prediction"
    PERFORMANCE_METRIC = "performance_metric"
    BUSINESS_EVENT = "business_event"
    SYSTEM_EVENT = "system_event"
    ERROR_EVENT = "error_event"


class AnalyticsLevel(str, Enum):
    """Analytics processing levels."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    HISTORICAL = "historical"


class InsightType(str, Enum):
    """Types of generated insights."""
    TREND = "trend"
    ANOMALY = "anomaly"
    PATTERN = "pattern"
    CORRELATION = "correlation"
    PREDICTION = "prediction"
    RECOMMENDATION = "recommendation"


@dataclass
class AnalyticsEvent:
    """Analytics event structure."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    data: Dict[str, Any]
    tags: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'data': self.data,
            'tags': self.tags,
            'metadata': self.metadata
        }


@dataclass
class Insight:
    """Generated analytics insight."""
    insight_id: str
    insight_type: InsightType
    title: str
    description: str
    confidence: float
    impact_score: float
    data: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'insight_id': self.insight_id,
            'insight_type': self.insight_type.value,
            'title': self.title,
            'description': self.description,
            'confidence': self.confidence,
            'impact_score': self.impact_score,
            'data': self.data,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags
        }


@dataclass
class AnalyticsQuery:
    """Analytics query structure."""
    query_id: str
    dimensions: List[str]
    metrics: List[str]
    filters: Dict[str, Any]
    time_range: Tuple[datetime, datetime]
    aggregation: str
    grouping: Optional[List[str]] = None


class AnalyticsEngine:
    """
    Advanced analytics engine for real-time event processing,
    pattern detection, and automated insights generation.
    """
    
    def __init__(
        self,
        enable_real_time: bool = True,
        batch_size: int = 1000,
        insight_generation_interval: int = 300,
        retention_days: int = 30
    ):
        """Initialize analytics engine."""
        self.enable_real_time = enable_real_time
        self.batch_size = batch_size
        self.insight_generation_interval = insight_generation_interval
        self.retention_days = retention_days
        
        # Event storage and processing
        self.event_queue: deque = deque()
        self.processed_events: Dict[str, List[AnalyticsEvent]] = defaultdict(list)
        self.event_indexes: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
        # Real-time metrics
        self.real_time_metrics: Dict[str, Any] = defaultdict(float)
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Insights and patterns
        self.insights: List[Insight] = []
        self.patterns: Dict[str, Dict[str, Any]] = {}
        self.anomalies: List[Dict[str, Any]] = []
        
        # Analytics processors
        self.processors: Dict[EventType, List[Callable]] = defaultdict(list)
        self.insight_generators: List[Callable] = []
        
        # Processing tasks
        self.processing_task: Optional[asyncio.Task] = None
        self.insight_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.analytics_stats = {
            'total_events_processed': 0,
            'events_by_type': defaultdict(int),
            'processing_latency_ms': 0.0,
            'insights_generated': 0,
            'anomalies_detected': 0,
            'patterns_discovered': 0,
            'queries_executed': 0
        }
        
        # Configuration
        self.config = {
            'anomaly_detection_threshold': 2.0,  # Z-score threshold
            'pattern_min_support': 0.1,           # Minimum pattern support
            'correlation_threshold': 0.7,         # Correlation threshold
            'trend_window_minutes': 60,           # Trend analysis window
            'insight_confidence_threshold': 0.6   # Minimum insight confidence
        }
        
        logger.info("AnalyticsEngine initialized")
    
    async def initialize(self):
        """Initialize analytics engine with processing tasks."""
        try:
            if self.enable_real_time:
                # Start real-time processing task
                self.processing_task = asyncio.create_task(self._processing_loop())
                
                # Start insight generation task
                self.insight_task = asyncio.create_task(self._insight_generation_loop())
            
            logger.info("AnalyticsEngine processing started")
            
        except Exception as e:
            logger.error(f"AnalyticsEngine initialization failed: {str(e)}")
            raise
    
    async def track_event(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Track an analytics event.
        
        Args:
            event_type: Type of event
            data: Event data
            user_id: User identifier
            session_id: Session identifier
            tags: Event tags
            metadata: Additional metadata
            
        Returns:
            Event ID
        """
        try:
            event_id = f"{event_type.value}_{int(time.time() * 1000)}_{len(self.event_queue)}"
            
            event = AnalyticsEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.now(),
                user_id=user_id,
                session_id=session_id,
                data=data,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # Add to processing queue
            self.event_queue.append(event)
            
            # Real-time processing if enabled
            if self.enable_real_time:
                await self._process_event_real_time(event)
            
            return event_id
            
        except Exception as e:
            logger.error(f"Event tracking failed: {str(e)}")
            return ""
    
    async def _process_event_real_time(self, event: AnalyticsEvent):
        """Process event in real-time."""
        try:
            # Store event
            event_key = f"{event.event_type.value}_{event.timestamp.date()}"
            self.processed_events[event_key].append(event)
            
            # Update indexes
            await self._update_event_indexes(event)
            
            # Update real-time metrics
            await self._update_real_time_metrics(event)
            
            # Run event processors
            processors = self.processors.get(event.event_type, [])
            for processor in processors:
                try:
                    if asyncio.iscoroutinefunction(processor):
                        await processor(event)
                    else:
                        processor(event)
                except Exception as proc_error:
                    logger.error(f"Event processor failed: {str(proc_error)}")
            
            # Check for real-time anomalies
            await self._check_real_time_anomalies(event)
            
            self.analytics_stats['total_events_processed'] += 1
            self.analytics_stats['events_by_type'][event.event_type.value] += 1
            
        except Exception as e:
            logger.error(f"Real-time event processing failed: {str(e)}")
    
    async def _update_event_indexes(self, event: AnalyticsEvent):
        """Update event indexes for fast querying."""
        try:
            event_key = f"{event.event_type.value}_{event.timestamp.date()}"
            
            # Index by user
            if event.user_id:
                self.event_indexes[event_key]['user_id'].append(event.event_id)
            
            # Index by session
            if event.session_id:
                self.event_indexes[event_key]['session_id'].append(event.event_id)
            
            # Index by tags
            for tag in event.tags:
                self.event_indexes[event_key][f'tag_{tag}'].append(event.event_id)
            
        except Exception as e:
            logger.error(f"Event indexing failed: {str(e)}")
    
    async def _update_real_time_metrics(self, event: AnalyticsEvent):
        """Update real-time metrics based on event."""
        try:
            current_time = datetime.now()
            
            # Event count metrics
            self.real_time_metrics[f"{event.event_type.value}_count"] += 1
            self.real_time_metrics['total_events'] += 1
            
            # User metrics
            if event.user_id:
                self.real_time_metrics['unique_users'] = len(
                    set(e.user_id for events in self.processed_events.values() 
                        for e in events if e.user_id)
                )
            
            # Time-based metrics
            minute_key = current_time.replace(second=0, microsecond=0)
            minute_metric = f"events_per_minute_{minute_key.isoformat()}"
            self.real_time_metrics[minute_metric] += 1
            
            # Custom metric extraction from event data
            for key, value in event.data.items():
                if isinstance(value, (int, float)):
                    metric_key = f"{event.event_type.value}_{key}"
                    
                    # Update running average
                    if metric_key in self.real_time_metrics:
                        current_avg = self.real_time_metrics[metric_key]
                        count = self.analytics_stats['events_by_type'][event.event_type.value]
                        new_avg = ((current_avg * (count - 1)) + value) / count
                        self.real_time_metrics[metric_key] = new_avg
                    else:
                        self.real_time_metrics[metric_key] = value
                    
                    # Add to history for trend analysis
                    self.metric_history[metric_key].append((current_time, value))
            
        except Exception as e:
            logger.error(f"Real-time metrics update failed: {str(e)}")
    
    async def _check_real_time_anomalies(self, event: AnalyticsEvent):
        """Check for real-time anomalies in event data."""
        try:
            for key, value in event.data.items():
                if not isinstance(value, (int, float)):
                    continue
                
                metric_key = f"{event.event_type.value}_{key}"
                history = self.metric_history.get(metric_key, deque())
                
                if len(history) < 30:  # Need sufficient history
                    continue
                
                # Get recent values for comparison
                recent_values = [v for t, v in list(history)[-30:]]
                mean_value = np.mean(recent_values)
                std_value = np.std(recent_values)
                
                if std_value > 0:
                    z_score = abs(value - mean_value) / std_value
                    
                    if z_score > self.config['anomaly_detection_threshold']:
                        anomaly = {
                            'event_id': event.event_id,
                            'metric': metric_key,
                            'value': value,
                            'expected_range': (mean_value - 2*std_value, mean_value + 2*std_value),
                            'z_score': z_score,
                            'timestamp': event.timestamp,
                            'severity': 'high' if z_score > 3.0 else 'medium'
                        }
                        
                        self.anomalies.append(anomaly)
                        self.analytics_stats['anomalies_detected'] += 1
                        
                        logger.warning(f"Anomaly detected: {metric_key} = {value} (z-score: {z_score:.2f})")
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
    
    async def _processing_loop(self):
        """Background processing loop for batch event processing."""
        try:
            while True:
                await asyncio.sleep(1)  # Process every second
                
                if len(self.event_queue) >= self.batch_size:
                    await self._process_event_batch()
                
        except asyncio.CancelledError:
            logger.info("Analytics processing loop cancelled")
        except Exception as e:
            logger.error(f"Processing loop error: {str(e)}")
    
    async def _process_event_batch(self):
        """Process a batch of events."""
        try:
            start_time = time.time()
            
            # Extract batch from queue
            batch = []
            for _ in range(min(self.batch_size, len(self.event_queue))):
                if self.event_queue:
                    batch.append(self.event_queue.popleft())
            
            if not batch:
                return
            
            # Process batch
            for event in batch:
                if not self.enable_real_time:
                    await self._process_event_real_time(event)
            
            # Batch-specific processing
            await self._analyze_event_batch(batch)
            
            # Update processing latency
            processing_time = (time.time() - start_time) * 1000
            self.analytics_stats['processing_latency_ms'] = processing_time
            
            logger.debug(f"Processed batch of {len(batch)} events in {processing_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
    
    async def _analyze_event_batch(self, events: List[AnalyticsEvent]):
        """Analyze a batch of events for patterns and correlations."""
        try:
            # Group events by type and user
            events_by_type = defaultdict(list)
            events_by_user = defaultdict(list)
            
            for event in events:
                events_by_type[event.event_type].append(event)
                if event.user_id:
                    events_by_user[event.user_id].append(event)
            
            # Detect patterns within the batch
            await self._detect_batch_patterns(events_by_type, events_by_user)
            
            # Analyze correlations
            await self._analyze_batch_correlations(events)
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {str(e)}")
    
    async def _detect_batch_patterns(
        self,
        events_by_type: Dict[EventType, List[AnalyticsEvent]],
        events_by_user: Dict[str, List[AnalyticsEvent]]
    ):
        """Detect patterns in event batch."""
        try:
            # User behavior patterns
            for user_id, user_events in events_by_user.items():
                if len(user_events) >= 3:  # Minimum events for pattern
                    # Analyze event sequence
                    event_sequence = [e.event_type.value for e in sorted(user_events, key=lambda x: x.timestamp)]
                    sequence_key = " -> ".join(event_sequence)
                    
                    if sequence_key not in self.patterns:
                        self.patterns[sequence_key] = {
                            'pattern': sequence_key,
                            'occurrences': 0,
                            'users': set(),
                            'first_seen': datetime.now(),
                            'last_seen': datetime.now()
                        }
                    
                    self.patterns[sequence_key]['occurrences'] += 1
                    self.patterns[sequence_key]['users'].add(user_id)
                    self.patterns[sequence_key]['last_seen'] = datetime.now()
                    
                    # Check if pattern is significant
                    total_users = len(events_by_user)
                    pattern_support = len(self.patterns[sequence_key]['users']) / total_users
                    
                    if pattern_support >= self.config['pattern_min_support']:
                        logger.info(f"Significant pattern detected: {sequence_key} (support: {pattern_support:.2f})")
                        self.analytics_stats['patterns_discovered'] += 1
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {str(e)}")
    
    async def _analyze_batch_correlations(self, events: List[AnalyticsEvent]):
        """Analyze correlations between different metrics in the batch."""
        try:
            # Extract numeric metrics from events
            metrics_data = defaultdict(list)
            
            for event in events:
                for key, value in event.data.items():
                    if isinstance(value, (int, float)):
                        metric_key = f"{event.event_type.value}_{key}"
                        metrics_data[metric_key].append(value)
            
            # Calculate correlations between metrics
            metric_keys = list(metrics_data.keys())
            
            for i, metric1 in enumerate(metric_keys):
                for metric2 in metric_keys[i+1:]:
                    if len(metrics_data[metric1]) >= 10 and len(metrics_data[metric2]) >= 10:
                        # Ensure same length
                        min_length = min(len(metrics_data[metric1]), len(metrics_data[metric2]))
                        values1 = metrics_data[metric1][:min_length]
                        values2 = metrics_data[metric2][:min_length]
                        
                        # Calculate correlation
                        correlation, p_value = stats.pearsonr(values1, values2)
                        
                        if abs(correlation) >= self.config['correlation_threshold'] and p_value < 0.05:
                            logger.info(f"Strong correlation detected: {metric1} <-> {metric2} (r={correlation:.3f})")
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {str(e)}")
    
    async def _insight_generation_loop(self):
        """Background loop for generating insights."""
        try:
            while True:
                await asyncio.sleep(self.insight_generation_interval)
                await self._generate_insights()
                
        except asyncio.CancelledError:
            logger.info("Insight generation loop cancelled")
        except Exception as e:
            logger.error(f"Insight generation loop error: {str(e)}")
    
    async def _generate_insights(self):
        """Generate analytics insights from processed data."""
        try:
            insights = []
            
            # Generate trend insights
            trend_insights = await self._generate_trend_insights()
            insights.extend(trend_insights)
            
            # Generate anomaly insights
            anomaly_insights = await self._generate_anomaly_insights()
            insights.extend(anomaly_insights)
            
            # Generate pattern insights
            pattern_insights = await self._generate_pattern_insights()
            insights.extend(pattern_insights)
            
            # Run custom insight generators
            for generator in self.insight_generators:
                try:
                    if asyncio.iscoroutinefunction(generator):
                        custom_insights = await generator(self)
                    else:
                        custom_insights = generator(self)
                    
                    if custom_insights:
                        insights.extend(custom_insights)
                        
                except Exception as gen_error:
                    logger.error(f"Custom insight generator failed: {str(gen_error)}")
            
            # Filter and store insights
            for insight in insights:
                if insight.confidence >= self.config['insight_confidence_threshold']:
                    self.insights.append(insight)
                    self.analytics_stats['insights_generated'] += 1
            
            # Keep only recent insights
            cutoff_time = datetime.now() - timedelta(days=self.retention_days)
            self.insights = [i for i in self.insights if i.timestamp >= cutoff_time]
            
            if insights:
                logger.info(f"Generated {len(insights)} new insights")
            
        except Exception as e:
            logger.error(f"Insight generation failed: {str(e)}")
    
    async def _generate_trend_insights(self) -> List[Insight]:
        """Generate trend-based insights."""
        try:
            insights = []
            
            # Analyze trends in real-time metrics
            for metric_key, history in self.metric_history.items():
                if len(history) < 20:  # Need sufficient data
                    continue
                
                # Get recent trend
                recent_data = list(history)[-20:]
                values = [v for t, v in recent_data]
                time_points = list(range(len(values)))
                
                # Calculate trend
                if len(values) >= 2:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, values)
                    
                    # Significant trend
                    if abs(r_value) > 0.7 and p_value < 0.05:
                        trend_direction = "increasing" if slope > 0 else "decreasing"
                        trend_strength = "strong" if abs(r_value) > 0.8 else "moderate"
                        
                        insight = Insight(
                            insight_id=f"trend_{metric_key}_{int(time.time())}",
                            insight_type=InsightType.TREND,
                            title=f"{trend_strength.title()} {trend_direction} trend in {metric_key}",
                            description=f"The metric {metric_key} shows a {trend_strength} {trend_direction} trend with correlation {r_value:.3f}",
                            confidence=abs(r_value),
                            impact_score=abs(slope) * abs(r_value),
                            data={
                                'metric': metric_key,
                                'slope': slope,
                                'correlation': r_value,
                                'p_value': p_value,
                                'trend_direction': trend_direction,
                                'values': values[-5:]  # Last 5 values
                            },
                            recommendations=[
                                f"Monitor {metric_key} closely due to {trend_direction} trend",
                                f"Consider investigating causes of {trend_direction} pattern"
                            ],
                            timestamp=datetime.now(),
                            tags=['trend', metric_key, trend_direction]
                        )
                        
                        insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Trend insight generation failed: {str(e)}")
            return []
    
    async def _generate_anomaly_insights(self) -> List[Insight]:
        """Generate anomaly-based insights."""
        try:
            insights = []
            
            # Analyze recent anomalies
            recent_anomalies = [
                a for a in self.anomalies
                if (datetime.now() - a['timestamp']).total_seconds() < 3600  # Last hour
            ]
            
            if not recent_anomalies:
                return insights
            
            # Group anomalies by metric
            anomalies_by_metric = defaultdict(list)
            for anomaly in recent_anomalies:
                anomalies_by_metric[anomaly['metric']].append(anomaly)
            
            # Generate insights for metrics with multiple anomalies
            for metric, metric_anomalies in anomalies_by_metric.items():
                if len(metric_anomalies) >= 3:  # Multiple anomalies
                    severity_counts = defaultdict(int)
                    for anomaly in metric_anomalies:
                        severity_counts[anomaly['severity']] += 1
                    
                    avg_z_score = np.mean([a['z_score'] for a in metric_anomalies])
                    
                    insight = Insight(
                        insight_id=f"anomaly_{metric}_{int(time.time())}",
                        insight_type=InsightType.ANOMALY,
                        title=f"Multiple anomalies detected in {metric}",
                        description=f"Detected {len(metric_anomalies)} anomalies in {metric} with average z-score of {avg_z_score:.2f}",
                        confidence=min(1.0, avg_z_score / 5.0),
                        impact_score=len(metric_anomalies) * avg_z_score,
                        data={
                            'metric': metric,
                            'anomaly_count': len(metric_anomalies),
                            'average_z_score': avg_z_score,
                            'severity_distribution': dict(severity_counts),
                            'time_range': [
                                min(a['timestamp'] for a in metric_anomalies).isoformat(),
                                max(a['timestamp'] for a in metric_anomalies).isoformat()
                            ]
                        },
                        recommendations=[
                            f"Investigate root cause of anomalies in {metric}",
                            "Check for system issues or data quality problems",
                            "Consider adjusting monitoring thresholds if anomalies are expected"
                        ],
                        timestamp=datetime.now(),
                        tags=['anomaly', metric, 'multiple']
                    )
                    
                    insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Anomaly insight generation failed: {str(e)}")
            return []
    
    async def _generate_pattern_insights(self) -> List[Insight]:
        """Generate pattern-based insights."""
        try:
            insights = []
            
            # Analyze significant patterns
            for pattern_key, pattern_data in self.patterns.items():
                support = len(pattern_data['users']) / max(1, self.real_time_metrics.get('unique_users', 1))
                
                if support >= self.config['pattern_min_support'] and pattern_data['occurrences'] >= 10:
                    insight = Insight(
                        insight_id=f"pattern_{pattern_key.replace(' -> ', '_')}_{int(time.time())}",
                        insight_type=InsightType.PATTERN,
                        title=f"Common user behavior pattern: {pattern_key}",
                        description=f"Pattern '{pattern_key}' occurs in {support:.1%} of users with {pattern_data['occurrences']} total occurrences",
                        confidence=min(1.0, support * 2),
                        impact_score=support * pattern_data['occurrences'],
                        data={
                            'pattern': pattern_key,
                            'support': support,
                            'occurrences': pattern_data['occurrences'],
                            'unique_users': len(pattern_data['users']),
                            'first_seen': pattern_data['first_seen'].isoformat(),
                            'last_seen': pattern_data['last_seen'].isoformat()
                        },
                        recommendations=[
                            f"Optimize user experience for common pattern: {pattern_key}",
                            "Consider creating targeted features for this user journey",
                            "Use this pattern for user segmentation and personalization"
                        ],
                        timestamp=datetime.now(),
                        tags=['pattern', 'user_behavior', 'common']
                    )
                    
                    insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Pattern insight generation failed: {str(e)}")
            return []
    
    def register_event_processor(self, event_type: EventType, processor: Callable) -> bool:
        """Register an event processor for specific event type."""
        try:
            self.processors[event_type].append(processor)
            logger.info(f"Registered processor for {event_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register event processor: {str(e)}")
            return False
    
    def register_insight_generator(self, generator: Callable) -> bool:
        """Register a custom insight generator."""
        try:
            self.insight_generators.append(generator)
            logger.info("Registered custom insight generator")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register insight generator: {str(e)}")
            return False
    
    async def query_events(self, query: AnalyticsQuery) -> Dict[str, Any]:
        """Execute analytics query on event data."""
        try:
            start_time = time.time()
            
            # Find relevant events
            relevant_events = []
            start_date = query.time_range[0].date()
            end_date = query.time_range[1].date()
            
            # Iterate through date range
            current_date = start_date
            while current_date <= end_date:
                for event_type_str in [et.value for et in EventType]:
                    event_key = f"{event_type_str}_{current_date}"
                    if event_key in self.processed_events:
                        events = self.processed_events[event_key]
                        
                        # Filter by time range
                        filtered_events = [
                            e for e in events
                            if query.time_range[0] <= e.timestamp <= query.time_range[1]
                        ]
                        
                        # Apply filters
                        for filter_key, filter_value in query.filters.items():
                            if filter_key == 'event_type':
                                filtered_events = [e for e in filtered_events if e.event_type.value == filter_value]
                            elif filter_key == 'user_id':
                                filtered_events = [e for e in filtered_events if e.user_id == filter_value]
                            elif filter_key == 'tag':
                                filtered_events = [e for e in filtered_events if filter_value in e.tags]
                        
                        relevant_events.extend(filtered_events)
                
                current_date += timedelta(days=1)
            
            # Apply aggregation
            result = await self._apply_query_aggregation(relevant_events, query)
            
            # Update query stats
            query_time = (time.time() - start_time) * 1000
            self.analytics_stats['queries_executed'] += 1
            
            result['query_metadata'] = {
                'execution_time_ms': query_time,
                'events_scanned': len(relevant_events),
                'query_id': query.query_id
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Analytics query failed: {str(e)}")
            return {'error': str(e)}
    
    async def _apply_query_aggregation(
        self,
        events: List[AnalyticsEvent],
        query: AnalyticsQuery
    ) -> Dict[str, Any]:
        """Apply aggregation to query results."""
        try:
            result = {
                'dimensions': query.dimensions,
                'metrics': query.metrics,
                'data': [],
                'totals': {}
            }
            
            if not events:
                return result
            
            # Group by dimensions
            if query.grouping:
                groups = defaultdict(list)
                
                for event in events:
                    group_key = []
                    for dimension in query.grouping:
                        if dimension == 'event_type':
                            group_key.append(event.event_type.value)
                        elif dimension == 'user_id':
                            group_key.append(event.user_id or 'unknown')
                        elif dimension == 'date':
                            group_key.append(event.timestamp.date().isoformat())
                        elif dimension == 'hour':
                            group_key.append(event.timestamp.hour)
                        else:
                            group_key.append(event.data.get(dimension, 'unknown'))
                    
                    groups[tuple(group_key)].append(event)
                
                # Aggregate each group
                for group_key, group_events in groups.items():
                    group_result = {'group': dict(zip(query.grouping, group_key))}
                    
                    # Calculate metrics for group
                    for metric in query.metrics:
                        group_result[metric] = await self._calculate_metric(group_events, metric, query.aggregation)
                    
                    result['data'].append(group_result)
            
            else:
                # No grouping - aggregate all events
                aggregated_result = {}
                for metric in query.metrics:
                    aggregated_result[metric] = await self._calculate_metric(events, metric, query.aggregation)
                
                result['data'] = [aggregated_result]
            
            # Calculate totals
            for metric in query.metrics:
                result['totals'][metric] = await self._calculate_metric(events, metric, 'sum')
            
            return result
            
        except Exception as e:
            logger.error(f"Query aggregation failed: {str(e)}")
            return {'error': str(e)}
    
    async def _calculate_metric(
        self,
        events: List[AnalyticsEvent],
        metric: str,
        aggregation: str
    ) -> Union[float, int]:
        """Calculate metric value for events."""
        try:
            if metric == 'count':
                return len(events)
            
            elif metric == 'unique_users':
                return len(set(e.user_id for e in events if e.user_id))
            
            elif metric == 'unique_sessions':
                return len(set(e.session_id for e in events if e.session_id))
            
            else:
                # Extract numeric values from event data
                values = []
                for event in events:
                    if metric in event.data and isinstance(event.data[metric], (int, float)):
                        values.append(event.data[metric])
                
                if not values:
                    return 0
                
                if aggregation == 'sum':
                    return sum(values)
                elif aggregation == 'avg':
                    return sum(values) / len(values)
                elif aggregation == 'min':
                    return min(values)
                elif aggregation == 'max':
                    return max(values)
                elif aggregation == 'count':
                    return len(values)
                else:
                    return sum(values)  # Default to sum
            
        except Exception as e:
            logger.error(f"Metric calculation failed: {str(e)}")
            return 0
    
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        try:
            return {
                'performance_stats': self.analytics_stats.copy(),
                'real_time_metrics': dict(self.real_time_metrics),
                'recent_insights': [
                    insight.to_dict() for insight in self.insights[-10:]
                ],
                'recent_anomalies': self.anomalies[-10:],
                'top_patterns': sorted(
                    [
                        {
                            'pattern': pattern,
                            'support': len(data['users']) / max(1, self.real_time_metrics.get('unique_users', 1)),
                            'occurrences': data['occurrences']
                        }
                        for pattern, data in self.patterns.items()
                    ],
                    key=lambda x: x['support'],
                    reverse=True
                )[:10],
                'event_queue_size': len(self.event_queue),
                'processed_events_count': sum(len(events) for events in self.processed_events.values()),
                'configuration': self.config.copy()
            }
            
        except Exception as e:
            logger.error(f"Failed to get analytics summary: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Clean up analytics engine resources."""
        try:
            # Cancel processing tasks
            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
            
            if self.insight_task:
                self.insight_task.cancel()
                try:
                    await self.insight_task
                except asyncio.CancelledError:
                    pass
            
            # Clear data structures
            self.event_queue.clear()
            self.processed_events.clear()
            self.event_indexes.clear()
            self.real_time_metrics.clear()
            self.metric_history.clear()
            self.insights.clear()
            self.patterns.clear()
            self.anomalies.clear()
            self.processors.clear()
            self.insight_generators.clear()
            
            logger.info("AnalyticsEngine cleanup completed")
            
        except Exception as e:
            logger.error(f"AnalyticsEngine cleanup failed: {str(e)}")