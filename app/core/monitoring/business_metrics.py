"""
Business metrics collection and analysis system.
Tracks key business KPIs and user satisfaction metrics.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading

from app.utils.logger import logger
from .metrics_collector import get_metrics_collector
from .correlation_tracker import get_current_correlation_id


class MatchType(str, Enum):
    """Types of matching operations."""
    RESUME_TO_JOBS = "resume_to_jobs"
    JOB_TO_RESUMES = "job_to_resumes"
    SIMILARITY_SEARCH = "similarity_search"
    RECOMMENDATION = "recommendation"


class UserType(str, Enum):
    """Types of users."""
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    TRIAL = "trial"


@dataclass
class MatchEvent:
    """Business event for match operations."""
    user_id: str
    match_type: MatchType
    user_type: UserType
    job_count: int
    processing_time: float
    match_score: float
    user_satisfaction: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserEngagementEvent:
    """User engagement tracking event."""
    user_id: str
    event_type: str  # view, click, apply, bookmark, etc.
    target_id: str  # job_id or match_id
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Quality metrics for matches."""
    relevance_score: float
    diversity_score: float
    freshness_score: float
    completeness_score: float
    user_feedback_score: Optional[float] = None


class BusinessMetricsCollector:
    """
    Collects and analyzes business metrics for the resume matching system.
    """
    
    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        self._lock = threading.RLock()
        
        # Event storage
        self._match_events: deque = deque(maxlen=100000)
        self._engagement_events: deque = deque(maxlen=100000)
        
        # Aggregated metrics
        self._hourly_stats: Dict[str, Dict] = defaultdict(dict)
        self._daily_stats: Dict[str, Dict] = defaultdict(dict)
        
        # Real-time metrics
        self._active_users: set = set()
        self._session_data: Dict[str, Dict] = {}
        
        # Get metrics collector
        self.metrics_collector = get_metrics_collector()
        
        logger.info("Business metrics collector initialized")
    
    def record_match_event(
        self,
        user_id: str,
        match_type: MatchType,
        user_type: UserType,
        job_count: int,
        processing_time: float,
        match_score: float,
        user_satisfaction: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a match operation event."""
        
        event = MatchEvent(
            user_id=user_id,
            match_type=match_type,
            user_type=user_type,
            job_count=job_count,
            processing_time=processing_time,
            match_score=match_score,
            user_satisfaction=user_satisfaction,
            correlation_id=get_current_correlation_id(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self._match_events.append(event)
            self._active_users.add(user_id)
        
        # Update Prometheus metrics
        self.metrics_collector.increment_counter(
            "match_requests",
            labels={
                "user_type": user_type.value,
                "match_type": match_type.value
            }
        )
        
        self.metrics_collector.observe_histogram(
            "match_duration",
            processing_time,
            labels={
                "match_type": match_type.value,
                "status": "completed"
            }
        )
        
        self.metrics_collector.observe_histogram(
            "match_quality",
            match_score,
            labels={"match_type": match_type.value}
        )
        
        # Log business event
        logger.info(
            "Match event recorded",
            extra={
                "user_id": user_id,
                "match_type": match_type.value,
                "processing_time": processing_time,
                "match_score": match_score,
                "job_count": job_count,
                "correlation_id": event.correlation_id
            }
        )
    
    def record_user_engagement(
        self,
        user_id: str,
        event_type: str,
        target_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record user engagement event."""
        
        event = UserEngagementEvent(
            user_id=user_id,
            event_type=event_type,
            target_id=target_id,
            correlation_id=get_current_correlation_id(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self._engagement_events.append(event)
            self._active_users.add(user_id)
        
        # Update metrics
        self.metrics_collector.increment_counter(
            "user_engagement_events",
            labels={
                "event_type": event_type,
                "target_type": "job" if event_type in ["apply", "bookmark"] else "match"
            }
        )
        
        logger.debug(
            "User engagement recorded",
            extra={
                "user_id": user_id,
                "event_type": event_type,
                "target_id": target_id,
                "correlation_id": event.correlation_id
            }
        )
    
    def calculate_user_satisfaction(self, user_id: str, time_window_hours: int = 24) -> Optional[float]:
        """Calculate user satisfaction score based on recent activity."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        with self._lock:
            # Get recent events for user
            recent_matches = [
                event for event in self._match_events
                if event.user_id == user_id and event.timestamp >= cutoff_time
            ]
            
            recent_engagements = [
                event for event in self._engagement_events
                if event.user_id == user_id and event.timestamp >= cutoff_time
            ]
        
        if not recent_matches:
            return None
        
        # Calculate satisfaction based on:
        # 1. Match quality scores
        # 2. User engagement rate
        # 3. Explicit feedback
        
        avg_match_score = sum(event.match_score for event in recent_matches) / len(recent_matches)
        
        engagement_rate = len(recent_engagements) / len(recent_matches) if recent_matches else 0
        engagement_score = min(engagement_rate / 3.0, 1.0)  # Normalize to 0-1
        
        # Explicit feedback if available
        explicit_feedback = [
            event.user_satisfaction for event in recent_matches
            if event.user_satisfaction is not None
        ]
        
        feedback_score = sum(explicit_feedback) / len(explicit_feedback) if explicit_feedback else 0.7
        
        # Weighted satisfaction score
        satisfaction = (
            avg_match_score * 0.4 +
            engagement_score * 0.3 +
            feedback_score * 0.3
        )
        
        return min(max(satisfaction, 0.0), 1.0)
    
    def get_conversion_rate(self, time_window_hours: int = 24) -> float:
        """Calculate conversion rate (applications per match)."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        with self._lock:
            recent_matches = len([
                event for event in self._match_events
                if event.timestamp >= cutoff_time
            ])
            
            recent_applications = len([
                event for event in self._engagement_events
                if event.event_type == "apply" and event.timestamp >= cutoff_time
            ])
        
        return recent_applications / recent_matches if recent_matches > 0 else 0.0
    
    def get_active_user_count(self, time_window_minutes: int = 60) -> int:
        """Get number of active users in the specified time window."""
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        active_users = set()
        
        with self._lock:
            # Count users from match events
            for event in self._match_events:
                if event.timestamp >= cutoff_time:
                    active_users.add(event.user_id)
            
            # Count users from engagement events
            for event in self._engagement_events:
                if event.timestamp >= cutoff_time:
                    active_users.add(event.user_id)
        
        return len(active_users)
    
    def get_performance_metrics(self, time_window_hours: int = 24) -> Dict[str, float]:
        """Get performance metrics for the specified time window."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        with self._lock:
            recent_matches = [
                event for event in self._match_events
                if event.timestamp >= cutoff_time
            ]
        
        if not recent_matches:
            return {}
        
        # Calculate metrics
        total_matches = len(recent_matches)
        avg_processing_time = sum(event.processing_time for event in recent_matches) / total_matches
        avg_match_score = sum(event.match_score for event in recent_matches) / total_matches
        avg_job_count = sum(event.job_count for event in recent_matches) / total_matches
        
        # User type breakdown
        user_type_counts = defaultdict(int)
        for event in recent_matches:
            user_type_counts[event.user_type.value] += 1
        
        # Match type breakdown
        match_type_counts = defaultdict(int)
        for event in recent_matches:
            match_type_counts[event.match_type.value] += 1
        
        return {
            "total_matches": total_matches,
            "avg_processing_time": avg_processing_time,
            "avg_match_score": avg_match_score,
            "avg_job_count": avg_job_count,
            "conversion_rate": self.get_conversion_rate(time_window_hours),
            "active_users": self.get_active_user_count(time_window_hours * 60),
            "user_type_distribution": dict(user_type_counts),
            "match_type_distribution": dict(match_type_counts),
        }
    
    def get_quality_metrics(self, time_window_hours: int = 24) -> QualityMetrics:
        """Calculate quality metrics for matches."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        with self._lock:
            recent_matches = [
                event for event in self._match_events
                if event.timestamp >= cutoff_time
            ]
        
        if not recent_matches:
            return QualityMetrics(
                relevance_score=0.0,
                diversity_score=0.0,
                freshness_score=0.0,
                completeness_score=0.0
            )
        
        # Relevance: Average match scores
        relevance_score = sum(event.match_score for event in recent_matches) / len(recent_matches)
        
        # Diversity: Variety in job counts and types
        job_counts = [event.job_count for event in recent_matches]
        diversity_score = min(len(set(job_counts)) / 10.0, 1.0)  # Normalized
        
        # Freshness: How recent are the matches
        avg_age_hours = sum(
            (datetime.utcnow() - event.timestamp).total_seconds() / 3600
            for event in recent_matches
        ) / len(recent_matches)
        freshness_score = max(0.0, 1.0 - (avg_age_hours / 24.0))  # 24h decay
        
        # Completeness: Average processing time (inverse correlation)
        avg_processing_time = sum(event.processing_time for event in recent_matches) / len(recent_matches)
        completeness_score = max(0.0, 1.0 - (avg_processing_time / 300.0))  # 5min max
        
        # User feedback if available
        feedback_scores = [
            event.user_satisfaction for event in recent_matches
            if event.user_satisfaction is not None
        ]
        user_feedback_score = sum(feedback_scores) / len(feedback_scores) if feedback_scores else None
        
        return QualityMetrics(
            relevance_score=relevance_score,
            diversity_score=diversity_score,
            freshness_score=freshness_score,
            completeness_score=completeness_score,
            user_feedback_score=user_feedback_score
        )
    
    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get analytics for a specific user."""
        
        with self._lock:
            user_matches = [
                event for event in self._match_events
                if event.user_id == user_id
            ]
            
            user_engagements = [
                event for event in self._engagement_events
                if event.user_id == user_id
            ]
        
        if not user_matches:
            return {"error": "No data found for user"}
        
        # Calculate user-specific metrics
        total_matches = len(user_matches)
        avg_match_score = sum(event.match_score for event in user_matches) / total_matches
        avg_processing_time = sum(event.processing_time for event in user_matches) / total_matches
        
        # Engagement analysis
        engagement_types = defaultdict(int)
        for event in user_engagements:
            engagement_types[event.event_type] += 1
        
        # Recent activity
        recent_activity = len([
            event for event in user_matches
            if (datetime.utcnow() - event.timestamp).days <= 7
        ])
        
        return {
            "user_id": user_id,
            "total_matches": total_matches,
            "avg_match_score": avg_match_score,
            "avg_processing_time": avg_processing_time,
            "engagement_breakdown": dict(engagement_types),
            "recent_activity_7days": recent_activity,
            "user_satisfaction": self.calculate_user_satisfaction(user_id),
            "last_activity": max(event.timestamp for event in user_matches).isoformat(),
        }
    
    def cleanup_old_events(self):
        """Clean up events older than retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        
        with self._lock:
            # Filter events
            self._match_events = deque([
                event for event in self._match_events
                if event.timestamp >= cutoff_date
            ], maxlen=self._match_events.maxlen)
            
            self._engagement_events = deque([
                event for event in self._engagement_events
                if event.timestamp >= cutoff_date
            ], maxlen=self._engagement_events.maxlen)
        
        logger.info(f"Cleaned up events older than {self.retention_days} days")
    
    def export_metrics(self, format: str = "json") -> Dict[str, Any]:
        """Export metrics in specified format."""
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "performance_metrics": self.get_performance_metrics(),
            "quality_metrics": self.get_quality_metrics().__dict__,
            "active_users_1h": self.get_active_user_count(60),
            "active_users_24h": self.get_active_user_count(1440),
            "conversion_rate_24h": self.get_conversion_rate(24),
            "total_events": {
                "matches": len(self._match_events),
                "engagements": len(self._engagement_events),
            }
        }
        
        return metrics


# Global business metrics collector
_global_business_collector: Optional[BusinessMetricsCollector] = None


def get_business_metrics_collector() -> BusinessMetricsCollector:
    """Get the global business metrics collector."""
    global _global_business_collector
    if _global_business_collector is None:
        _global_business_collector = BusinessMetricsCollector()
    return _global_business_collector


def initialize_business_metrics(retention_days: int = 30) -> BusinessMetricsCollector:
    """Initialize the global business metrics collector."""
    global _global_business_collector
    _global_business_collector = BusinessMetricsCollector(retention_days=retention_days)
    return _global_business_collector