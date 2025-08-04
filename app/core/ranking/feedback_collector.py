"""
Feedback Collector for user interaction tracking.

This module collects, validates, and processes user feedback for continuous
model improvement and learning-to-rank optimization.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid

from app.utils.logger import logger
from app.core.constants.ml_constants import (
    FeedbackConfig,
    EnsembleConfig,
    PerformanceMetrics
)
from app.core.cache.embedding_cache import EmbeddingCache
from app.db.session import get_db


class FeedbackType(str, Enum):
    """Types of user feedback events."""
    IMPRESSION = "impression"  # Job was shown to user
    CLICK = "click"  # User clicked on job
    SAVE = "save"  # User saved job
    APPLY = "apply"  # User applied to job
    DISMISS = "dismiss"  # User dismissed job
    INTERVIEW = "interview"  # User got interview
    OFFER = "offer"  # User received offer
    HIRED = "hired"  # User was hired


@dataclass
class FeedbackEvent:
    """Container for a single feedback event."""
    event_id: str
    user_id: str
    job_id: str
    resume_id: Optional[str]
    feedback_type: FeedbackType
    timestamp: datetime
    match_score: Optional[float]
    rank_position: Optional[int]
    session_id: Optional[str]
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'event_id': self.event_id,
            'user_id': self.user_id,
            'job_id': self.job_id,
            'resume_id': self.resume_id,
            'feedback_type': self.feedback_type.value,
            'timestamp': self.timestamp.isoformat(),
            'match_score': self.match_score,
            'rank_position': self.rank_position,
            'session_id': self.session_id,
            'context': self.context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackEvent':
        """Create from dictionary."""
        return cls(
            event_id=data['event_id'],
            user_id=data['user_id'],
            job_id=data['job_id'],
            resume_id=data.get('resume_id'),
            feedback_type=FeedbackType(data['feedback_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            match_score=data.get('match_score'),
            rank_position=data.get('rank_position'),
            session_id=data.get('session_id'),
            context=data.get('context', {})
        )


@dataclass
class FeedbackAggregation:
    """Aggregated feedback statistics."""
    total_events: int
    events_by_type: Dict[str, int]
    conversion_rates: Dict[str, float]
    average_match_scores: Dict[str, float]
    user_engagement_metrics: Dict[str, Any]
    temporal_patterns: Dict[str, Any]


class FeedbackCollector:
    """
    Advanced feedback collection system for tracking user interactions,
    validating feedback quality, and preparing data for model training.
    """
    
    def __init__(self):
        """Initialize the feedback collector."""
        self.feedback_weights = FeedbackConfig.FEEDBACK_WEIGHTS
        self.validation_rules = FeedbackConfig.VALIDATION_RULES
        self.learning_params = FeedbackConfig.LEARNING_PARAMS
        
        # Caching for feedback data
        self.embedding_cache = EmbeddingCache()
        
        # In-memory buffers for batch processing
        self.feedback_buffer: List[FeedbackEvent] = []
        self.buffer_size_limit = 1000
        
        # Validation state
        self.user_session_data: Dict[str, Dict[str, Any]] = {}
        self.spam_detection_cache: Dict[str, List[datetime]] = {}
        
        # Performance tracking
        self.collection_stats = {
            'total_collected': 0,
            'total_validated': 0,
            'total_rejected': 0,
            'events_by_type': {},
            'validation_errors': []
        }
        
        logger.info("FeedbackCollector initialized successfully")
    
    async def initialize(self):
        """Initialize async components."""
        try:
            await self.embedding_cache.initialize()
            logger.info("FeedbackCollector async initialization completed")
        except Exception as e:
            logger.error(f"FeedbackCollector initialization failed: {str(e)}")
    
    async def collect_feedback(
        self,
        user_id: str,
        job_id: str,
        feedback_type: str,
        context: Optional[Dict[str, Any]] = None,
        resume_id: Optional[str] = None,
        match_score: Optional[float] = None,
        rank_position: Optional[int] = None,
        session_id: Optional[str] = None
    ) -> bool:
        """
        Collect a single feedback event with validation.
        
        Args:
            user_id: Unique identifier for the user
            job_id: Unique identifier for the job
            feedback_type: Type of feedback event
            context: Additional context information
            resume_id: Associated resume identifier
            match_score: Original matching score
            rank_position: Position in ranking list
            session_id: User session identifier
            
        Returns:
            True if feedback was collected successfully, False otherwise
        """
        try:
            # Validate feedback type
            try:
                feedback_type_enum = FeedbackType(feedback_type)
            except ValueError:
                logger.warning(f"Invalid feedback type: {feedback_type}")
                return False
            
            # Create feedback event
            event = FeedbackEvent(
                event_id=str(uuid.uuid4()),
                user_id=user_id,
                job_id=job_id,
                resume_id=resume_id,
                feedback_type=feedback_type_enum,
                timestamp=datetime.now(),
                match_score=match_score,
                rank_position=rank_position,
                session_id=session_id or self._generate_session_id(user_id),
                context=context or {}
            )
            
            # Validate the feedback event
            if not await self._validate_feedback(event):
                self.collection_stats['total_rejected'] += 1
                return False
            
            # Store the feedback
            success = await self._store_feedback(event)
            
            if success:
                # Add to buffer for batch processing
                self.feedback_buffer.append(event)
                
                # Update statistics
                self.collection_stats['total_collected'] += 1
                self.collection_stats['total_validated'] += 1
                
                feedback_type_str = feedback_type_enum.value
                self.collection_stats['events_by_type'][feedback_type_str] = \
                    self.collection_stats['events_by_type'].get(feedback_type_str, 0) + 1
                
                # Process buffer if it's getting full
                if len(self.feedback_buffer) >= self.buffer_size_limit:
                    await self._process_feedback_buffer()
                
                logger.debug(f"Feedback collected: {feedback_type} for user {user_id}, job {job_id}")
                return True
            else:
                self.collection_stats['total_rejected'] += 1
                return False
                
        except Exception as e:
            logger.error(f"Feedback collection failed: {str(e)}")
            self.collection_stats['total_rejected'] += 1
            return False
    
    async def _validate_feedback(self, event: FeedbackEvent) -> bool:
        """
        Validate feedback event against quality rules.
        
        Args:
            event: Feedback event to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic field validation
            if not event.user_id or not event.job_id:
                self._log_validation_error("Missing required fields", event)
                return False
            
            # Timestamp validation
            if event.timestamp > datetime.now() + timedelta(minutes=5):
                self._log_validation_error("Future timestamp", event)
                return False
            
            # Session duration validation
            if not await self._validate_session_duration(event):
                self._log_validation_error("Invalid session duration", event)
                return False
            
            # Spam detection
            if not await self._validate_against_spam(event):
                self._log_validation_error("Spam detection triggered", event)
                return False
            
            # Rate limiting per user
            if not await self._validate_rate_limiting(event):
                self._log_validation_error("Rate limit exceeded", event)
                return False
            
            # Score validation
            if event.match_score is not None:
                if not (0.0 <= event.match_score <= 1.0):
                    self._log_validation_error("Invalid match score", event)
                    return False
            
            # Rank position validation
            if event.rank_position is not None:
                if event.rank_position < 1 or event.rank_position > 100:
                    self._log_validation_error("Invalid rank position", event)
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Feedback validation failed: {str(e)}")
            return False
    
    async def _validate_session_duration(self, event: FeedbackEvent) -> bool:
        """Validate session duration for meaningful feedback."""
        try:
            session_id = event.session_id
            if not session_id:
                return True  # No session tracking, assume valid
            
            # Get or create session data
            if session_id not in self.user_session_data:
                self.user_session_data[session_id] = {
                    'start_time': event.timestamp,
                    'events': [],
                    'user_id': event.user_id
                }
            
            session_data = self.user_session_data[session_id]
            session_duration = (event.timestamp - session_data['start_time']).total_seconds()
            
            # Check minimum session duration for meaningful feedback
            min_duration = self.validation_rules['min_session_duration']
            
            # Allow impression events without duration check
            if event.feedback_type == FeedbackType.IMPRESSION:
                return True
            
            # Other events need minimum session duration
            return session_duration >= min_duration
            
        except Exception as e:
            logger.error(f"Session duration validation failed: {str(e)}")
            return True  # Default to valid on error
    
    async def _validate_against_spam(self, event: FeedbackEvent) -> bool:
        """Validate against spam patterns."""
        try:
            user_id = event.user_id
            current_time = event.timestamp
            
            # Get recent feedback from this user
            if user_id not in self.spam_detection_cache:
                self.spam_detection_cache[user_id] = []
            
            user_events = self.spam_detection_cache[user_id]
            
            # Remove events older than 1 hour
            cutoff_time = current_time - timedelta(hours=1)
            user_events = [t for t in user_events if t > cutoff_time]
            self.spam_detection_cache[user_id] = user_events
            
            # Check rate limit
            max_events_per_hour = 100  # Configurable limit
            if len(user_events) >= max_events_per_hour:
                return False
            
            # Add current event
            user_events.append(current_time)
            
            # Check for rapid-fire patterns (more than 10 events in 1 minute)
            recent_cutoff = current_time - timedelta(minutes=1)
            recent_events = [t for t in user_events if t > recent_cutoff]
            
            if len(recent_events) > 10:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Spam validation failed: {str(e)}")
            return True  # Default to valid on error
    
    async def _validate_rate_limiting(self, event: FeedbackEvent) -> bool:
        """Validate against rate limiting rules."""
        try:
            max_per_day = self.validation_rules['max_feedback_per_user_per_day']
            
            # This would typically query the database for today's feedback count
            # For now, we'll use a simplified check
            return True  # Placeholder implementation
            
        except Exception as e:
            logger.error(f"Rate limiting validation failed: {str(e)}")
            return True
    
    def _log_validation_error(self, error_type: str, event: FeedbackEvent):
        """Log validation error for monitoring."""
        error_info = {
            'error_type': error_type,
            'user_id': event.user_id,
            'job_id': event.job_id,
            'feedback_type': event.feedback_type.value,
            'timestamp': event.timestamp.isoformat()
        }
        
        self.collection_stats['validation_errors'].append(error_info)
        
        # Keep only last 100 errors
        if len(self.collection_stats['validation_errors']) > 100:
            self.collection_stats['validation_errors'] = \
                self.collection_stats['validation_errors'][-100:]
    
    async def _store_feedback(self, event: FeedbackEvent) -> bool:
        """Store feedback event in database and cache."""
        try:
            # Store in database
            db_success = await self._store_in_database(event)
            
            # Store in cache for quick access
            cache_success = await self._store_in_cache(event)
            
            return db_success and cache_success
            
        except Exception as e:
            logger.error(f"Feedback storage failed: {str(e)}")
            return False
    
    async def _store_in_database(self, event: FeedbackEvent) -> bool:
        """Store feedback in database."""
        try:
            # This would use the actual database session
            # For now, we'll simulate success
            logger.debug(f"Storing feedback in database: {event.event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Database storage failed: {str(e)}")
            return False
    
    async def _store_in_cache(self, event: FeedbackEvent) -> bool:
        """Store feedback in cache for quick access."""
        try:
            cache_key = f"feedback:{event.user_id}:{event.job_id}:{event.timestamp.isoformat()}"
            
            # Store in embedding cache with appropriate TTL
            feedback_data = event.to_dict()
            
            # This would use the embedding cache
            # For now, we'll simulate success
            logger.debug(f"Storing feedback in cache: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Cache storage failed: {str(e)}")
            return False
    
    async def _process_feedback_buffer(self):
        """Process accumulated feedback for batch operations."""
        try:
            if not self.feedback_buffer:
                return
            
            logger.info(f"Processing feedback buffer with {len(self.feedback_buffer)} events")
            
            # Batch operations could include:
            # - Aggregating statistics
            # - Triggering model retraining
            # - Computing user engagement metrics
            
            # Clear the buffer
            self.feedback_buffer.clear()
            
            logger.debug("Feedback buffer processed successfully")
            
        except Exception as e:
            logger.error(f"Feedback buffer processing failed: {str(e)}")
    
    def _generate_session_id(self, user_id: str) -> str:
        """Generate a session ID for user tracking."""
        timestamp = datetime.now().isoformat()
        content = f"{user_id}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def get_user_feedback(
        self,
        user_id: str,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        feedback_types: Optional[List[str]] = None
    ) -> List[FeedbackEvent]:
        """
        Retrieve feedback events for a specific user.
        
        Args:
            user_id: User identifier
            time_range: Optional time range filter
            feedback_types: Optional feedback type filter
            
        Returns:
            List of feedback events
        """
        try:
            # This would query the database
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Failed to retrieve user feedback: {str(e)}")
            return []
    
    async def get_job_feedback(
        self,
        job_id: str,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[FeedbackEvent]:
        """
        Retrieve feedback events for a specific job.
        
        Args:
            job_id: Job identifier
            time_range: Optional time range filter
            
        Returns:
            List of feedback events
        """
        try:
            # This would query the database
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Failed to retrieve job feedback: {str(e)}")
            return []
    
    async def aggregate_feedback(
        self,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        user_id: Optional[str] = None
    ) -> FeedbackAggregation:
        """
        Aggregate feedback statistics.
        
        Args:
            time_range: Optional time range filter
            user_id: Optional user filter
            
        Returns:
            Aggregated feedback statistics
        """
        try:
            # This would compute real aggregations from database
            # For now, return placeholder data
            
            events_by_type = {}
            for feedback_type in FeedbackType:
                events_by_type[feedback_type.value] = 0
            
            conversion_rates = {}
            for feedback_type in FeedbackType:
                conversion_rates[feedback_type.value] = 0.0
            
            average_match_scores = {}
            for feedback_type in FeedbackType:
                average_match_scores[feedback_type.value] = 0.0
            
            return FeedbackAggregation(
                total_events=0,
                events_by_type=events_by_type,
                conversion_rates=conversion_rates,
                average_match_scores=average_match_scores,
                user_engagement_metrics={},
                temporal_patterns={}
            )
            
        except Exception as e:
            logger.error(f"Feedback aggregation failed: {str(e)}")
            return FeedbackAggregation(0, {}, {}, {}, {}, {})
    
    async def get_training_data(
        self,
        min_feedback_count: int = 10,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get training data for model updates.
        
        Args:
            min_feedback_count: Minimum feedback events required
            time_range: Optional time range filter
            
        Returns:
            Training data ready for model training
        """
        try:
            # This would prepare training data from feedback
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Training data preparation failed: {str(e)}")
            return []
    
    async def calculate_feedback_weights(
        self,
        feedback_events: List[FeedbackEvent]
    ) -> Dict[str, float]:
        """
        Calculate weighted feedback scores for events.
        
        Args:
            feedback_events: List of feedback events
            
        Returns:
            Dictionary mapping event IDs to weighted scores
        """
        try:
            weighted_scores = {}
            
            for event in feedback_events:
                # Get base weight for feedback type
                base_weight = self.feedback_weights.get(event.feedback_type.value, 0.0)
                
                # Apply temporal decay
                days_ago = (datetime.now() - event.timestamp).days
                decay_factor = self.learning_params['feedback_decay'] ** days_ago
                
                # Calculate final weight
                final_weight = base_weight * decay_factor
                
                # Consider match score if available
                if event.match_score is not None:
                    # Higher original match scores get slight boost
                    score_boost = 1.0 + (event.match_score - 0.5) * 0.2
                    final_weight *= score_boost
                
                # Consider rank position if available
                if event.rank_position is not None:
                    # Higher ranked positions (lower numbers) get slight boost
                    rank_boost = 1.0 + (10 - min(event.rank_position, 10)) * 0.05
                    final_weight *= rank_boost
                
                weighted_scores[event.event_id] = final_weight
            
            return weighted_scores
            
        except Exception as e:
            logger.error(f"Feedback weight calculation failed: {str(e)}")
            return {}
    
    async def detect_feedback_patterns(
        self,
        user_id: str,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Detect patterns in user feedback behavior.
        
        Args:
            user_id: User identifier
            lookback_days: Days to look back for pattern detection
            
        Returns:
            Dictionary with detected patterns
        """
        try:
            # Get user feedback for the specified period
            end_time = datetime.now()
            start_time = end_time - timedelta(days=lookback_days)
            
            user_feedback = await self.get_user_feedback(
                user_id, (start_time, end_time)
            )
            
            if not user_feedback:
                return {'patterns': [], 'confidence': 0.0}
            
            patterns = []
            
            # Analyze feedback type distribution
            type_counts = {}
            for event in user_feedback:
                type_counts[event.feedback_type.value] = \
                    type_counts.get(event.feedback_type.value, 0) + 1
            
            # Detect high engagement pattern
            high_engagement_types = ['click', 'save', 'apply']
            high_engagement_count = sum(
                type_counts.get(t, 0) for t in high_engagement_types
            )
            
            if high_engagement_count > len(user_feedback) * 0.5:
                patterns.append({
                    'type': 'high_engagement',
                    'confidence': high_engagement_count / len(user_feedback),
                    'description': 'User shows high engagement with job recommendations'
                })
            
            # Detect selective pattern
            apply_rate = type_counts.get('apply', 0) / max(type_counts.get('click', 1), 1)
            if apply_rate > 0.3:
                patterns.append({
                    'type': 'selective_applicant',
                    'confidence': apply_rate,
                    'description': 'User is selective but applies to clicked jobs'
                })
            
            # Calculate overall confidence
            overall_confidence = len(user_feedback) / 100.0  # More feedback = higher confidence
            overall_confidence = min(1.0, overall_confidence)
            
            return {
                'patterns': patterns,
                'confidence': overall_confidence,
                'feedback_count': len(user_feedback),
                'type_distribution': type_counts
            }
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {str(e)}")
            return {'patterns': [], 'confidence': 0.0}
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get feedback collection statistics."""
        try:
            # Calculate derived statistics
            validation_rate = (
                self.collection_stats['total_validated'] / 
                max(self.collection_stats['total_collected'] + self.collection_stats['total_rejected'], 1)
            )
            
            return {
                'collection_stats': self.collection_stats.copy(),
                'validation_rate': validation_rate,
                'buffer_size': len(self.feedback_buffer),
                'active_sessions': len(self.user_session_data),
                'spam_cache_size': len(self.spam_detection_cache),
                'configuration': {
                    'feedback_weights': self.feedback_weights,
                    'validation_rules': self.validation_rules,
                    'learning_params': self.learning_params
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {}
    
    async def cleanup_old_data(self, retention_days: int = 365):
        """Clean up old feedback data beyond retention period."""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Clean up session data
            sessions_to_remove = []
            for session_id, session_data in self.user_session_data.items():
                if session_data['start_time'] < cutoff_date:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.user_session_data[session_id]
            
            # Clean up spam detection cache
            for user_id in list(self.spam_detection_cache.keys()):
                user_events = self.spam_detection_cache[user_id]
                recent_events = [t for t in user_events if t > cutoff_date]
                
                if recent_events:
                    self.spam_detection_cache[user_id] = recent_events
                else:
                    del self.spam_detection_cache[user_id]
            
            logger.info(f"Cleaned up feedback data older than {retention_days} days")
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {str(e)}")
    
    async def cleanup(self):
        """Clean up collector resources."""
        try:
            # Process any remaining feedback in buffer
            if self.feedback_buffer:
                await self._process_feedback_buffer()
            
            # Clean up cache
            if self.embedding_cache:
                await self.embedding_cache.cleanup()
            
            logger.info("FeedbackCollector cleanup completed")
            
        except Exception as e:
            logger.error(f"FeedbackCollector cleanup failed: {str(e)}")