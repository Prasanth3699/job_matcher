"""
Performance Monitor for A/B testing framework.

This module tracks and analyzes performance metrics for A/B testing variants
with real-time monitoring and statistical analysis capabilities.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from scipy import stats

from app.utils.logger import logger
from app.core.constants.ml_constants import PerformanceMetrics

if TYPE_CHECKING:
    from app.core.ab_testing.experiment_manager import ExperimentConfig


@dataclass
class MetricEvent:
    """Individual metric event for tracking."""
    experiment_id: str
    user_id: str
    variant_id: str
    event_type: str
    event_data: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'experiment_id': self.experiment_id,
            'user_id': self.user_id,
            'variant_id': self.variant_id,
            'event_type': self.event_type,
            'event_data': self.event_data,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time."""
    experiment_id: str
    variant_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    sample_size: int
    confidence_interval: Dict[str, Tuple[float, float]]


class PerformanceMonitor:
    """
    Real-time performance monitoring system for A/B testing variants
    with statistical analysis and alerting capabilities.
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.experiment_data: Dict[str, Dict[str, List[MetricEvent]]] = defaultdict(lambda: defaultdict(list))
        self.performance_snapshots: Dict[str, List[PerformanceSnapshot]] = defaultdict(list)
        self.user_assignments: Dict[str, Dict[str, str]] = {}  # user_id -> {experiment_id: variant_id}
        
        # Configuration
        self.snapshot_interval_minutes = 5
        self.retention_days = 30
        self.confidence_level = 0.95
        
        # Performance tracking
        self.monitor_stats = {
            'total_events': 0,
            'events_by_experiment': defaultdict(int),
            'events_by_type': defaultdict(int),
            'snapshots_created': 0,
            'alerts_triggered': 0
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'performance_degradation': 0.15,  # 15% degradation
            'error_rate_increase': 0.1,       # 10% error rate increase
            'latency_increase': 2.0,           # 2x latency increase
            'sample_size_minimum': 100        # Minimum sample size for analysis
        }
        
        logger.info("PerformanceMonitor initialized successfully")
    
    async def initialize_experiment(
        self,
        experiment_id: str,
        config: "ExperimentConfig"
    ) -> bool:
        """
        Initialize monitoring for a new experiment.
        
        Args:
            experiment_id: Experiment identifier
            config: Experiment configuration
            
        Returns:
            True if initialized successfully
        """
        try:
            # Initialize data structures for each variant
            for variant_id in config.variants:
                self.experiment_data[experiment_id][variant_id] = []
            
            # Initialize monitoring stats
            self.monitor_stats['events_by_experiment'][experiment_id] = 0
            
            logger.info(f"Initialized performance monitoring for experiment {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize experiment monitoring: {str(e)}")
            return False
    
    async def track_assignment(
        self,
        experiment_id: str,
        user_id: str,
        variant_id: str,
        timestamp: datetime
    ) -> bool:
        """
        Track user assignment to variant.
        
        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            variant_id: Assigned variant
            timestamp: Assignment timestamp
            
        Returns:
            True if tracked successfully
        """
        try:
            # Store assignment
            if user_id not in self.user_assignments:
                self.user_assignments[user_id] = {}
            self.user_assignments[user_id][experiment_id] = variant_id
            
            # Track as an event
            assignment_event = MetricEvent(
                experiment_id=experiment_id,
                user_id=user_id,
                variant_id=variant_id,
                event_type='assignment',
                event_data={'assigned_at': timestamp.isoformat()},
                timestamp=timestamp
            )
            
            return await self.track_event(
                experiment_id, user_id, 'assignment', 
                assignment_event.event_data, timestamp
            )
            
        except Exception as e:
            logger.error(f"Failed to track assignment: {str(e)}")
            return False
    
    async def track_event(
        self,
        experiment_id: str,
        user_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Track a performance event.
        
        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            event_type: Type of event
            event_data: Event-specific data
            timestamp: Event timestamp
            
        Returns:
            True if tracked successfully
        """
        try:
            timestamp = timestamp or datetime.now()
            
            # Get user's variant assignment
            variant_id = None
            if user_id in self.user_assignments:
                variant_id = self.user_assignments[user_id].get(experiment_id)
            
            if not variant_id:
                logger.warning(f"No variant assignment found for user {user_id} in experiment {experiment_id}")
                return False
            
            # Create metric event
            event = MetricEvent(
                experiment_id=experiment_id,
                user_id=user_id,
                variant_id=variant_id,
                event_type=event_type,
                event_data=event_data,
                timestamp=timestamp
            )
            
            # Store event
            self.experiment_data[experiment_id][variant_id].append(event)
            
            # Update stats
            self.monitor_stats['total_events'] += 1
            self.monitor_stats['events_by_experiment'][experiment_id] += 1
            self.monitor_stats['events_by_type'][event_type] += 1
            
            # Check for alerts
            await self._check_performance_alerts(experiment_id, variant_id)
            
            logger.debug(f"Tracked event {event_type} for user {user_id} in variant {variant_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to track event: {str(e)}")
            return False
    
    async def get_experiment_data(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get all performance data for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dictionary with variant data
        """
        try:
            if experiment_id not in self.experiment_data:
                return {}
            
            experiment_data = {}
            
            for variant_id, events in self.experiment_data[experiment_id].items():
                experiment_data[variant_id] = {
                    'events': [event.to_dict() for event in events],
                    'event_count': len(events),
                    'latest_event': events[-1].timestamp.isoformat() if events else None
                }
            
            return experiment_data
            
        except Exception as e:
            logger.error(f"Failed to get experiment data: {str(e)}")
            return {}
    
    async def calculate_variant_metrics(
        self,
        experiment_id: str,
        variant_id: str,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, float]:
        """
        Calculate performance metrics for a variant.
        
        Args:
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            time_window: Time window for calculation
            
        Returns:
            Dictionary of calculated metrics
        """
        try:
            if experiment_id not in self.experiment_data:
                return {}
            
            if variant_id not in self.experiment_data[experiment_id]:
                return {}
            
            events = self.experiment_data[experiment_id][variant_id]
            
            # Filter by time window if specified
            if time_window:
                cutoff_time = datetime.now() - time_window
                events = [e for e in events if e.timestamp >= cutoff_time]
            
            if not events:
                return {}
            
            metrics = {}
            
            # Calculate basic metrics
            total_events = len(events)
            metrics['total_events'] = float(total_events)
            
            # Event type distribution
            event_types = defaultdict(int)
            for event in events:
                event_types[event.event_type] += 1
            
            # Conversion metrics
            impressions = event_types.get('impression', 0)
            clicks = event_types.get('click', 0)
            applications = event_types.get('apply', 0)
            
            if impressions > 0:
                metrics['click_through_rate'] = clicks / impressions
                metrics['application_rate'] = applications / impressions
            else:
                metrics['click_through_rate'] = 0.0
                metrics['application_rate'] = 0.0
            
            if clicks > 0:
                metrics['click_to_application_rate'] = applications / clicks
            else:
                metrics['click_to_application_rate'] = 0.0
            
            # User satisfaction metrics
            satisfaction_scores = []
            for event in events:
                if 'satisfaction_score' in event.event_data:
                    satisfaction_scores.append(event.event_data['satisfaction_score'])
            
            if satisfaction_scores:
                metrics['average_satisfaction'] = np.mean(satisfaction_scores)
                metrics['satisfaction_std'] = np.std(satisfaction_scores)
            else:
                metrics['average_satisfaction'] = 0.0
                metrics['satisfaction_std'] = 0.0
            
            # NDCG metrics
            ndcg_scores = []
            for event in events:
                if 'ndcg_10' in event.event_data:
                    ndcg_scores.append(event.event_data['ndcg_10'])
            
            if ndcg_scores:
                metrics['average_ndcg_10'] = np.mean(ndcg_scores)
                metrics['ndcg_std'] = np.std(ndcg_scores)
            else:
                metrics['average_ndcg_10'] = 0.0
                metrics['ndcg_std'] = 0.0
            
            # Latency metrics
            latencies = []
            for event in events:
                if 'latency_ms' in event.event_data:
                    latencies.append(event.event_data['latency_ms'])
            
            if latencies:
                metrics['average_latency_ms'] = np.mean(latencies)
                metrics['p95_latency_ms'] = np.percentile(latencies, 95)
                metrics['p99_latency_ms'] = np.percentile(latencies, 99)
            else:
                metrics['average_latency_ms'] = 0.0
                metrics['p95_latency_ms'] = 0.0
                metrics['p99_latency_ms'] = 0.0
            
            # Error rate
            error_events = sum(1 for event in events if event.event_type == 'error')
            metrics['error_rate'] = error_events / total_events if total_events > 0 else 0.0
            
            # Engagement metrics
            unique_users = len(set(event.user_id for event in events))
            metrics['unique_users'] = float(unique_users)
            metrics['events_per_user'] = total_events / unique_users if unique_users > 0 else 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate variant metrics: {str(e)}")
            return {}
    
    async def create_performance_snapshot(
        self,
        experiment_id: str,
        variant_id: str
    ) -> Optional[PerformanceSnapshot]:
        """Create a performance snapshot for a variant."""
        try:
            metrics = await self.calculate_variant_metrics(experiment_id, variant_id)
            
            if not metrics:
                return None
            
            # Calculate confidence intervals for key metrics
            confidence_intervals = {}
            
            events = self.experiment_data[experiment_id][variant_id]
            sample_size = len(events)
            
            if sample_size > 30:  # Sufficient sample size for confidence intervals
                # CTR confidence interval
                if 'click_through_rate' in metrics:
                    ctr = metrics['click_through_rate']
                    se = np.sqrt(ctr * (1 - ctr) / sample_size)
                    z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
                    margin = z_score * se
                    confidence_intervals['click_through_rate'] = (
                        max(0, ctr - margin), min(1, ctr + margin)
                    )
                
                # Satisfaction confidence interval
                if 'average_satisfaction' in metrics and metrics['satisfaction_std'] > 0:
                    mean_sat = metrics['average_satisfaction']
                    std_sat = metrics['satisfaction_std']
                    se = std_sat / np.sqrt(sample_size)
                    t_score = stats.t.ppf((1 + self.confidence_level) / 2, sample_size - 1)
                    margin = t_score * se
                    confidence_intervals['average_satisfaction'] = (
                        mean_sat - margin, mean_sat + margin
                    )
            
            snapshot = PerformanceSnapshot(
                experiment_id=experiment_id,
                variant_id=variant_id,
                timestamp=datetime.now(),
                metrics=metrics,
                sample_size=sample_size,
                confidence_interval=confidence_intervals
            )
            
            # Store snapshot
            self.performance_snapshots[experiment_id].append(snapshot)
            self.monitor_stats['snapshots_created'] += 1
            
            # Keep only recent snapshots
            cutoff_time = datetime.now() - timedelta(days=self.retention_days)
            self.performance_snapshots[experiment_id] = [
                s for s in self.performance_snapshots[experiment_id]
                if s.timestamp >= cutoff_time
            ]
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to create performance snapshot: {str(e)}")
            return None
    
    async def _check_performance_alerts(self, experiment_id: str, variant_id: str):
        """Check for performance alerts and trigger notifications."""
        try:
            # Create current snapshot
            current_snapshot = await self.create_performance_snapshot(experiment_id, variant_id)
            
            if not current_snapshot:
                return
            
            # Get baseline snapshot (first snapshot or control variant)
            baseline_snapshot = self._get_baseline_snapshot(experiment_id, variant_id)
            
            if not baseline_snapshot:
                return
            
            alerts = []
            
            # Check for performance degradation
            if 'average_satisfaction' in current_snapshot.metrics and \
               'average_satisfaction' in baseline_snapshot.metrics:
                
                current_satisfaction = current_snapshot.metrics['average_satisfaction']
                baseline_satisfaction = baseline_snapshot.metrics['average_satisfaction']
                
                if baseline_satisfaction > 0:
                    degradation = (baseline_satisfaction - current_satisfaction) / baseline_satisfaction
                    
                    if degradation > self.alert_thresholds['performance_degradation']:
                        alerts.append({
                            'type': 'performance_degradation',
                            'severity': 'high',
                            'message': f'Satisfaction degraded by {degradation:.1%} in variant {variant_id}',
                            'current_value': current_satisfaction,
                            'baseline_value': baseline_satisfaction
                        })
            
            # Check for error rate increase
            if 'error_rate' in current_snapshot.metrics and \
               'error_rate' in baseline_snapshot.metrics:
                
                current_error_rate = current_snapshot.metrics['error_rate']
                baseline_error_rate = baseline_snapshot.metrics['error_rate']
                
                error_increase = current_error_rate - baseline_error_rate
                
                if error_increase > self.alert_thresholds['error_rate_increase']:
                    alerts.append({
                        'type': 'error_rate_increase',
                        'severity': 'high',
                        'message': f'Error rate increased by {error_increase:.1%} in variant {variant_id}',
                        'current_value': current_error_rate,
                        'baseline_value': baseline_error_rate
                    })
            
            # Check for latency increase
            if 'average_latency_ms' in current_snapshot.metrics and \
               'average_latency_ms' in baseline_snapshot.metrics:
                
                current_latency = current_snapshot.metrics['average_latency_ms']
                baseline_latency = baseline_snapshot.metrics['average_latency_ms']
                
                if baseline_latency > 0:
                    latency_ratio = current_latency / baseline_latency
                    
                    if latency_ratio > self.alert_thresholds['latency_increase']:
                        alerts.append({
                            'type': 'latency_increase',
                            'severity': 'medium',
                            'message': f'Latency increased by {latency_ratio:.1f}x in variant {variant_id}',
                            'current_value': current_latency,
                            'baseline_value': baseline_latency
                        })
            
            # Check sample size
            if current_snapshot.sample_size < self.alert_thresholds['sample_size_minimum']:
                alerts.append({
                    'type': 'low_sample_size',
                    'severity': 'low',
                    'message': f'Low sample size ({current_snapshot.sample_size}) in variant {variant_id}',
                    'current_value': current_snapshot.sample_size,
                    'baseline_value': self.alert_thresholds['sample_size_minimum']
                })
            
            # Process alerts
            if alerts:
                await self._process_alerts(experiment_id, variant_id, alerts)
            
        except Exception as e:
            logger.error(f"Performance alert check failed: {str(e)}")
    
    def _get_baseline_snapshot(
        self,
        experiment_id: str,
        variant_id: str
    ) -> Optional[PerformanceSnapshot]:
        """Get baseline snapshot for comparison."""
        try:
            if experiment_id not in self.performance_snapshots:
                return None
            
            snapshots = self.performance_snapshots[experiment_id]
            
            # Try to find control variant snapshot
            control_snapshots = [s for s in snapshots if 'control' in s.variant_id]
            if control_snapshots:
                return control_snapshots[0]  # First control snapshot
            
            # Fallback to first snapshot of any variant
            variant_snapshots = [s for s in snapshots if s.variant_id == variant_id]
            if variant_snapshots:
                return variant_snapshots[0]  # First snapshot of this variant
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get baseline snapshot: {str(e)}")
            return None
    
    async def _process_alerts(
        self,
        experiment_id: str,
        variant_id: str,
        alerts: List[Dict[str, Any]]
    ):
        """Process performance alerts."""
        try:
            for alert in alerts:
                self.monitor_stats['alerts_triggered'] += 1
                
                # Log alert
                logger.warning(
                    f"Performance Alert - {alert['type']}: {alert['message']} "
                    f"(Experiment: {experiment_id}, Variant: {variant_id})"
                )
                
                # Here you would typically:
                # - Send notifications to stakeholders
                # - Update dashboards
                # - Trigger automatic responses
                # - Store alerts in database
                
        except Exception as e:
            logger.error(f"Alert processing failed: {str(e)}")
    
    async def get_variant_comparison(
        self,
        experiment_id: str,
        baseline_variant: str,
        comparison_variant: str
    ) -> Dict[str, Any]:
        """Compare performance between two variants."""
        try:
            baseline_metrics = await self.calculate_variant_metrics(experiment_id, baseline_variant)
            comparison_metrics = await self.calculate_variant_metrics(experiment_id, comparison_variant)
            
            if not baseline_metrics or not comparison_metrics:
                return {'comparison_available': False}
            
            comparison = {
                'comparison_available': True,
                'baseline_variant': baseline_variant,
                'comparison_variant': comparison_variant,
                'metric_differences': {},
                'statistical_tests': {},
                'recommendations': []
            }
            
            # Calculate differences
            for metric in set(baseline_metrics.keys()).union(comparison_metrics.keys()):
                baseline_value = baseline_metrics.get(metric, 0.0)
                comparison_value = comparison_metrics.get(metric, 0.0)
                
                if baseline_value != 0:
                    relative_change = (comparison_value - baseline_value) / baseline_value
                    comparison['metric_differences'][metric] = {
                        'baseline': baseline_value,
                        'comparison': comparison_value,
                        'absolute_change': comparison_value - baseline_value,
                        'relative_change': relative_change
                    }
            
            # Statistical significance tests
            if experiment_id in self.experiment_data:
                baseline_events = self.experiment_data[experiment_id].get(baseline_variant, [])
                comparison_events = self.experiment_data[experiment_id].get(comparison_variant, [])
                
                # CTR test
                baseline_ctr_data = self._extract_ctr_data(baseline_events)
                comparison_ctr_data = self._extract_ctr_data(comparison_events)
                
                if baseline_ctr_data and comparison_ctr_data:
                    ctr_test = self._perform_proportion_test(baseline_ctr_data, comparison_ctr_data)
                    comparison['statistical_tests']['click_through_rate'] = ctr_test
                
                # Satisfaction test
                baseline_satisfaction = self._extract_satisfaction_data(baseline_events)
                comparison_satisfaction = self._extract_satisfaction_data(comparison_events)
                
                if baseline_satisfaction and comparison_satisfaction:
                    satisfaction_test = self._perform_t_test(baseline_satisfaction, comparison_satisfaction)
                    comparison['statistical_tests']['satisfaction'] = satisfaction_test
            
            # Generate recommendations
            comparison['recommendations'] = self._generate_comparison_recommendations(comparison)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Variant comparison failed: {str(e)}")
            return {'comparison_available': False, 'error': str(e)}
    
    def _extract_ctr_data(self, events: List[MetricEvent]) -> Optional[Tuple[int, int]]:
        """Extract click-through rate data (clicks, impressions)."""
        try:
            impressions = sum(1 for event in events if event.event_type == 'impression')
            clicks = sum(1 for event in events if event.event_type == 'click')
            
            return (clicks, impressions) if impressions > 0 else None
            
        except Exception:
            return None
    
    def _extract_satisfaction_data(self, events: List[MetricEvent]) -> Optional[List[float]]:
        """Extract satisfaction scores."""
        try:
            scores = []
            for event in events:
                if 'satisfaction_score' in event.event_data:
                    scores.append(event.event_data['satisfaction_score'])
            
            return scores if scores else None
            
        except Exception:
            return None
    
    def _perform_proportion_test(
        self,
        baseline_data: Tuple[int, int],
        comparison_data: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Perform statistical test for proportions."""
        try:
            baseline_successes, baseline_trials = baseline_data
            comparison_successes, comparison_trials = comparison_data
            
            # Two-proportion z-test
            baseline_rate = baseline_successes / baseline_trials
            comparison_rate = comparison_successes / comparison_trials
            
            pooled_rate = (baseline_successes + comparison_successes) / (baseline_trials + comparison_trials)
            pooled_se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/baseline_trials + 1/comparison_trials))
            
            if pooled_se > 0:
                z_score = (comparison_rate - baseline_rate) / pooled_se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            else:
                z_score = 0
                p_value = 1.0
            
            return {
                'test_type': 'two_proportion_z_test',
                'baseline_rate': baseline_rate,
                'comparison_rate': comparison_rate,
                'z_score': z_score,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'sample_sizes': [baseline_trials, comparison_trials]
            }
            
        except Exception as e:
            logger.error(f"Proportion test failed: {str(e)}")
            return {'test_type': 'failed', 'error': str(e)}
    
    def _perform_t_test(
        self,
        baseline_data: List[float],
        comparison_data: List[float]
    ) -> Dict[str, Any]:
        """Perform t-test for continuous variables."""
        try:
            if len(baseline_data) < 2 or len(comparison_data) < 2:
                return {'test_type': 'insufficient_data'}
            
            t_statistic, p_value = stats.ttest_ind(baseline_data, comparison_data)
            
            return {
                'test_type': 'independent_t_test',
                'baseline_mean': np.mean(baseline_data),
                'comparison_mean': np.mean(comparison_data),
                't_statistic': t_statistic,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'sample_sizes': [len(baseline_data), len(comparison_data)]
            }
            
        except Exception as e:
            logger.error(f"T-test failed: {str(e)}")
            return {'test_type': 'failed', 'error': str(e)}
    
    def _generate_comparison_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []
        
        try:
            metric_diffs = comparison.get('metric_differences', {})
            statistical_tests = comparison.get('statistical_tests', {})
            
            # Check for significant improvements
            if 'click_through_rate' in statistical_tests:
                ctr_test = statistical_tests['click_through_rate']
                if ctr_test.get('is_significant') and ctr_test.get('comparison_rate', 0) > ctr_test.get('baseline_rate', 0):
                    improvement = (ctr_test['comparison_rate'] - ctr_test['baseline_rate']) / ctr_test['baseline_rate']
                    recommendations.append(f"Significant CTR improvement of {improvement:.1%} detected")
            
            if 'satisfaction' in statistical_tests:
                sat_test = statistical_tests['satisfaction']
                if sat_test.get('is_significant') and sat_test.get('comparison_mean', 0) > sat_test.get('baseline_mean', 0):
                    recommendations.append("Significant satisfaction improvement detected")
            
            # Check for concerning trends
            if 'error_rate' in metric_diffs:
                error_diff = metric_diffs['error_rate']
                if error_diff.get('relative_change', 0) > 0.1:  # 10% increase
                    recommendations.append("Monitor error rate increase closely")
            
            if 'average_latency_ms' in metric_diffs:
                latency_diff = metric_diffs['average_latency_ms']
                if latency_diff.get('relative_change', 0) > 0.2:  # 20% increase
                    recommendations.append("Address latency performance degradation")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Continue monitoring - no significant differences detected yet")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            return ["Unable to generate recommendations due to analysis error"]
    
    async def finalize_experiment(self, experiment_id: str) -> bool:
        """Finalize monitoring for a completed experiment."""
        try:
            if experiment_id not in self.experiment_data:
                return True  # Already finalized
            
            # Create final snapshots for all variants
            for variant_id in self.experiment_data[experiment_id]:
                await self.create_performance_snapshot(experiment_id, variant_id)
            
            logger.info(f"Finalized performance monitoring for experiment {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to finalize experiment monitoring: {str(e)}")
            return False
    
    async def get_monitor_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitor statistics."""
        try:
            # Calculate additional stats
            active_experiments = len(self.experiment_data)
            total_variants = sum(len(variants) for variants in self.experiment_data.values())
            total_snapshots = sum(len(snapshots) for snapshots in self.performance_snapshots.values())
            
            return {
                'monitor_stats': dict(self.monitor_stats),
                'active_experiments': active_experiments,
                'total_variants_monitored': total_variants,
                'total_snapshots': total_snapshots,
                'total_users_tracked': len(self.user_assignments),
                'alert_thresholds': self.alert_thresholds.copy(),
                'configuration': {
                    'snapshot_interval_minutes': self.snapshot_interval_minutes,
                    'retention_days': self.retention_days,
                    'confidence_level': self.confidence_level
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get monitor stats: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Clean up performance monitor resources."""
        try:
            # Clear all data
            self.experiment_data.clear()
            self.performance_snapshots.clear()
            self.user_assignments.clear()
            
            # Reset stats
            self.monitor_stats = {
                'total_events': 0,
                'events_by_experiment': defaultdict(int),
                'events_by_type': defaultdict(int),
                'snapshots_created': 0,
                'alerts_triggered': 0
            }
            
            logger.info("PerformanceMonitor cleanup completed")
            
        except Exception as e:
            logger.error(f"PerformanceMonitor cleanup failed: {str(e)}")