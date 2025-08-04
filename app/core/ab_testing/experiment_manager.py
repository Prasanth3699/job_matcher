"""
Experiment Manager for A/B testing framework.

This module manages A/B testing experiments, traffic allocation,
and statistical analysis for model comparison.
"""

import asyncio
import hashlib
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from scipy import stats

from app.utils.logger import logger
from app.core.constants.ml_constants import (
    PerformanceMetrics,
    EnsembleConfig,
    RankingConfig
)
from app.core.ab_testing.model_variant import ModelVariant
from app.core.ab_testing.traffic_splitter import TrafficSplitter
from app.core.ab_testing.performance_monitor import PerformanceMonitor


class ExperimentStatus(str, Enum):
    """Status of A/B testing experiments."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"


class ExperimentType(str, Enum):
    """Types of A/B testing experiments."""
    MODEL_COMPARISON = "model_comparison"
    FEATURE_COMPARISON = "feature_comparison"
    ALGORITHM_COMPARISON = "algorithm_comparison"
    RANKING_COMPARISON = "ranking_comparison"
    ENSEMBLE_COMPARISON = "ensemble_comparison"


@dataclass
class ExperimentConfig:
    """Configuration for an A/B testing experiment."""
    experiment_id: str
    name: str
    description: str
    experiment_type: ExperimentType
    variants: List[str]
    traffic_allocation: Dict[str, float]
    success_metrics: List[str]
    minimum_sample_size: int
    statistical_power: float
    significance_level: float
    start_date: datetime
    end_date: Optional[datetime]
    auto_terminate_conditions: Dict[str, Any]


@dataclass
class ExperimentResult:
    """Results of an A/B testing experiment."""
    experiment_id: str
    variant_results: Dict[str, Dict[str, Any]]
    statistical_significance: Dict[str, bool]
    confidence_intervals: Dict[str, Dict[str, Tuple[float, float]]]
    winner: Optional[str]
    recommendation: str
    sample_sizes: Dict[str, int]
    duration_days: float
    analysis_timestamp: datetime


class ExperimentManager:
    """
    Comprehensive A/B testing experiment manager for ML model comparison
    with statistical analysis and automated decision making.
    """
    
    def __init__(self):
        """Initialize the experiment manager."""
        self.active_experiments: Dict[str, ExperimentConfig] = {}
        self.experiment_history: List[ExperimentConfig] = []
        self.traffic_splitter = TrafficSplitter()
        self.performance_monitor = PerformanceMonitor()
        
        # Statistical parameters
        self.default_significance_level = 0.05
        self.default_statistical_power = 0.8
        self.minimum_effect_size = 0.05  # 5% minimum detectable effect
        
        # Performance tracking
        self.experiment_stats = {
            'total_experiments': 0,
            'active_experiments': 0,
            'completed_experiments': 0,
            'users_in_experiments': 0,
            'conversion_improvements': []
        }
        
        logger.info("ExperimentManager initialized successfully")
    
    async def initialize(self):
        """
        Async initialization method for interface compatibility.
        ExperimentManager initialization is already completed in __init__.
        """
        logger.info("ExperimentManager async initialization completed")
    
    async def create_experiment(
        self,
        name: str,
        description: str,
        experiment_type: ExperimentType,
        variants: List[ModelVariant],
        traffic_allocation: Optional[Dict[str, float]] = None,
        success_metrics: Optional[List[str]] = None,
        duration_days: int = 14,
        auto_terminate_conditions: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new A/B testing experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            experiment_type: Type of experiment
            variants: List of model variants to test
            traffic_allocation: Custom traffic allocation (defaults to equal split)
            success_metrics: Metrics to optimize for
            duration_days: Experiment duration in days
            auto_terminate_conditions: Conditions for automatic termination
            
        Returns:
            Experiment ID
        """
        try:
            experiment_id = str(uuid.uuid4())
            
            # Validate variants
            if len(variants) < 2:
                raise ValueError("At least 2 variants required for A/B testing")
            
            variant_names = [variant.variant_id for variant in variants]
            
            # Set default traffic allocation (equal split)
            if traffic_allocation is None:
                allocation_per_variant = 1.0 / len(variants)
                traffic_allocation = {name: allocation_per_variant for name in variant_names}
            
            # Validate traffic allocation
            if not self._validate_traffic_allocation(traffic_allocation, variant_names):
                raise ValueError("Invalid traffic allocation")
            
            # Set default success metrics
            if success_metrics is None:
                success_metrics = ['ndcg_at_10', 'user_satisfaction', 'click_through_rate']
            
            # Set default auto-termination conditions
            if auto_terminate_conditions is None:
                auto_terminate_conditions = {
                    'max_duration_days': duration_days,
                    'min_sample_size_per_variant': 1000,
                    'significance_threshold': self.default_significance_level,
                    'early_stopping_enabled': True,
                    'performance_degradation_threshold': 0.1
                }
            
            # Calculate minimum sample size
            minimum_sample_size = self._calculate_minimum_sample_size(
                len(variants), 
                self.minimum_effect_size,
                self.default_statistical_power,
                self.default_significance_level
            )
            
            # Create experiment configuration
            config = ExperimentConfig(
                experiment_id=experiment_id,
                name=name,
                description=description,
                experiment_type=experiment_type,
                variants=variant_names,
                traffic_allocation=traffic_allocation,
                success_metrics=success_metrics,
                minimum_sample_size=minimum_sample_size,
                statistical_power=self.default_statistical_power,
                significance_level=self.default_significance_level,
                start_date=datetime.now(),
                end_date=datetime.now() + timedelta(days=duration_days),
                auto_terminate_conditions=auto_terminate_conditions
            )
            
            # Register variants with traffic splitter
            await self.traffic_splitter.register_experiment(experiment_id, variants, traffic_allocation)
            
            # Initialize performance monitoring
            await self.performance_monitor.initialize_experiment(experiment_id, config)
            
            # Store experiment
            self.active_experiments[experiment_id] = config
            self.experiment_stats['total_experiments'] += 1
            self.experiment_stats['active_experiments'] += 1
            
            logger.info(f"Created experiment: {name} ({experiment_id}) with {len(variants)} variants")
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Experiment creation failed: {str(e)}")
            raise
    
    async def assign_variant(
        self,
        experiment_id: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Assign a user to a variant in an experiment.
        
        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            context: Additional context for assignment
            
        Returns:
            Assigned variant ID or None if not in experiment
        """
        try:
            if experiment_id not in self.active_experiments:
                return None
            
            config = self.active_experiments[experiment_id]
            
            # Check if experiment is active
            current_time = datetime.now()
            if current_time > config.end_date:
                logger.warning(f"Experiment {experiment_id} has expired")
                return None
            
            # Get variant assignment from traffic splitter
            variant_id = await self.traffic_splitter.assign_variant(
                experiment_id, user_id, context
            )
            
            if variant_id:
                # Track assignment
                await self.performance_monitor.track_assignment(
                    experiment_id, user_id, variant_id, current_time
                )
                
                logger.debug(f"Assigned user {user_id} to variant {variant_id} in experiment {experiment_id}")
            
            return variant_id
            
        except Exception as e:
            logger.error(f"Variant assignment failed: {str(e)}")
            return None
    
    async def track_event(
        self,
        experiment_id: str,
        user_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Track an event for experiment analysis.
        
        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            event_type: Type of event (e.g., 'click', 'conversion')
            event_data: Event-specific data
            timestamp: Event timestamp
            
        Returns:
            True if tracked successfully
        """
        try:
            if experiment_id not in self.active_experiments:
                return False
            
            timestamp = timestamp or datetime.now()
            
            # Track event with performance monitor
            success = await self.performance_monitor.track_event(
                experiment_id, user_id, event_type, event_data, timestamp
            )
            
            if success:
                logger.debug(f"Tracked event {event_type} for user {user_id} in experiment {experiment_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Event tracking failed: {str(e)}")
            return False
    
    async def analyze_experiment(
        self,
        experiment_id: str,
        force_analysis: bool = False
    ) -> Optional[ExperimentResult]:
        """
        Analyze experiment results and determine statistical significance.
        
        Args:
            experiment_id: Experiment identifier
            force_analysis: Force analysis even if sample size is insufficient
            
        Returns:
            Experiment analysis results
        """
        try:
            if experiment_id not in self.active_experiments:
                logger.error(f"Experiment {experiment_id} not found")
                return None
            
            config = self.active_experiments[experiment_id]
            
            # Get performance data
            performance_data = await self.performance_monitor.get_experiment_data(experiment_id)
            
            if not performance_data:
                logger.warning(f"No performance data available for experiment {experiment_id}")
                return None
            
            # Check minimum sample size
            sample_sizes = {variant: len(data.get('events', [])) 
                          for variant, data in performance_data.items()}
            
            total_samples = sum(sample_sizes.values())
            
            if not force_analysis and total_samples < config.minimum_sample_size:
                logger.info(f"Insufficient sample size for experiment {experiment_id}: {total_samples}/{config.minimum_sample_size}")
                return None
            
            # Analyze each success metric
            variant_results = {}
            statistical_significance = {}
            confidence_intervals = {}
            
            for metric in config.success_metrics:
                metric_results, significance, confidence = await self._analyze_metric(
                    performance_data, metric, config.significance_level
                )
                
                # Store results by variant
                for variant_id in config.variants:
                    if variant_id not in variant_results:
                        variant_results[variant_id] = {}
                    variant_results[variant_id][metric] = metric_results.get(variant_id, 0.0)
                
                statistical_significance[metric] = significance
                confidence_intervals[metric] = confidence
            
            # Determine winner
            winner = self._determine_winner(variant_results, config.success_metrics)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                variant_results, statistical_significance, winner, config
            )
            
            # Calculate experiment duration
            duration_days = (datetime.now() - config.start_date).days
            
            result = ExperimentResult(
                experiment_id=experiment_id,
                variant_results=variant_results,
                statistical_significance=statistical_significance,
                confidence_intervals=confidence_intervals,
                winner=winner,
                recommendation=recommendation,
                sample_sizes=sample_sizes,
                duration_days=duration_days,
                analysis_timestamp=datetime.now()
            )
            
            logger.info(f"Analyzed experiment {experiment_id}: winner={winner}, samples={total_samples}")
            
            return result
            
        except Exception as e:
            logger.error(f"Experiment analysis failed: {str(e)}")
            return None
    
    async def _analyze_metric(
        self,
        performance_data: Dict[str, Any],
        metric: str,
        significance_level: float
    ) -> Tuple[Dict[str, float], bool, Dict[str, Tuple[float, float]]]:
        """Analyze a specific metric across variants."""
        try:
            # Extract metric values for each variant
            variant_values = {}
            
            for variant_id, data in performance_data.items():
                events = data.get('events', [])
                
                # Calculate metric value based on type
                if metric == 'ndcg_at_10':
                    values = [event.get('ndcg_10', 0.0) for event in events if 'ndcg_10' in event]
                elif metric == 'user_satisfaction':
                    values = [event.get('satisfaction_score', 0.0) for event in events if 'satisfaction_score' in event]
                elif metric == 'click_through_rate':
                    clicks = sum(1 for event in events if event.get('event_type') == 'click')
                    impressions = sum(1 for event in events if event.get('event_type') == 'impression')
                    values = [clicks / max(impressions, 1)]
                else:
                    # Generic metric extraction
                    values = [event.get(metric, 0.0) for event in events if metric in event]
                
                if values:
                    variant_values[variant_id] = np.mean(values)
                else:
                    variant_values[variant_id] = 0.0
            
            # Statistical significance test (t-test for two variants, ANOVA for more)
            if len(variant_values) == 2:
                variant_ids = list(variant_values.keys())
                values_a = [event.get(metric, 0.0) for event in performance_data[variant_ids[0]].get('events', [])]
                values_b = [event.get(metric, 0.0) for event in performance_data[variant_ids[1]].get('events', [])]
                
                if len(values_a) > 1 and len(values_b) > 1:
                    statistic, p_value = stats.ttest_ind(values_a, values_b)
                    is_significant = p_value < significance_level
                else:
                    is_significant = False
            else:
                # ANOVA for multiple variants
                all_values = []
                for variant_id in variant_values.keys():
                    variant_data = [event.get(metric, 0.0) for event in performance_data[variant_id].get('events', [])]
                    if variant_data:
                        all_values.append(variant_data)
                
                if len(all_values) > 1 and all(len(v) > 1 for v in all_values):
                    statistic, p_value = stats.f_oneway(*all_values)
                    is_significant = p_value < significance_level
                else:
                    is_significant = False
            
            # Calculate confidence intervals
            confidence_intervals = {}
            for variant_id, mean_value in variant_values.items():
                values = [event.get(metric, 0.0) for event in performance_data[variant_id].get('events', [])]
                if len(values) > 1:
                    sem = stats.sem(values)
                    h = sem * stats.t.ppf((1 + (1 - significance_level)) / 2., len(values) - 1)
                    confidence_intervals[variant_id] = (mean_value - h, mean_value + h)
                else:
                    confidence_intervals[variant_id] = (mean_value, mean_value)
            
            return variant_values, is_significant, confidence_intervals
            
        except Exception as e:
            logger.error(f"Metric analysis failed for {metric}: {str(e)}")
            return {}, False, {}
    
    def _determine_winner(
        self,
        variant_results: Dict[str, Dict[str, float]],
        success_metrics: List[str]
    ) -> Optional[str]:
        """Determine the winning variant based on success metrics."""
        try:
            if not variant_results:
                return None
            
            # Calculate composite score for each variant
            variant_scores = {}
            
            for variant_id, metrics in variant_results.items():
                # Weight metrics equally for now (could be configurable)
                metric_weight = 1.0 / len(success_metrics)
                
                composite_score = 0.0
                for metric in success_metrics:
                    metric_value = metrics.get(metric, 0.0)
                    composite_score += metric_value * metric_weight
                
                variant_scores[variant_id] = composite_score
            
            # Find variant with highest composite score
            winner = max(variant_scores.items(), key=lambda x: x[1])[0]
            
            return winner
            
        except Exception as e:
            logger.error(f"Winner determination failed: {str(e)}")
            return None
    
    def _generate_recommendation(
        self,
        variant_results: Dict[str, Dict[str, float]],
        statistical_significance: Dict[str, bool],
        winner: Optional[str],
        config: ExperimentConfig
    ) -> str:
        """Generate actionable recommendation based on results."""
        try:
            if not winner:
                return "No clear winner identified. Consider extending experiment duration or revising variants."
            
            # Check statistical significance
            significant_metrics = [metric for metric, is_sig in statistical_significance.items() if is_sig]
            
            if not significant_metrics:
                return f"Variant {winner} shows highest performance but differences are not statistically significant. Consider longer experiment or larger sample size."
            
            # Calculate improvement
            if len(variant_results) >= 2:
                variant_ids = list(variant_results.keys())
                control_variant = variant_ids[0]  # Assume first variant is control
                
                if winner != control_variant:
                    improvements = []
                    for metric in significant_metrics:
                        control_value = variant_results[control_variant].get(metric, 0.0)
                        winner_value = variant_results[winner].get(metric, 0.0)
                        
                        if control_value > 0:
                            improvement = ((winner_value - control_value) / control_value) * 100
                            improvements.append(f"{metric}: {improvement:.1f}%")
                    
                    improvement_text = ", ".join(improvements)
                    return f"Deploy variant {winner}. Statistically significant improvements in {improvement_text}."
                else:
                    return f"Control variant {winner} remains the best performer. No changes needed."
            
            return f"Deploy variant {winner} based on superior performance across {len(significant_metrics)} metrics."
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            return "Unable to generate recommendation due to analysis error."
    
    async def terminate_experiment(
        self,
        experiment_id: str,
        reason: str = "Manual termination"
    ) -> bool:
        """
        Terminate an active experiment.
        
        Args:
            experiment_id: Experiment identifier
            reason: Reason for termination
            
        Returns:
            True if terminated successfully
        """
        try:
            if experiment_id not in self.active_experiments:
                logger.warning(f"Experiment {experiment_id} not found")
                return False
            
            config = self.active_experiments[experiment_id]
            
            # Move to history
            self.experiment_history.append(config)
            del self.active_experiments[experiment_id]
            
            # Clean up traffic splitter
            await self.traffic_splitter.remove_experiment(experiment_id)
            
            # Finalize performance monitoring
            await self.performance_monitor.finalize_experiment(experiment_id)
            
            # Update stats
            self.experiment_stats['active_experiments'] -= 1
            self.experiment_stats['completed_experiments'] += 1
            
            logger.info(f"Terminated experiment {experiment_id}: {reason}")
            
            return True
            
        except Exception as e:
            logger.error(f"Experiment termination failed: {str(e)}")
            return False
    
    async def check_auto_termination(self) -> List[str]:
        """Check for experiments that should be automatically terminated."""
        terminated_experiments = []
        
        try:
            for experiment_id, config in list(self.active_experiments.items()):
                should_terminate = False
                termination_reason = ""
                
                # Check duration
                current_time = datetime.now()
                if current_time > config.end_date:
                    should_terminate = True
                    termination_reason = "Experiment duration exceeded"
                
                # Check auto-termination conditions
                auto_conditions = config.auto_terminate_conditions
                
                if auto_conditions.get('early_stopping_enabled', False):
                    # Get current results
                    result = await self.analyze_experiment(experiment_id, force_analysis=True)
                    
                    if result:
                        # Check for early statistical significance
                        significant_metrics = [k for k, v in result.statistical_significance.items() if v]
                        
                        if len(significant_metrics) >= len(config.success_metrics) * 0.8:
                            should_terminate = True
                            termination_reason = "Early statistical significance achieved"
                        
                        # Check for performance degradation
                        degradation_threshold = auto_conditions.get('performance_degradation_threshold', 0.1)
                        
                        for variant_id, results in result.variant_results.items():
                            for metric, value in results.items():
                                if value < -degradation_threshold:  # Significant degradation
                                    should_terminate = True
                                    termination_reason = f"Performance degradation detected in {variant_id}"
                                    break
                
                if should_terminate:
                    await self.terminate_experiment(experiment_id, termination_reason)
                    terminated_experiments.append(experiment_id)
            
            return terminated_experiments
            
        except Exception as e:
            logger.error(f"Auto-termination check failed: {str(e)}")
            return []
    
    def _validate_traffic_allocation(
        self,
        allocation: Dict[str, float],
        variant_names: List[str]
    ) -> bool:
        """Validate traffic allocation configuration."""
        try:
            # Check if all variants have allocation
            if set(allocation.keys()) != set(variant_names):
                return False
            
            # Check if allocations sum to 1.0
            total_allocation = sum(allocation.values())
            if abs(total_allocation - 1.0) > 0.001:
                return False
            
            # Check if all allocations are positive
            if any(alloc <= 0 for alloc in allocation.values()):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _calculate_minimum_sample_size(
        self,
        num_variants: int,
        effect_size: float,
        power: float,
        significance_level: float
    ) -> int:
        """Calculate minimum sample size for statistical power."""
        try:
            # Simplified sample size calculation
            # For more accuracy, would use power analysis libraries
            
            from scipy.stats import norm
            
            z_alpha = norm.ppf(1 - significance_level / 2)
            z_beta = norm.ppf(power)
            
            # Adjust for multiple comparisons (Bonferroni correction)
            adjusted_alpha = significance_level / num_variants
            z_alpha_adj = norm.ppf(1 - adjusted_alpha / 2)
            
            # Sample size per variant
            n_per_variant = 2 * ((z_alpha_adj + z_beta) / effect_size) ** 2
            
            # Total sample size
            total_sample_size = int(n_per_variant * num_variants)
            
            # Minimum of 100 per variant
            minimum_total = max(total_sample_size, 100 * num_variants)
            
            return minimum_total
            
        except Exception as e:
            logger.error(f"Sample size calculation failed: {str(e)}")
            return 1000  # Default minimum
    
    async def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get current status and metrics for an experiment."""
        try:
            if experiment_id not in self.active_experiments:
                return None
            
            config = self.active_experiments[experiment_id]
            
            # Get performance data
            performance_data = await self.performance_monitor.get_experiment_data(experiment_id)
            
            # Calculate current metrics
            sample_sizes = {}
            if performance_data:
                sample_sizes = {variant: len(data.get('events', [])) 
                              for variant, data in performance_data.items()}
            
            total_samples = sum(sample_sizes.values())
            progress = min(1.0, total_samples / config.minimum_sample_size)
            
            # Time remaining
            time_remaining = config.end_date - datetime.now()
            
            return {
                'experiment_id': experiment_id,
                'name': config.name,
                'status': 'active',
                'progress': progress,
                'total_samples': total_samples,
                'sample_sizes': sample_sizes,
                'time_remaining_days': time_remaining.days,
                'variants': config.variants,
                'traffic_allocation': config.traffic_allocation,
                'success_metrics': config.success_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get experiment status: {str(e)}")
            return None
    
    async def get_manager_stats(self) -> Dict[str, Any]:
        """Get comprehensive experiment manager statistics."""
        try:
            # Calculate additional stats
            active_experiments_list = []
            for exp_id, config in self.active_experiments.items():
                status = await self.get_experiment_status(exp_id)
                if status:
                    active_experiments_list.append(status)
            
            return {
                'experiment_stats': self.experiment_stats.copy(),
                'active_experiments': active_experiments_list,
                'total_active': len(self.active_experiments),
                'experiment_history_count': len(self.experiment_history),
                'supported_experiment_types': [t.value for t in ExperimentType],
                'default_parameters': {
                    'significance_level': self.default_significance_level,
                    'statistical_power': self.default_statistical_power,
                    'minimum_effect_size': self.minimum_effect_size
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get manager stats: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Clean up experiment manager resources."""
        try:
            # Terminate all active experiments
            for experiment_id in list(self.active_experiments.keys()):
                await self.terminate_experiment(experiment_id, "Manager cleanup")
            
            # Clean up components
            if hasattr(self.traffic_splitter, 'cleanup'):
                await self.traffic_splitter.cleanup()
            
            if hasattr(self.performance_monitor, 'cleanup'):
                await self.performance_monitor.cleanup()
            
            logger.info("ExperimentManager cleanup completed")
            
        except Exception as e:
            logger.error(f"ExperimentManager cleanup failed: {str(e)}")