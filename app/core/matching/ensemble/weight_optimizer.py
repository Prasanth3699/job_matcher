"""
Weight Optimizer for ensemble model weights.

This module handles dynamic optimization of ensemble model weights based on
performance metrics and user feedback to continuously improve matching accuracy.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from pathlib import Path

from app.utils.logger import logger
from app.core.constants.ml_constants import (
    EnsembleConfig,
    FeedbackConfig,
    PerformanceMetrics,
    MatchingWeights
)


@dataclass
class OptimizationResult:
    """Result of weight optimization process."""
    new_weights: Dict[str, float]
    improvement_score: float
    confidence: float
    optimization_method: str
    metadata: Dict[str, Any]


class WeightOptimizer:
    """
    Dynamic weight optimizer for ensemble models using multiple optimization
    strategies including feedback-based learning and performance-driven updates.
    """

    def __init__(self):
        """Initialize the weight optimizer with configuration and state."""
        self.current_weights = EnsembleConfig.MODEL_WEIGHTS.copy()
        self.historical_weights = []
        self.optimization_history = []
        self.feedback_buffer = []
        
        # Configuration
        self.feedback_config = FeedbackConfig.FEEDBACK_WEIGHTS
        self.learning_params = FeedbackConfig.LEARNING_PARAMS
        self.validation_rules = FeedbackConfig.VALIDATION_RULES
        
        # Optimization parameters
        self.min_feedback_threshold = self.learning_params['min_feedback_threshold']
        self.learning_rate = self.learning_params['learning_rate']
        self.update_frequency = self.learning_params['update_frequency']
        self.feedback_decay = self.learning_params['feedback_decay']
        
        # Performance tracking
        self.performance_history = []
        self.last_optimization = None
        
        logger.info("WeightOptimizer initialized successfully")

    async def optimize_weights(
        self,
        current_weights: Dict[str, float],
        feedback_data: List[Dict[str, Any]],
        performance_metrics: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """
        Optimize ensemble weights based on feedback and performance data.
        
        Args:
            current_weights: Current ensemble model weights
            feedback_data: List of user feedback records
            performance_metrics: Current performance metrics
            
        Returns:
            Optimized weights or None if optimization not needed
        """
        try:
            logger.info("Starting weight optimization process")
            
            # Validate inputs
            if not self._validate_inputs(current_weights, feedback_data):
                return None
            
            # Update internal state
            self.current_weights = current_weights.copy()
            self.feedback_buffer.extend(feedback_data)
            
            # Check if optimization is needed
            if not self._should_optimize():
                logger.info("Optimization not needed at this time")
                return None
            
            # Prepare optimization data
            optimization_data = self._prepare_optimization_data(
                feedback_data, performance_metrics
            )
            
            # Try multiple optimization strategies
            optimization_results = []
            
            # 1. Feedback-based optimization
            feedback_result = await self._optimize_from_feedback(optimization_data)
            if feedback_result:
                optimization_results.append(feedback_result)
            
            # 2. Performance-based optimization
            performance_result = await self._optimize_from_performance(optimization_data)
            if performance_result:
                optimization_results.append(performance_result)
            
            # 3. Gradient-based optimization
            gradient_result = await self._optimize_with_gradients(optimization_data)
            if gradient_result:
                optimization_results.append(gradient_result)
            
            # Select best optimization result
            best_result = self._select_best_optimization(optimization_results)
            
            if best_result:
                # Validate and apply new weights
                if self._validate_weights(best_result.new_weights):
                    self._update_optimization_history(best_result)
                    self.last_optimization = datetime.now()
                    
                    logger.info(
                        f"Weight optimization successful - Method: {best_result.optimization_method}, "
                        f"Improvement: {best_result.improvement_score:.4f}"
                    )
                    
                    return best_result.new_weights
                else:
                    logger.error("Optimized weights failed validation")
                    return None
            else:
                logger.info("No significant weight improvements found")
                return None
                
        except Exception as e:
            logger.error(f"Weight optimization failed: {str(e)}", exc_info=True)
            return None

    def _validate_inputs(
        self,
        weights: Dict[str, float],
        feedback_data: List[Dict[str, Any]]
    ) -> bool:
        """Validate optimization inputs."""
        try:
            # Validate weights
            if not MatchingWeights.validate_weights(weights):
                logger.error("Invalid input weights")
                return False
            
            # Validate feedback data
            if not feedback_data:
                logger.warning("No feedback data provided")
                return False
            
            # Check feedback data structure
            required_fields = ['user_id', 'job_id', 'feedback_type', 'timestamp']
            for feedback in feedback_data[:5]:  # Check first few samples
                if not all(field in feedback for field in required_fields):
                    logger.error(f"Invalid feedback data structure: {feedback}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            return False

    def _should_optimize(self) -> bool:
        """Determine if weight optimization should be performed."""
        try:
            # Check if enough feedback has accumulated
            if len(self.feedback_buffer) < self.min_feedback_threshold:
                return False
            
            # Check if enough time has passed since last optimization
            if self.last_optimization:
                time_since_last = datetime.now() - self.last_optimization
                min_interval = timedelta(hours=1)  # Minimum 1 hour between optimizations
                if time_since_last < min_interval:
                    return False
            
            # Check if performance has degraded
            if self.performance_history:
                recent_performance = np.mean([p['accuracy'] for p in self.performance_history[-10:]])
                if recent_performance < 0.7:  # Below 70% accuracy threshold
                    logger.info("Performance degradation detected, optimization needed")
                    return True
            
            # Check feedback update frequency
            recent_feedback = [
                f for f in self.feedback_buffer
                if datetime.fromisoformat(f.get('timestamp', '2024-01-01'))
                > datetime.now() - timedelta(days=1)
            ]
            
            return len(recent_feedback) >= self.update_frequency
            
        except Exception as e:
            logger.error(f"Error checking optimization need: {str(e)}")
            return False

    def _prepare_optimization_data(
        self,
        feedback_data: List[Dict[str, Any]],
        performance_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare data for optimization algorithms."""
        try:
            # Process feedback data
            positive_feedback = []
            negative_feedback = []
            
            for feedback in feedback_data:
                feedback_type = feedback.get('feedback_type', '')
                feedback_weight = self.feedback_config.get(feedback_type, 0)
                
                if feedback_weight > 0:
                    positive_feedback.append(feedback)
                elif feedback_weight < 0:
                    negative_feedback.append(feedback)
            
            # Calculate feedback scores
            total_positive = len(positive_feedback)
            total_negative = len(negative_feedback)
            feedback_ratio = total_positive / max(total_positive + total_negative, 1)
            
            # Prepare model performance data
            model_performance = {}
            for model_name in self.current_weights.keys():
                model_performance[model_name] = {
                    'usage_count': performance_metrics.get('model_usage_stats', {}).get(model_name, 0),
                    'accuracy_estimate': self._estimate_model_accuracy(model_name, feedback_data),
                    'latency': self._estimate_model_latency(model_name)
                }
            
            return {
                'positive_feedback': positive_feedback,
                'negative_feedback': negative_feedback,
                'feedback_ratio': feedback_ratio,
                'model_performance': model_performance,
                'total_feedback_count': len(feedback_data),
                'performance_metrics': performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Error preparing optimization data: {str(e)}")
            return {}

    async def _optimize_from_feedback(self, data: Dict[str, Any]) -> Optional[OptimizationResult]:
        """Optimize weights based on user feedback patterns."""
        try:
            if data['total_feedback_count'] < self.min_feedback_threshold:
                return None
            
            # Calculate feedback-based weight adjustments
            new_weights = self.current_weights.copy()
            adjustment_factor = self.learning_rate * data['feedback_ratio']
            
            # Increase weights for models that led to positive feedback
            for feedback in data['positive_feedback']:
                # This is a simplified approach - in practice, you'd need to track
                # which model contributed most to each successful match
                model_contribution = self._estimate_model_contribution(feedback)
                
                for model_name, contribution in model_contribution.items():
                    if model_name in new_weights:
                        new_weights[model_name] += adjustment_factor * contribution * 0.1
            
            # Decrease weights for models that led to negative feedback
            for feedback in data['negative_feedback']:
                model_contribution = self._estimate_model_contribution(feedback)
                
                for model_name, contribution in model_contribution.items():
                    if model_name in new_weights:
                        new_weights[model_name] -= adjustment_factor * contribution * 0.05
            
            # Normalize weights
            new_weights = self._normalize_weights(new_weights)
            
            # Calculate improvement score
            improvement_score = self._calculate_feedback_improvement(data, new_weights)
            
            return OptimizationResult(
                new_weights=new_weights,
                improvement_score=improvement_score,
                confidence=min(0.9, data['feedback_ratio']),
                optimization_method='feedback_based',
                metadata={
                    'positive_feedback_count': len(data['positive_feedback']),
                    'negative_feedback_count': len(data['negative_feedback']),
                    'adjustment_factor': adjustment_factor
                }
            )
            
        except Exception as e:
            logger.error(f"Feedback-based optimization failed: {str(e)}")
            return None

    async def _optimize_from_performance(self, data: Dict[str, Any]) -> Optional[OptimizationResult]:
        """Optimize weights based on model performance metrics."""
        try:
            model_performance = data['model_performance']
            if not model_performance:
                return None
            
            new_weights = self.current_weights.copy()
            
            # Calculate performance-based adjustments
            for model_name, perf_data in model_performance.items():
                if model_name in new_weights:
                    accuracy = perf_data['accuracy_estimate']
                    latency = perf_data['latency']
                    
                    # Higher accuracy = higher weight
                    accuracy_factor = (accuracy - 0.5) * 0.2  # Scale around 0.5 baseline
                    
                    # Lower latency = slightly higher weight
                    latency_factor = max(0, (2.0 - latency) * 0.05)  # Prefer sub-2s latency
                    
                    total_adjustment = accuracy_factor + latency_factor
                    new_weights[model_name] += total_adjustment * self.learning_rate
            
            # Normalize weights
            new_weights = self._normalize_weights(new_weights)
            
            # Calculate improvement score based on expected performance gain
            improvement_score = self._calculate_performance_improvement(data, new_weights)
            
            return OptimizationResult(
                new_weights=new_weights,
                improvement_score=improvement_score,
                confidence=0.7,
                optimization_method='performance_based',
                metadata={
                    'model_performance': model_performance,
                    'optimization_type': 'accuracy_latency_balance'
                }
            )
            
        except Exception as e:
            logger.error(f"Performance-based optimization failed: {str(e)}")
            return None

    async def _optimize_with_gradients(self, data: Dict[str, Any]) -> Optional[OptimizationResult]:
        """Optimize weights using gradient-based approach."""
        try:
            # Simple gradient descent on feedback objective
            current_score = self._calculate_objective_score(self.current_weights, data)
            
            new_weights = self.current_weights.copy()
            step_size = self.learning_rate * 0.1
            
            # Calculate approximate gradients
            gradients = {}
            for model_name in new_weights.keys():
                # Perturb weight slightly and measure objective change
                perturbed_weights = new_weights.copy()
                perturbed_weights[model_name] += step_size
                perturbed_weights = self._normalize_weights(perturbed_weights)
                
                perturbed_score = self._calculate_objective_score(perturbed_weights, data)
                gradients[model_name] = (perturbed_score - current_score) / step_size
            
            # Update weights in direction of positive gradient
            for model_name in new_weights.keys():
                new_weights[model_name] += self.learning_rate * gradients[model_name]
            
            # Normalize weights
            new_weights = self._normalize_weights(new_weights)
            
            # Calculate improvement
            new_score = self._calculate_objective_score(new_weights, data)
            improvement_score = new_score - current_score
            
            return OptimizationResult(
                new_weights=new_weights,
                improvement_score=improvement_score,
                confidence=0.6,
                optimization_method='gradient_descent',
                metadata={
                    'gradients': gradients,
                    'step_size': step_size,
                    'objective_improvement': improvement_score
                }
            )
            
        except Exception as e:
            logger.error(f"Gradient-based optimization failed: {str(e)}")
            return None

    def _estimate_model_contribution(self, feedback: Dict[str, Any]) -> Dict[str, float]:
        """Estimate how much each model contributed to a feedback event."""
        # This is a simplified estimation - in practice, you'd need more
        # sophisticated attribution methods
        
        # For now, assume equal contribution from all models
        num_models = len(self.current_weights)
        equal_contribution = 1.0 / num_models
        
        return {model_name: equal_contribution for model_name in self.current_weights.keys()}

    def _estimate_model_accuracy(self, model_name: str, feedback_data: List[Dict[str, Any]]) -> float:
        """Estimate accuracy for a specific model based on feedback."""
        # Simplified accuracy estimation
        positive_feedback = sum(
            1 for f in feedback_data
            if self.feedback_config.get(f.get('feedback_type', ''), 0) > 0
        )
        
        total_feedback = len(feedback_data)
        if total_feedback == 0:
            return 0.7  # Default accuracy estimate
        
        base_accuracy = positive_feedback / total_feedback
        
        # Add model-specific adjustments
        model_adjustments = {
            'primary_semantic': 0.05,    # Slightly higher accuracy
            'fast_semantic': -0.02,      # Slightly lower accuracy but faster
            'domain_specific': 0.03,     # Domain expertise bonus
            'feature_based': -0.01,      # Traditional approach penalty
            'feedback_learned': 0.02     # Learning bonus
        }
        
        adjustment = model_adjustments.get(model_name, 0)
        return min(1.0, max(0.0, base_accuracy + adjustment))

    def _estimate_model_latency(self, model_name: str) -> float:
        """Estimate latency for a specific model."""
        # Model latency estimates in seconds
        latency_estimates = {
            'primary_semantic': 1.5,     # High-quality but slower
            'fast_semantic': 0.5,        # Fast and lightweight
            'domain_specific': 1.2,      # Moderate speed
            'feature_based': 0.3,        # Very fast traditional features
            'feedback_learned': 0.8      # Neural network - moderate speed
        }
        
        return latency_estimates.get(model_name, 1.0)

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0."""
        total = sum(weights.values())
        if total <= 0:
            # If all weights are zero or negative, reset to equal weights
            num_models = len(weights)
            return {model_name: 1.0 / num_models for model_name in weights.keys()}
        
        return {model_name: weight / total for model_name, weight in weights.items()}

    def _calculate_objective_score(self, weights: Dict[str, float], data: Dict[str, Any]) -> float:
        """Calculate objective score for given weights."""
        try:
            # Combine feedback ratio and performance metrics
            feedback_score = data.get('feedback_ratio', 0.5)
            
            # Weight by model performance
            weighted_performance = 0.0
            total_weight = 0.0
            
            for model_name, weight in weights.items():
                if model_name in data.get('model_performance', {}):
                    accuracy = data['model_performance'][model_name]['accuracy_estimate']
                    weighted_performance += weight * accuracy
                    total_weight += weight
            
            if total_weight > 0:
                avg_performance = weighted_performance / total_weight
            else:
                avg_performance = 0.5
            
            # Combine scores
            objective_score = 0.6 * feedback_score + 0.4 * avg_performance
            
            return objective_score
            
        except Exception as e:
            logger.error(f"Error calculating objective score: {str(e)}")
            return 0.0

    def _calculate_feedback_improvement(self, data: Dict[str, Any], new_weights: Dict[str, float]) -> float:
        """Calculate expected improvement from feedback-based optimization."""
        # Simplified improvement calculation
        feedback_ratio = data.get('feedback_ratio', 0.5)
        weight_change = sum(
            abs(new_weights[k] - self.current_weights[k])
            for k in new_weights.keys()
        )
        
        return feedback_ratio * weight_change * 0.1

    def _calculate_performance_improvement(self, data: Dict[str, Any], new_weights: Dict[str, float]) -> float:
        """Calculate expected improvement from performance-based optimization."""
        # Estimate improvement based on weight shifts toward better-performing models
        improvement = 0.0
        
        for model_name, new_weight in new_weights.items():
            old_weight = self.current_weights.get(model_name, 0)
            weight_change = new_weight - old_weight
            
            if model_name in data.get('model_performance', {}):
                model_accuracy = data['model_performance'][model_name]['accuracy_estimate']
                improvement += weight_change * model_accuracy
        
        return abs(improvement)

    def _select_best_optimization(self, results: List[OptimizationResult]) -> Optional[OptimizationResult]:
        """Select the best optimization result from multiple strategies."""
        if not results:
            return None
        
        # Score each result based on improvement and confidence
        scored_results = []
        for result in results:
            score = result.improvement_score * result.confidence
            scored_results.append((score, result))
        
        # Sort by score and return the best
        scored_results.sort(key=lambda x: x[0], reverse=True)
        best_score, best_result = scored_results[0]
        
        # Only return if improvement is significant
        if best_score > 0.01:  # Minimum improvement threshold
            return best_result
        
        return None

    def _validate_weights(self, weights: Dict[str, float]) -> bool:
        """Validate that weights are valid and sum to approximately 1.0."""
        return MatchingWeights.validate_weights(weights)

    def _update_optimization_history(self, result: OptimizationResult):
        """Update optimization history with new result."""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'method': result.optimization_method,
            'improvement_score': result.improvement_score,
            'confidence': result.confidence,
            'weights_before': self.current_weights.copy(),
            'weights_after': result.new_weights.copy(),
            'metadata': result.metadata
        }
        
        self.optimization_history.append(history_entry)
        
        # Keep only last 100 optimization records
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]

    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics and history."""
        return {
            'current_weights': self.current_weights.copy(),
            'optimization_history': self.optimization_history[-10:],  # Last 10 optimizations
            'feedback_buffer_size': len(self.feedback_buffer),
            'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None,
            'total_optimizations': len(self.optimization_history),
            'configuration': {
                'learning_rate': self.learning_rate,
                'min_feedback_threshold': self.min_feedback_threshold,
                'update_frequency': self.update_frequency
            }
        }

    def reset_weights(self) -> Dict[str, float]:
        """Reset weights to default configuration."""
        self.current_weights = EnsembleConfig.MODEL_WEIGHTS.copy()
        logger.info("Ensemble weights reset to default configuration")
        return self.current_weights.copy()