"""
Model Variant for A/B testing framework.

This module defines model variants for comparison in A/B testing experiments,
including configuration, performance tracking, and variant management.
"""

from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import json

from app.utils.logger import logger


class VariantType(str, Enum):
    """Types of model variants for A/B testing."""
    CONTROL = "control"
    TREATMENT = "treatment"
    CHALLENGER = "challenger"
    CHAMPION = "champion"


class ModelType(str, Enum):
    """Types of models that can be tested."""
    ENSEMBLE = "ensemble"
    NEURAL_RANKING = "neural_ranking"
    SEMANTIC_MATCHING = "semantic_matching"
    FEATURE_BASED = "feature_based"
    HYBRID = "hybrid"


@dataclass
class VariantConfig:
    """Configuration for a model variant."""
    model_type: ModelType
    model_parameters: Dict[str, Any]
    feature_config: Dict[str, Any]
    preprocessing_config: Dict[str, Any]
    postprocessing_config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_type': self.model_type.value,
            'model_parameters': self.model_parameters,
            'feature_config': self.feature_config,
            'preprocessing_config': self.preprocessing_config,
            'postprocessing_config': self.postprocessing_config
        }


@dataclass
class VariantMetrics:
    """Performance metrics for a model variant."""
    accuracy_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    business_metrics: Dict[str, float]
    user_satisfaction_metrics: Dict[str, float]
    latency_metrics: Dict[str, float]
    resource_metrics: Dict[str, float]
    
    def get_composite_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted composite score."""
        if weights is None:
            weights = {
                'accuracy': 0.3,
                'performance': 0.2,
                'business': 0.25,
                'satisfaction': 0.15,
                'latency': 0.05,
                'resources': 0.05
            }
        
        try:
            composite = 0.0
            
            # Accuracy score
            accuracy_score = sum(self.accuracy_metrics.values()) / len(self.accuracy_metrics) if self.accuracy_metrics else 0.0
            composite += weights.get('accuracy', 0.0) * accuracy_score
            
            # Performance score
            performance_score = sum(self.performance_metrics.values()) / len(self.performance_metrics) if self.performance_metrics else 0.0
            composite += weights.get('performance', 0.0) * performance_score
            
            # Business score
            business_score = sum(self.business_metrics.values()) / len(self.business_metrics) if self.business_metrics else 0.0
            composite += weights.get('business', 0.0) * business_score
            
            # User satisfaction score
            satisfaction_score = sum(self.user_satisfaction_metrics.values()) / len(self.user_satisfaction_metrics) if self.user_satisfaction_metrics else 0.0
            composite += weights.get('satisfaction', 0.0) * satisfaction_score
            
            # Latency score (inverted - lower is better)
            if self.latency_metrics:
                avg_latency = sum(self.latency_metrics.values()) / len(self.latency_metrics)
                latency_score = max(0.0, 1.0 - (avg_latency / 10000))  # Normalize assuming 10s max
            else:
                latency_score = 1.0
            composite += weights.get('latency', 0.0) * latency_score
            
            # Resource score (inverted - lower usage is better)
            if self.resource_metrics:
                avg_resource = sum(self.resource_metrics.values()) / len(self.resource_metrics)
                resource_score = max(0.0, 1.0 - (avg_resource / 100))  # Normalize assuming 100% max
            else:
                resource_score = 1.0
            composite += weights.get('resources', 0.0) * resource_score
            
            return composite
            
        except Exception as e:
            logger.error(f"Composite score calculation failed: {str(e)}")
            return 0.0


class ModelVariant:
    """
    Represents a model variant for A/B testing with configuration,
    metrics tracking, and prediction capabilities.
    """
    
    def __init__(
        self,
        variant_id: str,
        name: str,
        description: str,
        variant_type: VariantType,
        config: VariantConfig,
        prediction_function: Optional[Callable] = None
    ):
        """
        Initialize a model variant.
        
        Args:
            variant_id: Unique identifier for the variant
            name: Human-readable name
            description: Description of the variant
            variant_type: Type of variant (control, treatment, etc.)
            config: Variant configuration
            prediction_function: Function to generate predictions
        """
        self.variant_id = variant_id
        self.name = name
        self.description = description
        self.variant_type = variant_type
        self.config = config
        self.prediction_function = prediction_function
        
        # Performance tracking
        self.metrics_history: List[VariantMetrics] = []
        self.prediction_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        
        # State tracking
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.is_active = True
        
        logger.info(f"Created model variant: {name} ({variant_id})")
    
    async def predict(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate predictions using this variant.
        
        Args:
            input_data: Input data for prediction
            context: Additional context information
            
        Returns:
            Prediction results with metadata
        """
        try:
            start_time = datetime.now()
            
            # Check if variant is active
            if not self.is_active:
                raise ValueError(f"Variant {self.variant_id} is not active")
            
            # Use prediction function if available
            if self.prediction_function:
                prediction_result = await self.prediction_function(input_data, context, self.config)
            else:
                # Fallback prediction logic
                prediction_result = await self._fallback_prediction(input_data, context)
            
            # Calculate latency
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update metrics
            self.prediction_count += 1
            self.total_latency += latency_ms
            
            # Add variant metadata to result
            prediction_result['variant_metadata'] = {
                'variant_id': self.variant_id,
                'variant_name': self.name,
                'prediction_latency_ms': latency_ms,
                'prediction_timestamp': end_time.isoformat(),
                'model_type': self.config.model_type.value
            }
            
            logger.debug(f"Variant {self.variant_id} prediction completed in {latency_ms:.2f}ms")
            
            return prediction_result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Prediction failed for variant {self.variant_id}: {str(e)}")
            
            # Return error result
            return {
                'error': str(e),
                'variant_metadata': {
                    'variant_id': self.variant_id,
                    'variant_name': self.name,
                    'error': True,
                    'error_timestamp': datetime.now().isoformat()
                }
            }
    
    async def _fallback_prediction(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fallback prediction when no prediction function is provided."""
        logger.warning(f"Using fallback prediction for variant {self.variant_id}")
        
        # Simple fallback based on variant type
        if self.variant_type == VariantType.CONTROL:
            # Control variant returns baseline scores
            return {
                'scores': [0.5, 0.4, 0.3, 0.2, 0.1],
                'confidence': 0.6,
                'explanation': 'Control variant baseline prediction'
            }
        else:
            # Treatment variants return slightly modified scores
            return {
                'scores': [0.6, 0.5, 0.4, 0.3, 0.2],
                'confidence': 0.7,
                'explanation': 'Treatment variant enhanced prediction'
            }
    
    def update_metrics(self, metrics: VariantMetrics) -> None:
        """Update variant performance metrics."""
        try:
            self.metrics_history.append(metrics)
            self.last_updated = datetime.now()
            
            # Keep only last 100 metric entries
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
            
            logger.debug(f"Updated metrics for variant {self.variant_id}")
            
        except Exception as e:
            logger.error(f"Failed to update metrics for variant {self.variant_id}: {str(e)}")
    
    def get_current_metrics(self) -> Optional[VariantMetrics]:
        """Get the most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_average_metrics(self, last_n: int = 10) -> Optional[VariantMetrics]:
        """Get average metrics over the last N entries."""
        try:
            if not self.metrics_history:
                return None
            
            recent_metrics = self.metrics_history[-last_n:]
            
            # Calculate averages
            avg_accuracy = {}
            avg_performance = {}
            avg_business = {}
            avg_satisfaction = {}
            avg_latency = {}
            avg_resources = {}
            
            for metrics in recent_metrics:
                # Average accuracy metrics
                for key, value in metrics.accuracy_metrics.items():
                    avg_accuracy[key] = avg_accuracy.get(key, 0.0) + value
                
                # Average performance metrics
                for key, value in metrics.performance_metrics.items():
                    avg_performance[key] = avg_performance.get(key, 0.0) + value
                
                # Average business metrics
                for key, value in metrics.business_metrics.items():
                    avg_business[key] = avg_business.get(key, 0.0) + value
                
                # Average satisfaction metrics
                for key, value in metrics.user_satisfaction_metrics.items():
                    avg_satisfaction[key] = avg_satisfaction.get(key, 0.0) + value
                
                # Average latency metrics
                for key, value in metrics.latency_metrics.items():
                    avg_latency[key] = avg_latency.get(key, 0.0) + value
                
                # Average resource metrics
                for key, value in metrics.resource_metrics.items():
                    avg_resources[key] = avg_resources.get(key, 0.0) + value
            
            # Divide by number of entries
            count = len(recent_metrics)
            
            for key in avg_accuracy:
                avg_accuracy[key] /= count
            for key in avg_performance:
                avg_performance[key] /= count
            for key in avg_business:
                avg_business[key] /= count
            for key in avg_satisfaction:
                avg_satisfaction[key] /= count
            for key in avg_latency:
                avg_latency[key] /= count
            for key in avg_resources:
                avg_resources[key] /= count
            
            return VariantMetrics(
                accuracy_metrics=avg_accuracy,
                performance_metrics=avg_performance,
                business_metrics=avg_business,
                user_satisfaction_metrics=avg_satisfaction,
                latency_metrics=avg_latency,
                resource_metrics=avg_resources
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate average metrics: {str(e)}")
            return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary."""
        try:
            current_metrics = self.get_current_metrics()
            avg_metrics = self.get_average_metrics()
            
            # Calculate derived metrics
            avg_latency = self.total_latency / self.prediction_count if self.prediction_count > 0 else 0.0
            error_rate = self.error_count / max(self.prediction_count + self.error_count, 1)
            
            return {
                'variant_info': {
                    'variant_id': self.variant_id,
                    'name': self.name,
                    'type': self.variant_type.value,
                    'model_type': self.config.model_type.value,
                    'created_at': self.created_at.isoformat(),
                    'is_active': self.is_active
                },
                'performance_stats': {
                    'prediction_count': self.prediction_count,
                    'error_count': self.error_count,
                    'error_rate': error_rate,
                    'average_latency_ms': avg_latency,
                    'metrics_history_length': len(self.metrics_history)
                },
                'current_metrics': current_metrics.__dict__ if current_metrics else None,
                'average_metrics': avg_metrics.__dict__ if avg_metrics else None,
                'composite_scores': {
                    'current': current_metrics.get_composite_score() if current_metrics else 0.0,
                    'average': avg_metrics.get_composite_score() if avg_metrics else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate performance summary: {str(e)}")
            return {}
    
    def compare_with(self, other_variant: 'ModelVariant') -> Dict[str, Any]:
        """Compare this variant with another variant."""
        try:
            self_metrics = self.get_average_metrics()
            other_metrics = other_variant.get_average_metrics()
            
            if not self_metrics or not other_metrics:
                return {
                    'comparison_available': False,
                    'reason': 'Insufficient metrics data for comparison'
                }
            
            # Calculate differences
            accuracy_diff = {}
            performance_diff = {}
            
            # Compare accuracy metrics
            for metric in set(self_metrics.accuracy_metrics.keys()).union(other_metrics.accuracy_metrics.keys()):
                self_value = self_metrics.accuracy_metrics.get(metric, 0.0)
                other_value = other_metrics.accuracy_metrics.get(metric, 0.0)
                accuracy_diff[metric] = self_value - other_value
            
            # Compare performance metrics
            for metric in set(self_metrics.performance_metrics.keys()).union(other_metrics.performance_metrics.keys()):
                self_value = self_metrics.performance_metrics.get(metric, 0.0)
                other_value = other_metrics.performance_metrics.get(metric, 0.0)
                performance_diff[metric] = self_value - other_value
            
            # Compare composite scores
            self_composite = self_metrics.get_composite_score()
            other_composite = other_metrics.get_composite_score()
            composite_diff = self_composite - other_composite
            
            # Determine winner
            winner = self.variant_id if composite_diff > 0 else other_variant.variant_id
            
            return {
                'comparison_available': True,
                'self_variant': self.variant_id,
                'other_variant': other_variant.variant_id,
                'accuracy_differences': accuracy_diff,
                'performance_differences': performance_diff,
                'composite_score_difference': composite_diff,
                'winner': winner,
                'comparison_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Variant comparison failed: {str(e)}")
            return {
                'comparison_available': False,
                'error': str(e)
            }
    
    def update_config(self, new_config: VariantConfig) -> bool:
        """Update variant configuration."""
        try:
            self.config = new_config
            self.last_updated = datetime.now()
            
            logger.info(f"Updated configuration for variant {self.variant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update config for variant {self.variant_id}: {str(e)}")
            return False
    
    def activate(self) -> None:
        """Activate the variant."""
        self.is_active = True
        self.last_updated = datetime.now()
        logger.info(f"Activated variant {self.variant_id}")
    
    def deactivate(self) -> None:
        """Deactivate the variant."""
        self.is_active = False
        self.last_updated = datetime.now()
        logger.info(f"Deactivated variant {self.variant_id}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert variant to dictionary for serialization."""
        try:
            return {
                'variant_id': self.variant_id,
                'name': self.name,
                'description': self.description,
                'variant_type': self.variant_type.value,
                'config': self.config.to_dict(),
                'performance_stats': {
                    'prediction_count': self.prediction_count,
                    'error_count': self.error_count,
                    'total_latency': self.total_latency,
                    'metrics_history_length': len(self.metrics_history)
                },
                'created_at': self.created_at.isoformat(),
                'last_updated': self.last_updated.isoformat(),
                'is_active': self.is_active
            }
            
        except Exception as e:
            logger.error(f"Failed to serialize variant {self.variant_id}: {str(e)}")
            return {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVariant':
        """Create variant from dictionary."""
        try:
            config_data = data['config']
            config = VariantConfig(
                model_type=ModelType(config_data['model_type']),
                model_parameters=config_data['model_parameters'],
                feature_config=config_data['feature_config'],
                preprocessing_config=config_data['preprocessing_config'],
                postprocessing_config=config_data['postprocessing_config']
            )
            
            variant = cls(
                variant_id=data['variant_id'],
                name=data['name'],
                description=data['description'],
                variant_type=VariantType(data['variant_type']),
                config=config
            )
            
            # Restore state
            variant.prediction_count = data['performance_stats']['prediction_count']
            variant.error_count = data['performance_stats']['error_count']
            variant.total_latency = data['performance_stats']['total_latency']
            variant.created_at = datetime.fromisoformat(data['created_at'])
            variant.last_updated = datetime.fromisoformat(data['last_updated'])
            variant.is_active = data['is_active']
            
            return variant
            
        except Exception as e:
            logger.error(f"Failed to deserialize variant: {str(e)}")
            raise


# Helper functions for creating common variant types

def create_ensemble_variant(
    variant_id: str,
    name: str,
    variant_type: VariantType,
    model_weights: Dict[str, float],
    prediction_function: Optional[Callable] = None
) -> ModelVariant:
    """Create an ensemble model variant."""
    config = VariantConfig(
        model_type=ModelType.ENSEMBLE,
        model_parameters={'weights': model_weights},
        feature_config={'use_all_features': True},
        preprocessing_config={'normalize': True},
        postprocessing_config={'apply_calibration': True}
    )
    
    return ModelVariant(
        variant_id=variant_id,
        name=name,
        description=f"Ensemble variant with weights: {model_weights}",
        variant_type=variant_type,
        config=config,
        prediction_function=prediction_function
    )


def create_ranking_variant(
    variant_id: str,
    name: str,
    variant_type: VariantType,
    ranking_algorithm: str,
    prediction_function: Optional[Callable] = None
) -> ModelVariant:
    """Create a neural ranking model variant."""
    config = VariantConfig(
        model_type=ModelType.NEURAL_RANKING,
        model_parameters={'algorithm': ranking_algorithm},
        feature_config={'ranking_features': True},
        preprocessing_config={'feature_scaling': True},
        postprocessing_config={'rank_normalization': True}
    )
    
    return ModelVariant(
        variant_id=variant_id,
        name=name,
        description=f"Neural ranking variant using {ranking_algorithm}",
        variant_type=variant_type,
        config=config,
        prediction_function=prediction_function
    )