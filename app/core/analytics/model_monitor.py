"""
Model Performance Monitor for tracking ML model performance and drift detection.

This module provides comprehensive model monitoring including performance tracking,
data drift detection, model degradation alerts, and automated retraining triggers.
"""

import asyncio
import numpy as np
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
from scipy import stats
import pickle
import hashlib

from app.utils.logger import logger


class ModelType(str, Enum):
    """Types of models being monitored."""
    ENSEMBLE = "ensemble"
    NEURAL_RANKING = "neural_ranking"
    SEMANTIC_MATCHING = "semantic_matching"
    FEATURE_MATCHING = "feature_matching"
    HYBRID = "hybrid"


class DriftType(str, Enum):
    """Types of drift detection."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ModelPrediction:
    """Model prediction record."""
    model_id: str
    model_type: ModelType
    prediction_id: str
    input_features: Dict[str, Any]
    prediction: Any
    confidence: float
    timestamp: datetime
    actual_outcome: Optional[Any] = None
    feedback_score: Optional[float] = None
    processing_time_ms: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    """Model performance metrics."""
    model_id: str
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    ndcg_10: float
    mrr: float
    average_confidence: float
    prediction_count: int
    error_rate: float
    latency_p50: float
    latency_p95: float
    latency_p99: float


@dataclass
class DriftDetection:
    """Drift detection result."""
    model_id: str
    drift_type: DriftType
    severity: AlertSeverity
    drift_score: float
    threshold: float
    timestamp: datetime
    affected_features: List[str]
    details: Dict[str, Any]
    recommendations: List[str]


@dataclass
class ModelAlert:
    """Model performance alert."""
    alert_id: str
    model_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    data: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False


class ModelPerformanceMonitor:
    """
    Comprehensive model performance monitoring system with drift detection,
    performance tracking, and automated alerting capabilities.
    """
    
    def __init__(
        self,
        monitoring_interval_seconds: int = 300,
        drift_detection_window: int = 1000,
        performance_window_hours: int = 24,
        enable_auto_retraining: bool = True
    ):
        """Initialize model performance monitor."""
        self.monitoring_interval_seconds = monitoring_interval_seconds
        self.drift_detection_window = drift_detection_window
        self.performance_window_hours = performance_window_hours
        self.enable_auto_retraining = enable_auto_retraining
        
        # Model tracking
        self.registered_models: Dict[str, Dict[str, Any]] = {}
        self.model_predictions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.baseline_distributions: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_history: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.current_performance: Dict[str, PerformanceMetrics] = {}
        
        # Drift detection
        self.drift_detectors: Dict[str, Dict[str, Any]] = {}
        self.drift_history: List[DriftDetection] = []
        self.active_drifts: Dict[str, DriftDetection] = {}
        
        # Alerts and notifications
        self.alerts: List[ModelAlert] = []
        self.active_alerts: Dict[str, ModelAlert] = {}
        self.alert_callbacks: List[Callable] = []
        
        # Monitoring tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.drift_detection_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.thresholds = {
            'accuracy_degradation': 0.05,       # 5% accuracy drop
            'latency_increase': 2.0,             # 2x latency increase
            'error_rate_increase': 0.1,          # 10% error rate increase
            'confidence_drop': 0.1,              # 10% confidence drop
            'drift_threshold': 0.1,              # Statistical drift threshold
            'performance_degradation': 0.1       # 10% performance degradation
        }
        
        # Statistics
        self.monitor_stats = {
            'total_predictions_monitored': 0,
            'models_monitored': 0,
            'drift_detections': 0,
            'alerts_triggered': 0,
            'retraining_triggered': 0,
            'monitoring_cycles': 0
        }
        
        logger.info("ModelPerformanceMonitor initialized")
    
    async def initialize(self):
        """Initialize monitoring tasks."""
        try:
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Start drift detection task
            self.drift_detection_task = asyncio.create_task(self._drift_detection_loop())
            
            logger.info("ModelPerformanceMonitor started")
            
        except Exception as e:
            logger.error(f"ModelPerformanceMonitor initialization failed: {str(e)}")
            raise
    
    async def register_model(
        self,
        model_id: str,
        model_type: ModelType,
        baseline_data: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a model for monitoring.
        
        Args:
            model_id: Unique model identifier
            model_type: Type of model
            baseline_data: Baseline data for drift detection
            metadata: Additional model metadata
            
        Returns:
            True if registered successfully
        """
        try:
            self.registered_models[model_id] = {
                'model_type': model_type,
                'registered_at': datetime.now(),
                'metadata': metadata or {},
                'prediction_count': 0,
                'last_performance_check': None,
                'baseline_established': False
            }
            
            # Establish baseline if data provided
            if baseline_data:
                await self._establish_baseline(model_id, baseline_data)
            
            self.monitor_stats['models_monitored'] += 1
            
            logger.info(f"Registered model for monitoring: {model_id} ({model_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {str(e)}")
            return False
    
    async def track_prediction(
        self,
        model_id: str,
        prediction_data: ModelPrediction
    ) -> bool:
        """
        Track a model prediction for monitoring.
        
        Args:
            model_id: Model identifier
            prediction_data: Prediction data to track
            
        Returns:
            True if tracked successfully
        """
        try:
            if model_id not in self.registered_models:
                logger.warning(f"Model {model_id} not registered for monitoring")
                return False
            
            # Store prediction
            self.model_predictions[model_id].append(prediction_data)
            
            # Update model stats
            self.registered_models[model_id]['prediction_count'] += 1
            self.monitor_stats['total_predictions_monitored'] += 1
            
            # Real-time drift detection if enabled
            if len(self.model_predictions[model_id]) % 100 == 0:  # Check every 100 predictions
                await self._check_real_time_drift(model_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to track prediction for model {model_id}: {str(e)}")
            return False
    
    async def update_prediction_outcome(
        self,
        model_id: str,
        prediction_id: str,
        actual_outcome: Any,
        feedback_score: Optional[float] = None
    ) -> bool:
        """
        Update prediction with actual outcome for performance calculation.
        
        Args:
            model_id: Model identifier
            prediction_id: Prediction identifier
            actual_outcome: Actual outcome
            feedback_score: Optional feedback score
            
        Returns:
            True if updated successfully
        """
        try:
            if model_id not in self.model_predictions:
                return False
            
            # Find and update prediction
            predictions = self.model_predictions[model_id]
            for prediction in reversed(predictions):  # Search from most recent
                if prediction.prediction_id == prediction_id:
                    prediction.actual_outcome = actual_outcome
                    prediction.feedback_score = feedback_score
                    
                    logger.debug(f"Updated prediction outcome for {prediction_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update prediction outcome: {str(e)}")
            return False
    
    async def _establish_baseline(
        self,
        model_id: str,
        baseline_data: List[Dict[str, Any]]
    ):
        """Establish baseline distributions for drift detection."""
        try:
            if not baseline_data:
                return
            
            baseline_stats = {}
            
            # Calculate baseline statistics for each feature
            feature_values = defaultdict(list)
            
            for data_point in baseline_data:
                for feature, value in data_point.items():
                    if isinstance(value, (int, float)):
                        feature_values[feature].append(value)
            
            # Calculate statistics for each feature
            for feature, values in feature_values.items():
                if len(values) > 0:
                    baseline_stats[feature] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'percentiles': {
                            '25': np.percentile(values, 25),
                            '50': np.percentile(values, 50),
                            '75': np.percentile(values, 75),
                            '95': np.percentile(values, 95)
                        },
                        'distribution': np.histogram(values, bins=20)[0].tolist(),
                        'sample_size': len(values)
                    }
            
            self.baseline_distributions[model_id] = baseline_stats
            self.registered_models[model_id]['baseline_established'] = True
            
            logger.info(f"Established baseline for model {model_id} with {len(baseline_data)} data points")
            
        except Exception as e:
            logger.error(f"Failed to establish baseline for model {model_id}: {str(e)}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        try:
            while True:
                await asyncio.sleep(self.monitoring_interval_seconds)
                
                try:
                    await self._perform_monitoring_cycle()
                    self.monitor_stats['monitoring_cycles'] += 1
                    
                except Exception as e:
                    logger.error(f"Monitoring cycle error: {str(e)}")
                
        except asyncio.CancelledError:
            logger.info("Model monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Monitoring loop error: {str(e)}")
    
    async def _perform_monitoring_cycle(self):
        """Perform a complete monitoring cycle."""
        try:
            for model_id in self.registered_models.keys():
                # Calculate performance metrics
                await self._calculate_model_performance(model_id)
                
                # Check for performance degradation
                await self._check_performance_degradation(model_id)
                
                # Update model status
                self.registered_models[model_id]['last_performance_check'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Monitoring cycle failed: {str(e)}")
    
    async def _calculate_model_performance(self, model_id: str):
        """Calculate current performance metrics for a model."""
        try:
            predictions = list(self.model_predictions[model_id])
            
            if not predictions:
                return
            
            # Filter predictions with outcomes for performance calculation
            labeled_predictions = [
                p for p in predictions
                if p.actual_outcome is not None
            ]
            
            if len(labeled_predictions) < 10:  # Need minimum predictions
                return
            
            # Calculate metrics based on model type
            model_type = self.registered_models[model_id]['model_type']
            
            if model_type in [ModelType.ENSEMBLE, ModelType.HYBRID]:
                metrics = await self._calculate_ranking_metrics(labeled_predictions)
            else:
                metrics = await self._calculate_classification_metrics(labeled_predictions)
            
            # Calculate latency metrics
            latencies = [p.processing_time_ms for p in predictions if p.processing_time_ms > 0]
            if latencies:
                metrics.latency_p50 = np.percentile(latencies, 50)
                metrics.latency_p95 = np.percentile(latencies, 95)
                metrics.latency_p99 = np.percentile(latencies, 99)
            
            # Store performance metrics
            self.performance_history[model_id].append(metrics)
            self.current_performance[model_id] = metrics
            
            # Keep only recent history
            if len(self.performance_history[model_id]) > 1000:
                self.performance_history[model_id] = self.performance_history[model_id][-1000:]
            
        except Exception as e:
            logger.error(f"Performance calculation failed for model {model_id}: {str(e)}")
    
    async def _calculate_ranking_metrics(
        self,
        predictions: List[ModelPrediction]
    ) -> PerformanceMetrics:
        """Calculate ranking-specific metrics."""
        try:
            # Extract ranking data
            confidences = [p.confidence for p in predictions]
            feedback_scores = [p.feedback_score for p in predictions if p.feedback_score is not None]
            
            # Calculate basic metrics
            avg_confidence = np.mean(confidences) if confidences else 0.0
            prediction_count = len(predictions)
            
            # Calculate NDCG@10 and MRR if feedback available
            ndcg_10 = 0.0
            mrr = 0.0
            
            if feedback_scores:
                # Simplified NDCG calculation
                # In practice, this would use proper ranking evaluation
                ndcg_10 = np.mean(feedback_scores)
                mrr = np.mean([1.0 / (i + 1) for i, score in enumerate(feedback_scores) if score > 0.5])
            
            # Error rate calculation
            error_predictions = [p for p in predictions if p.feedback_score is not None and p.feedback_score < 0.3]
            error_rate = len(error_predictions) / prediction_count if prediction_count > 0 else 0.0
            
            return PerformanceMetrics(
                model_id=predictions[0].model_id,
                timestamp=datetime.now(),
                accuracy=ndcg_10,  # Use NDCG as accuracy proxy
                precision=0.0,     # Not applicable for ranking
                recall=0.0,        # Not applicable for ranking
                f1_score=0.0,      # Not applicable for ranking
                auc_roc=0.0,       # Not applicable for ranking
                ndcg_10=ndcg_10,
                mrr=mrr,
                average_confidence=avg_confidence,
                prediction_count=prediction_count,
                error_rate=error_rate,
                latency_p50=0.0,   # Will be filled by caller
                latency_p95=0.0,   # Will be filled by caller
                latency_p99=0.0    # Will be filled by caller
            )
            
        except Exception as e:
            logger.error(f"Ranking metrics calculation failed: {str(e)}")
            raise
    
    async def _calculate_classification_metrics(
        self,
        predictions: List[ModelPrediction]
    ) -> PerformanceMetrics:
        """Calculate classification metrics."""
        try:
            # This is a simplified implementation
            # In practice, you'd implement proper classification metrics
            
            confidences = [p.confidence for p in predictions]
            feedback_scores = [p.feedback_score for p in predictions if p.feedback_score is not None]
            
            avg_confidence = np.mean(confidences) if confidences else 0.0
            prediction_count = len(predictions)
            
            # Simplified accuracy calculation
            accuracy = np.mean(feedback_scores) if feedback_scores else 0.0
            
            return PerformanceMetrics(
                model_id=predictions[0].model_id,
                timestamp=datetime.now(),
                accuracy=accuracy,
                precision=accuracy,  # Simplified
                recall=accuracy,     # Simplified
                f1_score=accuracy,   # Simplified
                auc_roc=accuracy,    # Simplified
                ndcg_10=0.0,
                mrr=0.0,
                average_confidence=avg_confidence,
                prediction_count=prediction_count,
                error_rate=1.0 - accuracy,
                latency_p50=0.0,
                latency_p95=0.0,
                latency_p99=0.0
            )
            
        except Exception as e:
            logger.error(f"Classification metrics calculation failed: {str(e)}")
            raise
    
    async def _check_performance_degradation(self, model_id: str):
        """Check for performance degradation and trigger alerts."""
        try:
            if model_id not in self.current_performance:
                return
            
            current_metrics = self.current_performance[model_id]
            history = self.performance_history[model_id]
            
            if len(history) < 5:  # Need sufficient history
                return
            
            # Calculate baseline performance (average of last 5 measurements before current)
            baseline_metrics = history[-6:-1]  # Exclude current measurement
            baseline_accuracy = np.mean([m.accuracy for m in baseline_metrics])
            baseline_latency = np.mean([m.latency_p95 for m in baseline_metrics])
            baseline_error_rate = np.mean([m.error_rate for m in baseline_metrics])
            
            alerts_to_create = []
            
            # Check accuracy degradation
            accuracy_drop = baseline_accuracy - current_metrics.accuracy
            if accuracy_drop > self.thresholds['accuracy_degradation']:
                alerts_to_create.append({
                    'type': 'accuracy_degradation',
                    'severity': AlertSeverity.HIGH if accuracy_drop > 0.1 else AlertSeverity.MEDIUM,
                    'message': f"Accuracy dropped by {accuracy_drop:.3f} from baseline {baseline_accuracy:.3f}",
                    'data': {
                        'current_accuracy': current_metrics.accuracy,
                        'baseline_accuracy': baseline_accuracy,
                        'drop_amount': accuracy_drop
                    }
                })
            
            # Check latency increase
            if baseline_latency > 0:
                latency_ratio = current_metrics.latency_p95 / baseline_latency
                if latency_ratio > self.thresholds['latency_increase']:
                    alerts_to_create.append({
                        'type': 'latency_increase',
                        'severity': AlertSeverity.HIGH if latency_ratio > 3.0 else AlertSeverity.MEDIUM,
                        'message': f"Latency increased by {latency_ratio:.2f}x from baseline {baseline_latency:.2f}ms",
                        'data': {
                            'current_latency': current_metrics.latency_p95,
                            'baseline_latency': baseline_latency,
                            'ratio': latency_ratio
                        }
                    })
            
            # Check error rate increase
            error_rate_increase = current_metrics.error_rate - baseline_error_rate
            if error_rate_increase > self.thresholds['error_rate_increase']:
                alerts_to_create.append({
                    'type': 'error_rate_increase',
                    'severity': AlertSeverity.CRITICAL if error_rate_increase > 0.2 else AlertSeverity.HIGH,
                    'message': f"Error rate increased by {error_rate_increase:.3f} from baseline {baseline_error_rate:.3f}",
                    'data': {
                        'current_error_rate': current_metrics.error_rate,
                        'baseline_error_rate': baseline_error_rate,
                        'increase_amount': error_rate_increase
                    }
                })
            
            # Create alerts
            for alert_data in alerts_to_create:
                await self._create_alert(model_id, alert_data)
            
        except Exception as e:
            logger.error(f"Performance degradation check failed for model {model_id}: {str(e)}")
    
    async def _drift_detection_loop(self):
        """Background drift detection loop."""
        try:
            while True:
                await asyncio.sleep(600)  # Check drift every 10 minutes
                
                try:
                    for model_id in self.registered_models.keys():
                        await self._detect_drift(model_id)
                        
                except Exception as e:
                    logger.error(f"Drift detection cycle error: {str(e)}")
                
        except asyncio.CancelledError:
            logger.info("Drift detection loop cancelled")
        except Exception as e:
            logger.error(f"Drift detection loop error: {str(e)}")
    
    async def _check_real_time_drift(self, model_id: str):
        """Check for drift in real-time."""
        try:
            if model_id not in self.baseline_distributions:
                return
            
            recent_predictions = list(self.model_predictions[model_id])[-self.drift_detection_window:]
            
            if len(recent_predictions) < 100:
                return
            
            # Extract feature values from recent predictions
            recent_features = defaultdict(list)
            for prediction in recent_predictions:
                for feature, value in prediction.input_features.items():
                    if isinstance(value, (int, float)):
                        recent_features[feature].append(value)
            
            # Check for drift in each feature
            drift_detected = False
            affected_features = []
            
            baseline = self.baseline_distributions[model_id]
            
            for feature, recent_values in recent_features.items():
                if feature in baseline and len(recent_values) >= 30:
                    drift_score = await self._calculate_drift_score(
                        baseline[feature], recent_values
                    )
                    
                    if drift_score > self.thresholds['drift_threshold']:
                        drift_detected = True
                        affected_features.append(feature)
            
            if drift_detected:
                await self._create_drift_alert(model_id, DriftType.DATA_DRIFT, affected_features)
            
        except Exception as e:
            logger.error(f"Real-time drift detection failed for model {model_id}: {str(e)}")
    
    async def _detect_drift(self, model_id: str):
        """Comprehensive drift detection for a model."""
        try:
            if not self.registered_models[model_id]['baseline_established']:
                return
            
            # Get recent predictions
            recent_predictions = list(self.model_predictions[model_id])[-self.drift_detection_window:]
            
            if len(recent_predictions) < 100:
                return
            
            # Data drift detection
            await self._detect_data_drift(model_id, recent_predictions)
            
            # Concept drift detection
            await self._detect_concept_drift(model_id, recent_predictions)
            
            # Prediction drift detection
            await self._detect_prediction_drift(model_id, recent_predictions)
            
        except Exception as e:
            logger.error(f"Drift detection failed for model {model_id}: {str(e)}")
    
    async def _detect_data_drift(self, model_id: str, predictions: List[ModelPrediction]):
        """Detect data drift in input features."""
        try:
            if model_id not in self.baseline_distributions:
                return
            
            # Extract feature values
            feature_values = defaultdict(list)
            for prediction in predictions:
                for feature, value in prediction.input_features.items():
                    if isinstance(value, (int, float)):
                        feature_values[feature].append(value)
            
            baseline = self.baseline_distributions[model_id]
            drift_scores = {}
            affected_features = []
            
            # Check each feature for drift
            for feature, values in feature_values.items():
                if feature in baseline and len(values) >= 30:
                    drift_score = await self._calculate_drift_score(baseline[feature], values)
                    drift_scores[feature] = drift_score
                    
                    if drift_score > self.thresholds['drift_threshold']:
                        affected_features.append(feature)
            
            if affected_features:
                max_drift_score = max(drift_scores[f] for f in affected_features)
                severity = self._determine_drift_severity(max_drift_score)
                
                drift_detection = DriftDetection(
                    model_id=model_id,
                    drift_type=DriftType.DATA_DRIFT,
                    severity=severity,
                    drift_score=max_drift_score,
                    threshold=self.thresholds['drift_threshold'],
                    timestamp=datetime.now(),
                    affected_features=affected_features,
                    details={'feature_drift_scores': drift_scores},
                    recommendations=[
                        f"Investigate data sources for features: {', '.join(affected_features)}",
                        "Consider retraining model with recent data",
                        "Review data preprocessing pipeline"
                    ]
                )
                
                await self._handle_drift_detection(drift_detection)
            
        except Exception as e:
            logger.error(f"Data drift detection failed: {str(e)}")
    
    async def _detect_concept_drift(self, model_id: str, predictions: List[ModelPrediction]):
        """Detect concept drift in model predictions vs outcomes."""
        try:
            # Get predictions with outcomes
            labeled_predictions = [
                p for p in predictions
                if p.actual_outcome is not None and p.feedback_score is not None
            ]
            
            if len(labeled_predictions) < 50:
                return
            
            # Split into time windows
            mid_point = len(labeled_predictions) // 2
            early_predictions = labeled_predictions[:mid_point]
            recent_predictions = labeled_predictions[mid_point:]
            
            # Calculate performance for each window
            early_performance = np.mean([p.feedback_score for p in early_predictions])
            recent_performance = np.mean([p.feedback_score for p in recent_predictions])
            
            # Check for significant performance change
            performance_change = abs(early_performance - recent_performance)
            
            if performance_change > self.thresholds['performance_degradation']:
                severity = self._determine_drift_severity(performance_change * 10)  # Scale for severity
                
                drift_detection = DriftDetection(
                    model_id=model_id,
                    drift_type=DriftType.CONCEPT_DRIFT,
                    severity=severity,
                    drift_score=performance_change,
                    threshold=self.thresholds['performance_degradation'],
                    timestamp=datetime.now(),
                    affected_features=[],
                    details={
                        'early_performance': early_performance,
                        'recent_performance': recent_performance,
                        'performance_change': performance_change
                    },
                    recommendations=[
                        "Investigate changes in target distribution",
                        "Consider retraining with recent feedback data",
                        "Review business logic and success metrics"
                    ]
                )
                
                await self._handle_drift_detection(drift_detection)
            
        except Exception as e:
            logger.error(f"Concept drift detection failed: {str(e)}")
    
    async def _detect_prediction_drift(self, model_id: str, predictions: List[ModelPrediction]):
        """Detect drift in prediction distributions."""
        try:
            if len(predictions) < 100:
                return
            
            # Split predictions into two time windows
            mid_point = len(predictions) // 2
            early_predictions = predictions[:mid_point]
            recent_predictions = predictions[mid_point:]
            
            # Extract prediction values
            early_values = [p.confidence for p in early_predictions]
            recent_values = [p.confidence for p in recent_predictions]
            
            # Perform statistical test for distribution change
            statistic, p_value = stats.ks_2samp(early_values, recent_values)
            
            if p_value < 0.05:  # Significant difference
                severity = self._determine_drift_severity(statistic * 10)
                
                drift_detection = DriftDetection(
                    model_id=model_id,
                    drift_type=DriftType.PREDICTION_DRIFT,
                    severity=severity,
                    drift_score=statistic,
                    threshold=0.05,
                    timestamp=datetime.now(),
                    affected_features=[],
                    details={
                        'ks_statistic': statistic,
                        'p_value': p_value,
                        'early_mean_confidence': np.mean(early_values),
                        'recent_mean_confidence': np.mean(recent_values)
                    },
                    recommendations=[
                        "Investigate changes in model behavior",
                        "Check for model version changes",
                        "Review confidence calibration"
                    ]
                )
                
                await self._handle_drift_detection(drift_detection)
            
        except Exception as e:
            logger.error(f"Prediction drift detection failed: {str(e)}")
    
    async def _calculate_drift_score(
        self,
        baseline_stats: Dict[str, Any],
        recent_values: List[float]
    ) -> float:
        """Calculate drift score using statistical tests."""
        try:
            if len(recent_values) < 10:
                return 0.0
            
            baseline_mean = baseline_stats['mean']
            baseline_std = baseline_stats['std']
            
            recent_mean = np.mean(recent_values)
            recent_std = np.std(recent_values)
            
            # Calculate multiple drift indicators
            
            # 1. Mean shift (standardized)
            if baseline_std > 0:
                mean_shift = abs(recent_mean - baseline_mean) / baseline_std
            else:
                mean_shift = 0.0
            
            # 2. Variance change
            if baseline_std > 0:
                variance_ratio = recent_std / baseline_std
                variance_change = abs(1.0 - variance_ratio)
            else:
                variance_change = 0.0
            
            # 3. Distribution comparison using Earth Mover's Distance (simplified)
            baseline_dist = np.array(baseline_stats['distribution'])
            recent_hist, _ = np.histogram(recent_values, bins=20)
            recent_dist = recent_hist / np.sum(recent_hist) if np.sum(recent_hist) > 0 else recent_hist
            
            # Normalize baseline distribution
            baseline_dist = baseline_dist / np.sum(baseline_dist) if np.sum(baseline_dist) > 0 else baseline_dist
            
            # Calculate Wasserstein distance (simplified)
            if len(baseline_dist) == len(recent_dist):
                emd_distance = np.sum(np.abs(baseline_dist - recent_dist))
            else:
                emd_distance = 0.0
            
            # Combine drift indicators
            drift_score = (mean_shift + variance_change + emd_distance) / 3.0
            
            return drift_score
            
        except Exception as e:
            logger.error(f"Drift score calculation failed: {str(e)}")
            return 0.0
    
    def _determine_drift_severity(self, drift_score: float) -> AlertSeverity:
        """Determine alert severity based on drift score."""
        if drift_score > 1.0:
            return AlertSeverity.CRITICAL
        elif drift_score > 0.5:
            return AlertSeverity.HIGH
        elif drift_score > 0.2:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    async def _handle_drift_detection(self, drift_detection: DriftDetection):
        """Handle detected drift."""
        try:
            # Store drift detection
            self.drift_history.append(drift_detection)
            self.active_drifts[f"{drift_detection.model_id}_{drift_detection.drift_type.value}"] = drift_detection
            self.monitor_stats['drift_detections'] += 1
            
            # Create alert
            alert_data = {
                'type': f'drift_detected_{drift_detection.drift_type.value}',
                'severity': drift_detection.severity,
                'message': f"{drift_detection.drift_type.value.replace('_', ' ').title()} detected with score {drift_detection.drift_score:.3f}",
                'data': {
                    'drift_type': drift_detection.drift_type.value,
                    'drift_score': drift_detection.drift_score,
                    'affected_features': drift_detection.affected_features,
                    'details': drift_detection.details
                }
            }
            
            await self._create_alert(drift_detection.model_id, alert_data)
            
            # Trigger retraining if enabled and severity is high
            if (self.enable_auto_retraining and 
                drift_detection.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]):
                await self._trigger_retraining(drift_detection.model_id, drift_detection)
            
            logger.warning(f"Drift detected for model {drift_detection.model_id}: {drift_detection.drift_type.value}")
            
        except Exception as e:
            logger.error(f"Drift handling failed: {str(e)}")
    
    async def _create_alert(self, model_id: str, alert_data: Dict[str, Any]):
        """Create and process a model alert."""
        try:
            alert_id = f"{model_id}_{alert_data['type']}_{int(time.time())}"
            
            alert = ModelAlert(
                alert_id=alert_id,
                model_id=model_id,
                alert_type=alert_data['type'],
                severity=alert_data['severity'],
                message=alert_data['message'],
                timestamp=datetime.now(),
                data=alert_data['data']
            )
            
            # Store alert
            self.alerts.append(alert)
            self.active_alerts[alert_id] = alert
            self.monitor_stats['alerts_triggered'] += 1
            
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as callback_error:
                    logger.error(f"Alert callback failed: {str(callback_error)}")
            
            logger.warning(f"Model alert created: {alert.message}")
            
        except Exception as e:
            logger.error(f"Alert creation failed: {str(e)}")
    
    async def _trigger_retraining(self, model_id: str, drift_detection: DriftDetection):
        """Trigger model retraining due to drift."""
        try:
            # This would typically integrate with your ML training pipeline
            logger.info(f"Triggering retraining for model {model_id} due to {drift_detection.drift_type.value}")
            
            self.monitor_stats['retraining_triggered'] += 1
            
            # Create retraining alert
            alert_data = {
                'type': 'retraining_triggered',
                'severity': AlertSeverity.MEDIUM,
                'message': f"Automatic retraining triggered for model {model_id}",
                'data': {
                    'trigger_reason': drift_detection.drift_type.value,
                    'drift_score': drift_detection.drift_score
                }
            }
            
            await self._create_alert(model_id, alert_data)
            
        except Exception as e:
            logger.error(f"Retraining trigger failed: {str(e)}")
    
    def register_alert_callback(self, callback: Callable) -> bool:
        """Register callback for alert notifications."""
        try:
            self.alert_callbacks.append(callback)
            logger.info("Registered alert callback")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register alert callback: {str(e)}")
            return False
    
    async def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive status for a model."""
        try:
            if model_id not in self.registered_models:
                return {'error': 'Model not found'}
            
            model_info = self.registered_models[model_id]
            current_perf = self.current_performance.get(model_id)
            
            # Get recent alerts
            recent_alerts = [
                {
                    'alert_id': alert.alert_id,
                    'type': alert.alert_type,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'acknowledged': alert.acknowledged
                }
                for alert in self.alerts[-10:]
                if alert.model_id == model_id
            ]
            
            # Get active drifts
            active_drifts = [
                {
                    'drift_type': drift.drift_type.value,
                    'severity': drift.severity.value,
                    'drift_score': drift.drift_score,
                    'affected_features': drift.affected_features,
                    'timestamp': drift.timestamp.isoformat()
                }
                for key, drift in self.active_drifts.items()
                if key.startswith(model_id)
            ]
            
            return {
                'model_id': model_id,
                'model_info': {
                    'type': model_info['model_type'].value,
                    'registered_at': model_info['registered_at'].isoformat(),
                    'prediction_count': model_info['prediction_count'],
                    'baseline_established': model_info['baseline_established']
                },
                'current_performance': asdict(current_perf) if current_perf else None,
                'recent_alerts': recent_alerts,
                'active_drifts': active_drifts,
                'health_status': self._calculate_health_status(model_id)
            }
            
        except Exception as e:
            logger.error(f"Failed to get model status: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_health_status(self, model_id: str) -> str:
        """Calculate overall health status for a model."""
        try:
            # Check for critical alerts
            critical_alerts = [
                alert for alert in self.alerts[-10:]
                if (alert.model_id == model_id and 
                    alert.severity == AlertSeverity.CRITICAL and 
                    not alert.resolved)
            ]
            
            if critical_alerts:
                return "critical"
            
            # Check for active high-severity drifts
            high_severity_drifts = [
                drift for key, drift in self.active_drifts.items()
                if (key.startswith(model_id) and 
                    drift.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL])
            ]
            
            if high_severity_drifts:
                return "degraded"
            
            # Check recent performance
            if model_id in self.current_performance:
                current_perf = self.current_performance[model_id]
                if current_perf.error_rate > 0.1 or current_perf.accuracy < 0.7:
                    return "warning"
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"Health status calculation failed: {str(e)}")
            return "unknown"
    
    async def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        try:
            return {
                'monitor_stats': self.monitor_stats.copy(),
                'registered_models': {
                    model_id: {
                        'type': info['model_type'].value,
                        'prediction_count': info['prediction_count'],
                        'health_status': self._calculate_health_status(model_id)
                    }
                    for model_id, info in self.registered_models.items()
                },
                'active_alerts_count': len(self.active_alerts),
                'active_drifts_count': len(self.active_drifts),
                'recent_alerts': [
                    {
                        'model_id': alert.model_id,
                        'type': alert.alert_type,
                        'severity': alert.severity.value,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in self.alerts[-20:]
                ],
                'drift_summary': {
                    drift_type.value: len([
                        d for d in self.drift_history[-100:]
                        if d.drift_type == drift_type
                    ])
                    for drift_type in DriftType
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get monitoring summary: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Clean up monitoring resources."""
        try:
            # Cancel monitoring tasks
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            if self.drift_detection_task:
                self.drift_detection_task.cancel()
                try:
                    await self.drift_detection_task
                except asyncio.CancelledError:
                    pass
            
            # Clear data structures
            self.registered_models.clear()
            self.model_predictions.clear()
            self.baseline_distributions.clear()
            self.performance_history.clear()
            self.current_performance.clear()
            self.drift_detectors.clear()
            self.drift_history.clear()
            self.active_drifts.clear()
            self.alerts.clear()
            self.active_alerts.clear()
            self.alert_callbacks.clear()
            
            logger.info("ModelPerformanceMonitor cleanup completed")
            
        except Exception as e:
            logger.error(f"ModelPerformanceMonitor cleanup failed: {str(e)}")