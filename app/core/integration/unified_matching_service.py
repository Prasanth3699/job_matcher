"""
Unified Matching Service for the complete ML pipeline integration.

This service provides a single entry point that orchestrates all advanced ML
components including ensemble scoring, neural ranking, and A/B testing.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from app.utils.logger import logger
from app.core.constants.ml_constants import (
    PerformanceMetrics,
    EnsembleConfig,
    RankingConfig
)
from app.core.integration.ensemble_ranking_bridge import (
    EnsembleRankingBridge,
    IntegrationStrategy,
    IntegratedResult
)
from app.core.ab_testing.experiment_manager import ExperimentManager
from app.core.ab_testing.traffic_splitter import TrafficSplitter
from app.core.ab_testing.performance_monitor import PerformanceMonitor
from app.core.ab_testing.model_variant import ModelVariant, VariantType


class ServiceMode(str, Enum):
    """Operating modes for the unified service."""
    PRODUCTION = "production"          # Full production mode with A/B testing
    CHAMPION_ONLY = "champion_only"    # Only use champion model
    ENSEMBLE_ONLY = "ensemble_only"    # Only ensemble scoring
    RANKING_ONLY = "ranking_only"      # Only neural ranking
    TESTING = "testing"                # Testing mode with all features


@dataclass
class MatchingRequest:
    """Request structure for unified matching."""
    user_id: str
    resume_data: Dict[str, Any]
    jobs: List[Dict[str, Any]]
    top_n: int = 5
    include_explanations: bool = True
    context: Optional[Dict[str, Any]] = None
    force_variant: Optional[str] = None


@dataclass
class MatchingResponse:
    """Response structure for unified matching."""
    matches: List[IntegratedResult]
    metadata: Dict[str, Any]
    performance_info: Dict[str, Any]
    experiment_info: Optional[Dict[str, Any]] = None


class UnifiedMatchingService:
    """
    Unified matching service that integrates ensemble scoring, neural ranking,
    and A/B testing for comprehensive job matching capabilities.
    """
    
    def __init__(
        self,
        mode: ServiceMode = ServiceMode.PRODUCTION,
        enable_ab_testing: bool = True,
        default_integration_strategy: IntegrationStrategy = IntegrationStrategy.HYBRID
    ):
        """Initialize the unified matching service."""
        self.mode = mode
        self.enable_ab_testing = enable_ab_testing
        self.default_integration_strategy = default_integration_strategy
        
        # Core components
        self.ensemble_ranking_bridge: Optional[EnsembleRankingBridge] = None
        self.experiment_manager: Optional[ExperimentManager] = None
        self.traffic_splitter: Optional[TrafficSplitter] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        
        # Service state
        self.is_initialized = False
        self.champion_variant: Optional[str] = None
        self.active_experiments: Dict[str, str] = {}
        
        # Performance tracking
        self.service_stats = {
            'total_requests': 0,
            'requests_by_mode': {mode.value: 0 for mode in ServiceMode},
            'ab_test_assignments': 0,
            'fallback_predictions': 0,
            'error_count': 0,
            'average_latency_ms': 0.0,
            'total_latency_ms': 0.0
        }
        
        logger.info(f"UnifiedMatchingService initialized in {mode.value} mode")
    
    async def initialize(self) -> bool:
        """Initialize all service components."""
        try:
            # Initialize ensemble-ranking bridge
            self.ensemble_ranking_bridge = EnsembleRankingBridge(
                integration_strategy=self.default_integration_strategy
            )
            await self.ensemble_ranking_bridge.initialize()
            
            # Initialize A/B testing components if enabled
            if self.enable_ab_testing and self.mode in [ServiceMode.PRODUCTION, ServiceMode.TESTING]:
                self.experiment_manager = ExperimentManager()
                await self.experiment_manager.initialize()
                
                self.traffic_splitter = TrafficSplitter()
                self.performance_monitor = PerformanceMonitor()
                
                # Setup default experiment if in production mode
                if self.mode == ServiceMode.PRODUCTION:
                    await self._setup_default_experiment()
            
            self.is_initialized = True
            logger.info("UnifiedMatchingService initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"UnifiedMatchingService initialization failed: {str(e)}")
            return False
    
    async def match_jobs(self, request: MatchingRequest) -> MatchingResponse:
        """
        Perform unified job matching with full pipeline integration.
        
        Args:
            request: Matching request with all required parameters
            
        Returns:
            Comprehensive matching response with results and metadata
        """
        try:
            start_time = datetime.now()
            
            if not self.is_initialized:
                raise ValueError("Service not initialized")
            
            # Update request stats
            self.service_stats['total_requests'] += 1
            self.service_stats['requests_by_mode'][self.mode.value] += 1
            
            # Determine execution strategy
            execution_strategy = await self._determine_execution_strategy(request)
            
            # Execute matching based on strategy
            if execution_strategy['use_ab_testing']:
                matches, experiment_info = await self._execute_ab_testing_matching(request, execution_strategy)
            else:
                matches = await self._execute_direct_matching(request, execution_strategy)
                experiment_info = None
            
            # Calculate performance metrics
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update performance tracking
            self._update_service_stats(latency_ms)
            
            # Track performance event if monitoring is enabled
            if self.performance_monitor and experiment_info:
                await self._track_performance_event(request, matches, latency_ms, experiment_info)
            
            # Build response
            response = MatchingResponse(
                matches=matches,
                metadata=self._build_response_metadata(request, execution_strategy),
                performance_info={
                    'latency_ms': latency_ms,
                    'processing_timestamp': end_time.isoformat(),
                    'execution_strategy': execution_strategy
                },
                experiment_info=experiment_info
            )
            
            logger.info(f"Unified matching completed for user {request.user_id} in {latency_ms:.2f}ms")
            
            return response
            
        except Exception as e:
            self.service_stats['error_count'] += 1
            logger.error(f"Unified matching failed: {str(e)}")
            
            # Return fallback response
            return await self._create_fallback_response(request, str(e))
    
    async def _determine_execution_strategy(self, request: MatchingRequest) -> Dict[str, Any]:
        """Determine how to execute the matching request."""
        strategy = {
            'use_ab_testing': False,
            'variant_id': None,
            'integration_strategy': self.default_integration_strategy,
            'fallback_reason': None
        }
        
        try:
            # Check if A/B testing should be used
            if (self.enable_ab_testing and 
                self.mode in [ServiceMode.PRODUCTION, ServiceMode.TESTING] and
                self.experiment_manager and 
                self.traffic_splitter and
                not request.force_variant):
                
                # Get active experiments
                active_experiments = await self.experiment_manager.get_active_experiments()
                
                if active_experiments:
                    strategy['use_ab_testing'] = True
                    
                    # Get variant assignment
                    experiment_id = list(active_experiments.keys())[0]  # Use first active experiment
                    variant_id = await self.traffic_splitter.assign_variant(
                        experiment_id, request.user_id, request.context
                    )
                    
                    if variant_id:
                        strategy['variant_id'] = variant_id
                        self.service_stats['ab_test_assignments'] += 1
                    else:
                        strategy['use_ab_testing'] = False
                        strategy['fallback_reason'] = 'variant_assignment_failed'
            
            # Handle forced variant
            elif request.force_variant:
                strategy['variant_id'] = request.force_variant
                strategy['use_ab_testing'] = True
            
            # Mode-specific overrides
            elif self.mode == ServiceMode.CHAMPION_ONLY:
                strategy['variant_id'] = self.champion_variant
            elif self.mode == ServiceMode.ENSEMBLE_ONLY:
                strategy['integration_strategy'] = IntegrationStrategy.ENSEMBLE_FIRST
            elif self.mode == ServiceMode.RANKING_ONLY:
                strategy['integration_strategy'] = IntegrationStrategy.RANKING_FIRST
            
            return strategy
            
        except Exception as e:
            logger.error(f"Strategy determination failed: {str(e)}")
            strategy['fallback_reason'] = f'strategy_error: {str(e)}'
            return strategy
    
    async def _execute_ab_testing_matching(
        self,
        request: MatchingRequest,
        strategy: Dict[str, Any]
    ) -> Tuple[List[IntegratedResult], Dict[str, Any]]:
        """Execute matching with A/B testing."""
        try:
            variant_id = strategy['variant_id']
            
            if not variant_id:
                # Fallback to direct matching
                matches = await self._execute_direct_matching(request, strategy)
                return matches, {'variant_used': 'fallback', 'reason': 'no_variant_assigned'}
            
            # Get variant configuration
            variant_config = await self._get_variant_config(variant_id)
            
            if not variant_config:
                # Fallback to direct matching
                matches = await self._execute_direct_matching(request, strategy)
                return matches, {'variant_used': 'fallback', 'reason': 'variant_config_not_found'}
            
            # Execute matching with variant configuration
            integration_strategy = variant_config.get('integration_strategy', self.default_integration_strategy)
            
            matches = await self.ensemble_ranking_bridge.integrated_match(
                request.resume_data,
                request.jobs,
                top_n=request.top_n,
                include_explanations=request.include_explanations,
                strategy_override=integration_strategy
            )
            
            experiment_info = {
                'variant_used': variant_id,
                'integration_strategy': integration_strategy.value,
                'experiment_mode': True,
                'variant_config': variant_config
            }
            
            return matches, experiment_info
            
        except Exception as e:
            logger.error(f"A/B testing matching failed: {str(e)}")
            
            # Fallback to direct matching
            matches = await self._execute_direct_matching(request, strategy)
            return matches, {'variant_used': 'fallback', 'reason': f'ab_error: {str(e)}'}
    
    async def _execute_direct_matching(
        self,
        request: MatchingRequest,
        strategy: Dict[str, Any]
    ) -> List[IntegratedResult]:
        """Execute direct matching without A/B testing."""
        try:
            integration_strategy = strategy.get('integration_strategy', self.default_integration_strategy)
            
            matches = await self.ensemble_ranking_bridge.integrated_match(
                request.resume_data,
                request.jobs,
                top_n=request.top_n,
                include_explanations=request.include_explanations,
                strategy_override=integration_strategy
            )
            
            return matches
            
        except Exception as e:
            logger.error(f"Direct matching failed: {str(e)}")
            
            # Create fallback matches
            return await self._create_fallback_matches(request)
    
    async def _get_variant_config(self, variant_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific variant."""
        try:
            if not self.experiment_manager:
                return None
            
            # Get active experiments
            active_experiments = await self.experiment_manager.get_active_experiments()
            
            for experiment_id, experiment_data in active_experiments.items():
                variants = experiment_data.get('variants', {})
                if variant_id in variants:
                    variant_info = variants[variant_id]
                    
                    # Map variant to integration strategy
                    variant_config = {
                        'integration_strategy': self._map_variant_to_strategy(variant_info),
                        'variant_info': variant_info
                    }
                    
                    return variant_config
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get variant config for {variant_id}: {str(e)}")
            return None
    
    def _map_variant_to_strategy(self, variant_info: Dict[str, Any]) -> IntegrationStrategy:
        """Map variant information to integration strategy."""
        try:
            variant_name = variant_info.get('name', '').lower()
            
            if 'ensemble' in variant_name and 'first' in variant_name:
                return IntegrationStrategy.ENSEMBLE_FIRST
            elif 'ranking' in variant_name and 'first' in variant_name:
                return IntegrationStrategy.RANKING_FIRST
            elif 'parallel' in variant_name:
                return IntegrationStrategy.PARALLEL
            elif 'adaptive' in variant_name:
                return IntegrationStrategy.ADAPTIVE
            else:
                return IntegrationStrategy.HYBRID
                
        except Exception:
            return self.default_integration_strategy
    
    async def _track_performance_event(
        self,
        request: MatchingRequest,
        matches: List[IntegratedResult],
        latency_ms: float,
        experiment_info: Dict[str, Any]
    ):
        """Track performance event for monitoring."""
        try:
            if not self.performance_monitor:
                return
            
            # Extract experiment information
            variant_id = experiment_info.get('variant_used')
            if not variant_id or variant_id == 'fallback':
                return
            
            # Find experiment ID
            experiment_id = None
            active_experiments = await self.experiment_manager.get_active_experiments()
            
            for exp_id, exp_data in active_experiments.items():
                if variant_id in exp_data.get('variants', {}):
                    experiment_id = exp_id
                    break
            
            if not experiment_id:
                return
            
            # Prepare event data
            event_data = {
                'latency_ms': latency_ms,
                'match_count': len(matches),
                'top_score': matches[0].final_score if matches else 0.0,
                'average_confidence': sum(m.confidence for m in matches) / len(matches) if matches else 0.0,
                'integration_strategy': experiment_info.get('integration_strategy', 'unknown')
            }
            
            # Track performance event
            await self.performance_monitor.track_event(
                experiment_id=experiment_id,
                user_id=request.user_id,
                event_type='prediction',
                event_data=event_data
            )
            
        except Exception as e:
            logger.error(f"Performance event tracking failed: {str(e)}")
    
    async def _create_fallback_matches(self, request: MatchingRequest) -> List[IntegratedResult]:
        """Create fallback matches when normal processing fails."""
        try:
            self.service_stats['fallback_predictions'] += 1
            
            fallback_matches = []
            
            # Create simple fallback matches
            for i, job in enumerate(request.jobs[:request.top_n]):
                job_id = job.get('id', f'job_{i}')
                
                # Simple scoring based on position
                score = max(0.1, 1.0 - (i * 0.1))
                
                fallback_match = IntegratedResult(
                    job_id=job_id,
                    final_score=score,
                    ensemble_score=score,
                    ranking_score=score,
                    confidence=0.5,
                    explanation={
                        'type': 'fallback',
                        'message': 'Fallback prediction due to system error',
                        'position': i + 1
                    },
                    job_details=job,
                    ranking_position=i + 1
                )
                
                fallback_matches.append(fallback_match)
            
            return fallback_matches
            
        except Exception as e:
            logger.error(f"Fallback match creation failed: {str(e)}")
            return []
    
    async def _create_fallback_response(
        self,
        request: MatchingRequest,
        error_message: str
    ) -> MatchingResponse:
        """Create fallback response when matching completely fails."""
        fallback_matches = await self._create_fallback_matches(request)
        
        return MatchingResponse(
            matches=fallback_matches,
            metadata={
                'service_mode': self.mode.value,
                'fallback_response': True,
                'error_message': error_message,
                'request_timestamp': datetime.now().isoformat()
            },
            performance_info={
                'latency_ms': 0.0,
                'fallback': True,
                'error': error_message
            }
        )
    
    def _build_response_metadata(
        self,
        request: MatchingRequest,
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build response metadata."""
        return {
            'service_mode': self.mode.value,
            'ab_testing_enabled': self.enable_ab_testing,
            'user_id': request.user_id,
            'request_timestamp': datetime.now().isoformat(),
            'execution_strategy': strategy,
            'job_count': len(request.jobs),
            'requested_top_n': request.top_n,
            'include_explanations': request.include_explanations,
            'service_stats': self.service_stats.copy()
        }
    
    def _update_service_stats(self, latency_ms: float):
        """Update service performance statistics."""
        try:
            self.service_stats['total_latency_ms'] += latency_ms
            
            if self.service_stats['total_requests'] > 0:
                self.service_stats['average_latency_ms'] = (
                    self.service_stats['total_latency_ms'] / self.service_stats['total_requests']
                )
            
        except Exception as e:
            logger.error(f"Service stats update failed: {str(e)}")
    
    async def _setup_default_experiment(self) -> bool:
        """Setup default A/B testing experiment."""
        try:
            if not self.experiment_manager or not self.traffic_splitter:
                return False
            
            # Create model variants
            control_variant = self._create_control_variant()
            treatment_variants = self._create_treatment_variants()
            
            all_variants = [control_variant] + treatment_variants
            
            # Register variants with traffic splitter
            traffic_allocation = {
                'control_ensemble_hybrid': 0.4,      # 40% control
                'treatment_ensemble_first': 0.2,     # 20% ensemble-first
                'treatment_ranking_first': 0.2,      # 20% ranking-first
                'treatment_adaptive': 0.2            # 20% adaptive
            }
            
            await self.traffic_splitter.register_experiment(
                experiment_id='default_integration_experiment',
                variants=all_variants,
                traffic_allocation=traffic_allocation
            )
            
            # Create experiment
            experiment_id = await self.experiment_manager.create_experiment(
                name='Default Integration Strategy Experiment',
                description='Compare different integration strategies for ensemble and ranking',
                experiment_type='integration_strategy',
                variants=all_variants,
                traffic_allocation=traffic_allocation,
                duration_days=30
            )
            
            if experiment_id:
                self.active_experiments['default'] = experiment_id
                self.champion_variant = 'control_ensemble_hybrid'
                
                logger.info(f"Setup default experiment: {experiment_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Default experiment setup failed: {str(e)}")
            return False
    
    def _create_control_variant(self) -> ModelVariant:
        """Create control variant."""
        from app.core.ab_testing.model_variant import (
            ModelVariant, VariantType, VariantConfig, ModelType
        )
        
        config = VariantConfig(
            model_type=ModelType.HYBRID,
            model_parameters={'integration_strategy': 'hybrid'},
            feature_config={'use_all_features': True},
            preprocessing_config={'standard_preprocessing': True},
            postprocessing_config={'standard_postprocessing': True}
        )
        
        return ModelVariant(
            variant_id='control_ensemble_hybrid',
            name='Control: Ensemble Hybrid',
            description='Control variant using hybrid integration strategy',
            variant_type=VariantType.CONTROL,
            config=config
        )
    
    def _create_treatment_variants(self) -> List[ModelVariant]:
        """Create treatment variants."""
        from app.core.ab_testing.model_variant import (
            ModelVariant, VariantType, VariantConfig, ModelType
        )
        
        variants = []
        
        # Ensemble-first treatment
        config1 = VariantConfig(
            model_type=ModelType.HYBRID,
            model_parameters={'integration_strategy': 'ensemble_first'},
            feature_config={'use_all_features': True},
            preprocessing_config={'standard_preprocessing': True},
            postprocessing_config={'standard_postprocessing': True}
        )
        
        variants.append(ModelVariant(
            variant_id='treatment_ensemble_first',
            name='Treatment: Ensemble First',
            description='Treatment variant using ensemble-first strategy',
            variant_type=VariantType.TREATMENT,
            config=config1
        ))
        
        # Ranking-first treatment
        config2 = VariantConfig(
            model_type=ModelType.HYBRID,
            model_parameters={'integration_strategy': 'ranking_first'},
            feature_config={'use_all_features': True},
            preprocessing_config={'standard_preprocessing': True},
            postprocessing_config={'standard_postprocessing': True}
        )
        
        variants.append(ModelVariant(
            variant_id='treatment_ranking_first',
            name='Treatment: Ranking First',
            description='Treatment variant using ranking-first strategy',
            variant_type=VariantType.TREATMENT,
            config=config2
        ))
        
        # Adaptive treatment
        config3 = VariantConfig(
            model_type=ModelType.HYBRID,
            model_parameters={'integration_strategy': 'adaptive'},
            feature_config={'use_all_features': True},
            preprocessing_config={'standard_preprocessing': True},
            postprocessing_config={'standard_postprocessing': True}
        )
        
        variants.append(ModelVariant(
            variant_id='treatment_adaptive',
            name='Treatment: Adaptive',
            description='Treatment variant using adaptive integration strategy',
            variant_type=VariantType.TREATMENT,
            config=config3
        ))
        
        return variants
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        try:
            status = {
                'service_info': {
                    'mode': self.mode.value,
                    'ab_testing_enabled': self.enable_ab_testing,
                    'is_initialized': self.is_initialized,
                    'default_integration_strategy': self.default_integration_strategy.value
                },
                'component_status': {
                    'ensemble_ranking_bridge': self.ensemble_ranking_bridge is not None,
                    'experiment_manager': self.experiment_manager is not None,
                    'traffic_splitter': self.traffic_splitter is not None,
                    'performance_monitor': self.performance_monitor is not None
                },
                'service_stats': self.service_stats.copy(),
                'active_experiments': self.active_experiments.copy(),
                'champion_variant': self.champion_variant
            }
            
            # Get component-specific status
            if self.ensemble_ranking_bridge:
                bridge_stats = await self.ensemble_ranking_bridge.get_integration_stats()
                status['bridge_stats'] = bridge_stats
            
            if self.experiment_manager:
                experiment_stats = await self.experiment_manager.get_manager_stats()
                status['experiment_stats'] = experiment_stats
            
            if self.traffic_splitter:
                splitter_stats = await self.traffic_splitter.get_splitter_stats()
                status['splitter_stats'] = splitter_stats
            
            if self.performance_monitor:
                monitor_stats = await self.performance_monitor.get_monitor_stats()
                status['monitor_stats'] = monitor_stats
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get service status: {str(e)}")
            return {'error': str(e)}
    
    async def update_service_mode(self, new_mode: ServiceMode) -> bool:
        """Update service operating mode."""
        try:
            old_mode = self.mode
            self.mode = new_mode
            
            logger.info(f"Service mode changed from {old_mode.value} to {new_mode.value}")
            
            # Reinitialize if needed
            if new_mode in [ServiceMode.PRODUCTION, ServiceMode.TESTING] and not self.experiment_manager:
                await self.initialize()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update service mode: {str(e)}")
            return False
    
    async def cleanup(self):
        """Clean up service resources."""
        try:
            # Clean up components
            if self.ensemble_ranking_bridge:
                await self.ensemble_ranking_bridge.cleanup()
            
            if self.experiment_manager:
                await self.experiment_manager.cleanup()
            
            if self.traffic_splitter:
                await self.traffic_splitter.cleanup()
            
            if self.performance_monitor:
                await self.performance_monitor.cleanup()
            
            # Reset state
            self.is_initialized = False
            self.active_experiments.clear()
            self.champion_variant = None
            
            # Reset stats
            self.service_stats = {
                'total_requests': 0,
                'requests_by_mode': {mode.value: 0 for mode in ServiceMode},
                'ab_test_assignments': 0,
                'fallback_predictions': 0,
                'error_count': 0,
                'average_latency_ms': 0.0,
                'total_latency_ms': 0.0
            }
            
            logger.info("UnifiedMatchingService cleanup completed")
            
        except Exception as e:
            logger.error(f"UnifiedMatchingService cleanup failed: {str(e)}")