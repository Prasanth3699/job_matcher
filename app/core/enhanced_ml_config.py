"""
Enhanced ML Pipeline Configuration and Initialization System.

This module provides centralized configuration and initialization for all
enhanced ML components including advanced pipeline features, A/B testing, 
optimization, and analytics capabilities.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from app.utils.logger import logger
from app.core.constants.ml_constants import EnsembleConfig, RankingConfig, PerformanceMetrics

# Import Enhanced ML Pipeline components
from app.core.integration.unified_matching_service import UnifiedMatchingService, ServiceMode
from app.core.integration.ensemble_ranking_bridge import EnsembleRankingBridge, IntegrationStrategy
from app.core.ab_testing.experiment_manager import ExperimentManager
from app.core.ab_testing.traffic_splitter import TrafficSplitter, AllocationStrategy
from app.core.ab_testing.performance_monitor import PerformanceMonitor
from app.core.optimization.cache_optimizer import CacheOptimizer, CacheStrategy
from app.core.optimization.query_optimizer import QueryOptimizer
from app.core.optimization.batch_processor import BatchProcessor, BatchStrategy
from app.core.optimization.resource_manager import ResourceManager
from app.core.analytics.analytics_engine import AnalyticsEngine
from app.core.analytics.model_monitor import ModelPerformanceMonitor
from app.core.analytics.business_metrics import BusinessMetricsCollector
from app.core.analytics.dashboard_service import DashboardService


class EnhancedMLMode(str, Enum):
    """Enhanced ML Pipeline operation modes."""
    FULL_PRODUCTION = "full_production"           # All features enabled
    BASIC_PRODUCTION = "basic_production"         # Core features only
    TESTING = "testing"                           # Testing mode
    DEVELOPMENT = "development"                   # Development mode
    ANALYTICS_ONLY = "analytics_only"             # Analytics features only


@dataclass
class EnhancedMLConfig:
    """Configuration for Enhanced ML Pipeline components."""
    # Core integration settings
    unified_service_mode: ServiceMode = ServiceMode.PRODUCTION
    integration_strategy: IntegrationStrategy = IntegrationStrategy.HYBRID
    enable_ab_testing: bool = True
    
    # A/B testing settings
    ab_testing_config: Dict[str, Any] = None
    traffic_allocation_strategy: AllocationStrategy = AllocationStrategy.HASH_BASED
    
    # Optimization settings
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    cache_size_mb: int = 500
    batch_strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    enable_resource_monitoring: bool = True
    
    # Analytics settings
    enable_analytics: bool = True
    enable_model_monitoring: bool = True
    enable_business_metrics: bool = True
    enable_dashboards: bool = True
    analytics_retention_days: int = 30
    
    # Performance settings
    monitoring_interval_seconds: int = 300
    optimization_interval_seconds: int = 600
    
    def __post_init__(self):
        if self.ab_testing_config is None:
            self.ab_testing_config = {
                'default_experiment_duration_days': 14,
                'min_sample_size': 1000,
                'significance_level': 0.05,
                'power': 0.8
            }


@dataclass
class EnhancedMLStatus:
    """Status of Enhanced ML Pipeline components."""
    mode: EnhancedMLMode
    initialized_at: datetime
    components_status: Dict[str, str]
    health_score: float
    active_experiments: int
    total_requests_processed: int
    error_count: int
    last_health_check: datetime


class EnhancedMLManager:
    """
    Centralized manager for all Enhanced ML Pipeline components providing unified
    initialization, configuration, and lifecycle management.
    """
    
    def __init__(
        self,
        mode: EnhancedMLMode = EnhancedMLMode.FULL_PRODUCTION,
        config: Optional[EnhancedMLConfig] = None
    ):
        """Initialize Enhanced ML Pipeline manager."""
        self.mode = mode
        self.config = config or EnhancedMLConfig()
        
        # Component instances
        self.unified_service: Optional[UnifiedMatchingService] = None
        self.ensemble_ranking_bridge: Optional[EnsembleRankingBridge] = None
        self.experiment_manager: Optional[ExperimentManager] = None
        self.traffic_splitter: Optional[TrafficSplitter] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.cache_optimizer: Optional[CacheOptimizer] = None
        self.query_optimizer: Optional[QueryOptimizer] = None
        self.batch_processor: Optional[BatchProcessor] = None
        self.resource_manager: Optional[ResourceManager] = None
        self.analytics_engine: Optional[AnalyticsEngine] = None
        self.model_monitor: Optional[ModelPerformanceMonitor] = None
        self.business_metrics: Optional[BusinessMetricsCollector] = None
        self.dashboard_service: Optional[DashboardService] = None
        
        # Manager state
        self.is_initialized = False
        self.initialization_order: List[str] = []
        self.component_dependencies: Dict[str, List[str]] = {}
        self.status: Optional[EnhancedMLStatus] = None
        
        # Statistics
        self.manager_stats = {
            'initialization_count': 0,
            'total_requests': 0,
            'successful_initializations': 0,
            'failed_initializations': 0,
            'component_restart_count': 0,
            'health_checks_performed': 0
        }
        
        self._setup_component_dependencies()
        
        logger.info(f"EnhancedMLManager initialized in {mode.value} mode")
    
    def _setup_component_dependencies(self):
        """Setup component initialization dependencies."""
        try:
            # Define component dependencies for proper initialization order
            self.component_dependencies = {
                'cache_optimizer': [],
                'query_optimizer': [],
                'resource_manager': [],
                'batch_processor': [],
                'analytics_engine': [],
                'business_metrics': [],
                'model_monitor': [],
                'ensemble_ranking_bridge': [],
                'experiment_manager': [],
                'traffic_splitter': ['experiment_manager'],
                'performance_monitor': ['experiment_manager'],
                'unified_service': [
                    'ensemble_ranking_bridge', 
                    'experiment_manager', 
                    'traffic_splitter', 
                    'performance_monitor'
                ],
                'dashboard_service': [
                    'analytics_engine', 
                    'model_monitor', 
                    'business_metrics'
                ]
            }
            
            # Calculate initialization order
            self.initialization_order = self._calculate_initialization_order()
            
        except Exception as e:
            logger.error(f"Failed to setup component dependencies: {str(e)}")
    
    def _calculate_initialization_order(self) -> List[str]:
        """Calculate component initialization order based on dependencies."""
        try:
            order = []
            remaining = set(self.component_dependencies.keys())
            
            while remaining:
                # Find components with no remaining dependencies
                ready = []
                for component in remaining:
                    deps = self.component_dependencies[component]
                    if all(dep in order for dep in deps):
                        ready.append(component)
                
                if not ready:
                    # Circular dependency or error
                    logger.warning("Circular dependency detected, using fallback order")
                    order.extend(list(remaining))
                    break
                
                # Sort ready components for consistent ordering
                ready.sort()
                order.extend(ready)
                remaining -= set(ready)
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to calculate initialization order: {str(e)}")
            return list(self.component_dependencies.keys())
    
    async def initialize(self) -> bool:
        """Initialize all Enhanced ML Pipeline components based on mode and configuration."""
        try:
            logger.info(f"Starting Enhanced ML Pipeline initialization in {self.mode.value} mode")
            start_time = datetime.now()
            
            self.manager_stats['initialization_count'] += 1
            
            # Initialize components based on mode
            components_to_init = self._get_components_for_mode()
            
            initialization_success = True
            
            for component_name in self.initialization_order:
                if component_name in components_to_init:
                    try:
                        success = await self._initialize_component(component_name)
                        if success:
                            self.manager_stats['successful_initializations'] += 1
                        else:
                            self.manager_stats['failed_initializations'] += 1
                            initialization_success = False
                            
                            # Determine if this is a critical failure
                            if self._is_critical_component(component_name):
                                logger.error(f"Critical component {component_name} failed to initialize")
                                return False
                            
                    except Exception as comp_error:
                        logger.error(f"Component {component_name} initialization failed: {str(comp_error)}")
                        self.manager_stats['failed_initializations'] += 1
                        
                        if self._is_critical_component(component_name):
                            return False
            
            # Initialize cross-component integrations
            await self._setup_component_integrations()
            
            # Create status
            self.status = EnhancedMLStatus(
                mode=self.mode,
                initialized_at=start_time,
                components_status=self._get_components_status(),
                health_score=self._calculate_health_score(),
                active_experiments=0,
                total_requests_processed=0,
                error_count=0,
                last_health_check=datetime.now()
            )
            
            self.is_initialized = initialization_success
            
            initialization_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Enhanced ML Pipeline initialization completed in {initialization_time:.2f}s, success: {initialization_success}")
            
            return initialization_success
            
        except Exception as e:
            logger.error(f"Enhanced ML Pipeline initialization failed: {str(e)}")
            return False
    
    def _get_components_for_mode(self) -> List[str]:
        """Get list of components to initialize based on mode."""
        try:
            if self.mode == EnhancedMLMode.FULL_PRODUCTION:
                return list(self.component_dependencies.keys())
            
            elif self.mode == EnhancedMLMode.BASIC_PRODUCTION:
                return [
                    'cache_optimizer',
                    'query_optimizer',
                    'ensemble_ranking_bridge',
                    'unified_service',
                    'analytics_engine',
                    'business_metrics'
                ]
            
            elif self.mode == EnhancedMLMode.TESTING:
                return [
                    'cache_optimizer',
                    'ensemble_ranking_bridge',
                    'experiment_manager',
                    'traffic_splitter',
                    'performance_monitor',
                    'unified_service',
                    'analytics_engine'
                ]
            
            elif self.mode == EnhancedMLMode.DEVELOPMENT:
                return [
                    'ensemble_ranking_bridge',
                    'unified_service',
                    'analytics_engine'
                ]
            
            elif self.mode == EnhancedMLMode.ANALYTICS_ONLY:
                return [
                    'analytics_engine',
                    'model_monitor',
                    'business_metrics',
                    'dashboard_service'
                ]
            
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to get components for mode {self.mode}: {str(e)}")
            return []
    
    def _is_critical_component(self, component_name: str) -> bool:
        """Check if component is critical for the current mode."""
        critical_components = {
            EnhancedMLMode.FULL_PRODUCTION: [
                'unified_service', 
                'ensemble_ranking_bridge'
            ],
            EnhancedMLMode.BASIC_PRODUCTION: [
                'unified_service', 
                'ensemble_ranking_bridge'
            ],
            EnhancedMLMode.TESTING: [
                'unified_service'
            ],
            EnhancedMLMode.DEVELOPMENT: [
                'unified_service'
            ],
            EnhancedMLMode.ANALYTICS_ONLY: [
                'analytics_engine'
            ]
        }
        
        return component_name in critical_components.get(self.mode, [])
    
    async def _initialize_component(self, component_name: str) -> bool:
        """Initialize a specific component."""
        try:
            logger.info(f"Initializing component: {component_name}")
            
            if component_name == 'cache_optimizer':
                self.cache_optimizer = CacheOptimizer(
                    max_size_mb=self.config.cache_size_mb,
                    strategy=self.config.cache_strategy
                )
                await self.cache_optimizer.initialize()
                
            elif component_name == 'query_optimizer':
                self.query_optimizer = QueryOptimizer(
                    enable_caching=True,
                    enable_batching=True
                )
                
            elif component_name == 'batch_processor':
                self.batch_processor = BatchProcessor(
                    default_strategy=self.config.batch_strategy
                )
                await self.batch_processor.initialize()
                
            elif component_name == 'resource_manager':
                self.resource_manager = ResourceManager(
                    monitoring_interval_seconds=self.config.monitoring_interval_seconds
                )
                await self.resource_manager.initialize()
                
            elif component_name == 'analytics_engine':
                self.analytics_engine = AnalyticsEngine(
                    enable_real_time=True,
                    retention_days=self.config.analytics_retention_days
                )
                await self.analytics_engine.initialize()
                
            elif component_name == 'model_monitor':
                self.model_monitor = ModelPerformanceMonitor(
                    monitoring_interval_seconds=self.config.monitoring_interval_seconds
                )
                await self.model_monitor.initialize()
                
            elif component_name == 'business_metrics':
                self.business_metrics = BusinessMetricsCollector(
                    collection_interval_seconds=self.config.monitoring_interval_seconds
                )
                await self.business_metrics.initialize()
                
            elif component_name == 'ensemble_ranking_bridge':
                self.ensemble_ranking_bridge = EnsembleRankingBridge(
                    integration_strategy=self.config.integration_strategy
                )
                await self.ensemble_ranking_bridge.initialize()
                
            elif component_name == 'experiment_manager':
                self.experiment_manager = ExperimentManager()
                await self.experiment_manager.initialize()
                
            elif component_name == 'traffic_splitter':
                self.traffic_splitter = TrafficSplitter(
                    default_strategy=self.config.traffic_allocation_strategy
                )
                
            elif component_name == 'performance_monitor':
                self.performance_monitor = PerformanceMonitor()
                
            elif component_name == 'unified_service':
                self.unified_service = UnifiedMatchingService(
                    mode=self.config.unified_service_mode,
                    enable_ab_testing=self.config.enable_ab_testing,
                    default_integration_strategy=self.config.integration_strategy
                )
                await self.unified_service.initialize()
                
            elif component_name == 'dashboard_service':
                self.dashboard_service = DashboardService(
                    analytics_engine=self.analytics_engine,
                    model_monitor=self.model_monitor,
                    business_metrics=self.business_metrics
                )
                await self.dashboard_service.initialize()
                
            else:
                logger.warning(f"Unknown component: {component_name}")
                return False
            
            logger.info(f"Successfully initialized component: {component_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize component {component_name}: {str(e)}")
            return False
    
    async def _setup_component_integrations(self):
        """Setup cross-component integrations and callbacks."""
        try:
            # Connect analytics engine with business metrics
            if self.analytics_engine and self.business_metrics:
                # Register business events processor
                self.analytics_engine.register_event_processor(
                    'business_event',
                    self._handle_business_analytics_event
                )
            
            # Connect model monitor with analytics
            if self.model_monitor and self.analytics_engine:
                # Register model performance callback
                self.model_monitor.register_alert_callback(
                    self._handle_model_alert
                )
            
            # Connect unified service with analytics
            if self.unified_service and self.analytics_engine:
                # This would integrate matching requests with analytics
                pass
            
            # Connect resource manager with optimization components
            if self.resource_manager and self.cache_optimizer:
                # Register scaling callbacks for cache optimization
                from app.core.optimization.resource_manager import ScalingAction, ResourceType
                
                self.resource_manager.register_scaling_callback(
                    ScalingAction.SCALE_UP,
                    self._handle_resource_scale_up
                )
            
            logger.info("Component integrations setup completed")
            
        except Exception as e:
            logger.error(f"Component integration setup failed: {str(e)}")
    
    async def _handle_business_analytics_event(self, event):
        """Handle business analytics events."""
        try:
            if self.business_metrics:
                # Convert analytics event to business event
                await self.business_metrics.track_event(
                    event_type=event.event_type.value,
                    user_id=event.user_id,
                    session_id=event.session_id,
                    properties=event.data
                )
        except Exception as e:
            logger.error(f"Business analytics event handling failed: {str(e)}")
    
    async def _handle_model_alert(self, alert):
        """Handle model performance alerts."""
        try:
            if self.analytics_engine:
                # Track model alert as analytics event
                await self.analytics_engine.track_event(
                    event_type='model_alert',
                    data={
                        'alert_type': alert.alert_type,
                        'severity': alert.severity.value,
                        'model_id': alert.model_id,
                        'message': alert.message
                    }
                )
        except Exception as e:
            logger.error(f"Model alert handling failed: {str(e)}")
    
    async def _handle_resource_scale_up(self, resource_type, current_value, threshold):
        """Handle resource scaling events."""
        try:
            if self.cache_optimizer and resource_type.value == 'memory':
                # Trigger cache optimization when memory pressure increases
                await self.cache_optimizer.optimize_cache()
                return True
            return False
        except Exception as e:
            logger.error(f"Resource scaling handling failed: {str(e)}")
            return False
    
    def _get_components_status(self) -> Dict[str, str]:
        """Get status of all components."""
        try:
            status = {}
            
            components = {
                'unified_service': self.unified_service,
                'ensemble_ranking_bridge': self.ensemble_ranking_bridge,
                'experiment_manager': self.experiment_manager,
                'traffic_splitter': self.traffic_splitter,
                'performance_monitor': self.performance_monitor,
                'cache_optimizer': self.cache_optimizer,
                'query_optimizer': self.query_optimizer,
                'batch_processor': self.batch_processor,
                'resource_manager': self.resource_manager,
                'analytics_engine': self.analytics_engine,
                'model_monitor': self.model_monitor,
                'business_metrics': self.business_metrics,
                'dashboard_service': self.dashboard_service
            }
            
            for name, component in components.items():
                if component is not None:
                    status[name] = 'running'
                else:
                    status[name] = 'not_initialized'
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get components status: {str(e)}")
            return {}
    
    def _calculate_health_score(self) -> float:
        """Calculate overall health score for Enhanced ML Pipeline."""
        try:
            components_status = self._get_components_status()
            
            if not components_status:
                return 0.0
            
            running_components = sum(1 for status in components_status.values() if status == 'running')
            total_components = len(components_status)
            
            base_score = (running_components / total_components) * 100
            
            # Adjust based on critical components
            critical_components = self._get_components_for_mode()
            critical_running = sum(
                1 for name in critical_components 
                if components_status.get(name) == 'running'
            )
            
            if len(critical_components) > 0:
                critical_score = (critical_running / len(critical_components)) * 100
                # Weight critical components more heavily
                health_score = (base_score * 0.3) + (critical_score * 0.7)
            else:
                health_score = base_score
            
            return min(100.0, max(0.0, health_score))
            
        except Exception as e:
            logger.error(f"Health score calculation failed: {str(e)}")
            return 0.0
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of all Enhanced ML Pipeline components."""
        try:
            self.manager_stats['health_checks_performed'] += 1
            
            health_info = {
                'overall_status': 'healthy',
                'health_score': self._calculate_health_score(),
                'components': {},
                'timestamp': datetime.now().isoformat(),
                'mode': self.mode.value,
                'uptime_seconds': (datetime.now() - self.status.initialized_at).total_seconds() if self.status else 0
            }
            
            # Check individual components
            components_to_check = {
                'unified_service': self.unified_service,
                'analytics_engine': self.analytics_engine,
                'model_monitor': self.model_monitor,
                'business_metrics': self.business_metrics,
                'cache_optimizer': self.cache_optimizer,
                'resource_manager': self.resource_manager
            }
            
            for name, component in components_to_check.items():
                if component:
                    try:
                        # Get component-specific health info
                        if hasattr(component, 'get_service_status'):
                            comp_status = await component.get_service_status()
                        elif hasattr(component, 'get_analytics_summary'):
                            comp_status = await component.get_analytics_summary()
                        elif hasattr(component, 'get_cache_stats'):
                            comp_status = await component.get_cache_stats()
                        else:
                            comp_status = {'status': 'running'}
                        
                        health_info['components'][name] = {
                            'status': 'healthy',
                            'details': comp_status
                        }
                        
                    except Exception as comp_error:
                        health_info['components'][name] = {
                            'status': 'unhealthy',
                            'error': str(comp_error)
                        }
                        health_info['overall_status'] = 'degraded'
                else:
                    health_info['components'][name] = {
                        'status': 'not_running'
                    }
            
            # Update status if available
            if self.status:
                self.status.last_health_check = datetime.now()
                self.status.health_score = health_info['health_score']
                self.status.components_status = self._get_components_status()
            
            return health_info
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'overall_status': 'error',
                'health_score': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive Enhanced ML Pipeline status."""
        try:
            if not self.status:
                return {'error': 'Enhanced ML Pipeline not initialized'}
            
            return {
                'enhanced_ml_status': asdict(self.status),
                'manager_stats': self.manager_stats.copy(),
                'config': asdict(self.config),
                'components_info': {
                    name: component is not None
                    for name, component in {
                        'unified_service': self.unified_service,
                        'ensemble_ranking_bridge': self.ensemble_ranking_bridge,
                        'experiment_manager': self.experiment_manager,
                        'analytics_engine': self.analytics_engine,
                        'model_monitor': self.model_monitor,
                        'business_metrics': self.business_metrics,
                        'dashboard_service': self.dashboard_service
                    }.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get Enhanced ML Pipeline status: {str(e)}")
            return {'error': str(e)}
    
    async def restart_component(self, component_name: str) -> bool:
        """Restart a specific component."""
        try:
            logger.info(f"Restarting component: {component_name}")
            
            # Cleanup existing component
            component = getattr(self, component_name, None)
            if component and hasattr(component, 'cleanup'):
                await component.cleanup()
            
            # Reinitialize component
            success = await self._initialize_component(component_name)
            
            if success:
                self.manager_stats['component_restart_count'] += 1
                logger.info(f"Successfully restarted component: {component_name}")
            else:
                logger.error(f"Failed to restart component: {component_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Component restart failed for {component_name}: {str(e)}")
            return False
    
    async def shutdown(self):
        """Shutdown all Enhanced ML Pipeline components gracefully."""
        try:
            logger.info("Starting Enhanced ML Pipeline shutdown")
            
            # Shutdown components in reverse order
            shutdown_order = list(reversed(self.initialization_order))
            
            for component_name in shutdown_order:
                try:
                    component = getattr(self, component_name, None)
                    if component and hasattr(component, 'cleanup'):
                        await component.cleanup()
                        logger.info(f"Shutdown component: {component_name}")
                except Exception as comp_error:
                    logger.error(f"Component shutdown failed for {component_name}: {str(comp_error)}")
            
            # Reset state
            self.is_initialized = False
            self.status = None
            
            logger.info("Enhanced ML Pipeline shutdown completed")
            
        except Exception as e:
            logger.error(f"Enhanced ML Pipeline shutdown failed: {str(e)}")
    
    # Convenience methods for accessing components
    
    def get_unified_service(self) -> Optional[UnifiedMatchingService]:
        """Get unified matching service instance."""
        return self.unified_service
    
    def get_analytics_engine(self) -> Optional[AnalyticsEngine]:
        """Get analytics engine instance."""
        return self.analytics_engine
    
    def get_model_monitor(self) -> Optional[ModelPerformanceMonitor]:
        """Get model monitor instance."""
        return self.model_monitor
    
    def get_business_metrics(self) -> Optional[BusinessMetricsCollector]:
        """Get business metrics collector instance."""
        return self.business_metrics
    
    def get_dashboard_service(self) -> Optional[DashboardService]:
        """Get dashboard service instance."""
        return self.dashboard_service
    
    def get_experiment_manager(self) -> Optional[ExperimentManager]:
        """Get experiment manager instance."""
        return self.experiment_manager


# Global Enhanced ML Pipeline manager instance
_enhanced_ml_manager: Optional[EnhancedMLManager] = None


async def initialize_enhanced_ml(
    mode: EnhancedMLMode = EnhancedMLMode.FULL_PRODUCTION,
    config: Optional[EnhancedMLConfig] = None
) -> bool:
    """
    Initialize Enhanced ML Pipeline system globally.
    
    Args:
        mode: Enhanced ML Pipeline operation mode
        config: Component configuration
        
    Returns:
        True if initialization successful
    """
    global _enhanced_ml_manager
    
    try:
        _enhanced_ml_manager = EnhancedMLManager(mode=mode, config=config)
        success = await _enhanced_ml_manager.initialize()
        
        if success:
            logger.info("Global Enhanced ML Pipeline initialization successful")
        else:
            logger.error("Global Enhanced ML Pipeline initialization failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Global Enhanced ML Pipeline initialization error: {str(e)}")
        return False


def get_enhanced_ml_manager() -> Optional[EnhancedMLManager]:
    """Get global Enhanced ML Pipeline manager instance."""
    return _enhanced_ml_manager


async def shutdown_enhanced_ml():
    """Shutdown global Enhanced ML Pipeline system."""
    global _enhanced_ml_manager
    
    try:
        if _enhanced_ml_manager:
            await _enhanced_ml_manager.shutdown()
            _enhanced_ml_manager = None
            logger.info("Global Enhanced ML Pipeline shutdown completed")
        
    except Exception as e:
        logger.error(f"Global Enhanced ML Pipeline shutdown error: {str(e)}")


# Legacy aliases for backward compatibility
initialize_phase2 = initialize_enhanced_ml
get_phase2_manager = get_enhanced_ml_manager
shutdown_phase2 = shutdown_enhanced_ml
Phase2Manager = EnhancedMLManager
Phase2Mode = EnhancedMLMode
Phase2ComponentConfig = EnhancedMLConfig
Phase2Status = EnhancedMLStatus