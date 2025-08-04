"""
Traffic Splitter for A/B testing framework.

This module handles traffic allocation and user assignment to variants
with consistent hashing and advanced allocation strategies.
"""

import hashlib
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from app.utils.logger import logger
from app.core.ab_testing.model_variant import ModelVariant


class AllocationStrategy(str, Enum):
    """Traffic allocation strategies."""
    RANDOM = "random"
    HASH_BASED = "hash_based"
    WEIGHTED_RANDOM = "weighted_random"
    CONTEXTUAL = "contextual"
    ADAPTIVE = "adaptive"


@dataclass
class AllocationRule:
    """Rule for traffic allocation."""
    condition: str
    allocation: Dict[str, float]
    priority: int = 0


class TrafficSplitter:
    """
    Advanced traffic splitting system for A/B testing with consistent
    user assignment and configurable allocation strategies.
    """
    
    def __init__(self, default_strategy: AllocationStrategy = AllocationStrategy.HASH_BASED):
        """Initialize traffic splitter."""
        self.default_strategy = default_strategy
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.user_assignments: Dict[str, Dict[str, str]] = {}  # user_id -> {experiment_id: variant_id}
        self.allocation_rules: Dict[str, List[AllocationRule]] = {}
        
        # Performance tracking
        self.assignment_stats = {
            'total_assignments': 0,
            'assignments_by_experiment': {},
            'assignments_by_variant': {},
            'hash_collisions': 0,
            'context_overrides': 0
        }
        
        logger.info(f"TrafficSplitter initialized with strategy: {default_strategy}")
    
    async def register_experiment(
        self,
        experiment_id: str,
        variants: List[ModelVariant],
        traffic_allocation: Dict[str, float],
        strategy: Optional[AllocationStrategy] = None
    ) -> bool:
        """
        Register an experiment with traffic allocation.
        
        Args:
            experiment_id: Unique experiment identifier
            variants: List of model variants
            traffic_allocation: Traffic allocation percentages
            strategy: Allocation strategy to use
            
        Returns:
            True if registered successfully
        """
        try:
            if experiment_id in self.experiments:
                logger.warning(f"Experiment {experiment_id} already registered")
                return False
            
            # Validate allocation
            if not self._validate_allocation(traffic_allocation, [v.variant_id for v in variants]):
                raise ValueError("Invalid traffic allocation")
            
            # Store experiment configuration
            self.experiments[experiment_id] = {
                'variants': {v.variant_id: v for v in variants},
                'traffic_allocation': traffic_allocation.copy(),
                'strategy': strategy or self.default_strategy,
                'registered_at': datetime.now(),
                'total_assignments': 0,
                'variant_assignments': {v.variant_id: 0 for v in variants}
            }
            
            # Initialize allocation rules
            self.allocation_rules[experiment_id] = []
            
            # Initialize stats
            self.assignment_stats['assignments_by_experiment'][experiment_id] = 0
            for variant in variants:
                self.assignment_stats['assignments_by_variant'][variant.variant_id] = 0
            
            logger.info(f"Registered experiment {experiment_id} with {len(variants)} variants")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register experiment {experiment_id}: {str(e)}")
            return False
    
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
            Assigned variant ID or None if assignment fails
        """
        try:
            if experiment_id not in self.experiments:
                logger.warning(f"Experiment {experiment_id} not found")
                return None
            
            experiment = self.experiments[experiment_id]
            
            # Check for existing assignment (consistency)
            if user_id in self.user_assignments:
                existing_assignment = self.user_assignments[user_id].get(experiment_id)
                if existing_assignment:
                    logger.debug(f"User {user_id} already assigned to {existing_assignment}")
                    return existing_assignment
            
            # Apply allocation rules first
            variant_id = await self._apply_allocation_rules(experiment_id, user_id, context)
            
            if not variant_id:
                # Use configured strategy
                strategy = experiment['strategy']
                variant_id = await self._assign_by_strategy(
                    experiment_id, user_id, context, strategy
                )
            
            if variant_id:
                # Store assignment
                if user_id not in self.user_assignments:
                    self.user_assignments[user_id] = {}
                self.user_assignments[user_id][experiment_id] = variant_id
                
                # Update stats
                self.assignment_stats['total_assignments'] += 1
                self.assignment_stats['assignments_by_experiment'][experiment_id] += 1
                self.assignment_stats['assignments_by_variant'][variant_id] += 1
                
                experiment['total_assignments'] += 1
                experiment['variant_assignments'][variant_id] += 1
                
                logger.debug(f"Assigned user {user_id} to variant {variant_id}")
            
            return variant_id
            
        except Exception as e:
            logger.error(f"Variant assignment failed: {str(e)}")
            return None
    
    async def _assign_by_strategy(
        self,
        experiment_id: str,
        user_id: str,
        context: Optional[Dict[str, Any]],
        strategy: AllocationStrategy
    ) -> Optional[str]:
        """Assign variant using specified strategy."""
        experiment = self.experiments[experiment_id]
        traffic_allocation = experiment['traffic_allocation']
        variants = list(experiment['variants'].keys())
        
        try:
            if strategy == AllocationStrategy.HASH_BASED:
                return self._hash_based_assignment(experiment_id, user_id, traffic_allocation)
            
            elif strategy == AllocationStrategy.RANDOM:
                return self._random_assignment(traffic_allocation)
            
            elif strategy == AllocationStrategy.WEIGHTED_RANDOM:
                return self._weighted_random_assignment(traffic_allocation)
            
            elif strategy == AllocationStrategy.CONTEXTUAL:
                return self._contextual_assignment(experiment_id, user_id, context, traffic_allocation)
            
            elif strategy == AllocationStrategy.ADAPTIVE:
                return await self._adaptive_assignment(experiment_id, user_id, context, traffic_allocation)
            
            else:
                logger.error(f"Unknown allocation strategy: {strategy}")
                return self._hash_based_assignment(experiment_id, user_id, traffic_allocation)
                
        except Exception as e:
            logger.error(f"Strategy assignment failed: {str(e)}")
            return None
    
    def _hash_based_assignment(
        self,
        experiment_id: str,
        user_id: str,
        traffic_allocation: Dict[str, float]
    ) -> Optional[str]:
        """Assign variant using consistent hashing."""
        try:
            # Create hash input
            hash_input = f"{experiment_id}:{user_id}"
            hash_value = hashlib.md5(hash_input.encode()).hexdigest()
            
            # Convert to number between 0 and 1
            hash_number = int(hash_value[:8], 16) / (16**8)
            
            # Determine variant based on cumulative allocation
            cumulative = 0.0
            for variant_id, allocation in traffic_allocation.items():
                cumulative += allocation
                if hash_number <= cumulative:
                    return variant_id
            
            # Fallback to last variant
            return list(traffic_allocation.keys())[-1]
            
        except Exception as e:
            logger.error(f"Hash-based assignment failed: {str(e)}")
            return None
    
    def _random_assignment(self, traffic_allocation: Dict[str, float]) -> Optional[str]:
        """Assign variant using pure random selection."""
        try:
            random_value = random.random()
            
            cumulative = 0.0
            for variant_id, allocation in traffic_allocation.items():
                cumulative += allocation
                if random_value <= cumulative:
                    return variant_id
            
            return list(traffic_allocation.keys())[-1]
            
        except Exception as e:
            logger.error(f"Random assignment failed: {str(e)}")
            return None
    
    def _weighted_random_assignment(self, traffic_allocation: Dict[str, float]) -> Optional[str]:
        """Assign variant using weighted random selection."""
        try:
            variants = list(traffic_allocation.keys())
            weights = list(traffic_allocation.values())
            
            return random.choices(variants, weights=weights, k=1)[0]
            
        except Exception as e:
            logger.error(f"Weighted random assignment failed: {str(e)}")
            return None
    
    def _contextual_assignment(
        self,
        experiment_id: str,
        user_id: str,
        context: Optional[Dict[str, Any]],
        traffic_allocation: Dict[str, float]
    ) -> Optional[str]:
        """Assign variant based on context."""
        try:
            if not context:
                return self._hash_based_assignment(experiment_id, user_id, traffic_allocation)
            
            # Context-based assignment logic
            user_segment = context.get('user_segment', 'default')
            device_type = context.get('device_type', 'desktop')
            
            # Modify allocation based on context
            modified_allocation = traffic_allocation.copy()
            
            # Example: Mobile users get different allocation
            if device_type == 'mobile':
                # Boost treatment variants for mobile users
                for variant_id in modified_allocation:
                    if 'treatment' in variant_id or 'challenger' in variant_id:
                        modified_allocation[variant_id] *= 1.2
            
            # Example: Premium users get experimental features
            if user_segment == 'premium':
                for variant_id in modified_allocation:
                    if 'experimental' in variant_id:
                        modified_allocation[variant_id] *= 1.5
            
            # Renormalize allocation
            total = sum(modified_allocation.values())
            if total > 0:
                modified_allocation = {k: v/total for k, v in modified_allocation.items()}
            
            self.assignment_stats['context_overrides'] += 1
            
            return self._hash_based_assignment(experiment_id, user_id, modified_allocation)
            
        except Exception as e:
            logger.error(f"Contextual assignment failed: {str(e)}")
            return self._hash_based_assignment(experiment_id, user_id, traffic_allocation)
    
    async def _adaptive_assignment(
        self,
        experiment_id: str,
        user_id: str,
        context: Optional[Dict[str, Any]],
        traffic_allocation: Dict[str, float]
    ) -> Optional[str]:
        """Assign variant using adaptive allocation based on performance."""
        try:
            experiment = self.experiments[experiment_id]
            
            # Get current performance for each variant
            variant_performances = {}
            
            for variant_id, variant in experiment['variants'].items():
                performance_summary = variant.get_performance_summary()
                composite_score = performance_summary.get('composite_scores', {}).get('current', 0.0)
                variant_performances[variant_id] = composite_score
            
            # Adjust allocation based on performance (Thompson Sampling approach)
            if any(score > 0 for score in variant_performances.values()):
                # Calculate adaptive weights
                adaptive_weights = {}
                total_performance = sum(variant_performances.values())
                
                for variant_id, performance in variant_performances.items():
                    if total_performance > 0:
                        # Weight by performance, but maintain minimum allocation
                        min_allocation = 0.1  # Minimum 10% allocation
                        performance_weight = performance / total_performance
                        adaptive_weights[variant_id] = max(min_allocation, performance_weight)
                    else:
                        adaptive_weights[variant_id] = traffic_allocation[variant_id]
                
                # Normalize weights
                total_weight = sum(adaptive_weights.values())
                if total_weight > 0:
                    adaptive_allocation = {k: v/total_weight for k, v in adaptive_weights.items()}
                else:
                    adaptive_allocation = traffic_allocation
                
                return self._hash_based_assignment(experiment_id, user_id, adaptive_allocation)
            
            # Fallback to original allocation if no performance data
            return self._hash_based_assignment(experiment_id, user_id, traffic_allocation)
            
        except Exception as e:
            logger.error(f"Adaptive assignment failed: {str(e)}")
            return self._hash_based_assignment(experiment_id, user_id, traffic_allocation)
    
    async def _apply_allocation_rules(
        self,
        experiment_id: str,
        user_id: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Apply custom allocation rules."""
        try:
            if experiment_id not in self.allocation_rules:
                return None
            
            rules = sorted(self.allocation_rules[experiment_id], key=lambda r: r.priority, reverse=True)
            
            for rule in rules:
                if self._evaluate_rule_condition(rule.condition, user_id, context):
                    # Apply rule allocation
                    return self._hash_based_assignment(experiment_id, user_id, rule.allocation)
            
            return None
            
        except Exception as e:
            logger.error(f"Allocation rule application failed: {str(e)}")
            return None
    
    def _evaluate_rule_condition(
        self,
        condition: str,
        user_id: str,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Evaluate allocation rule condition."""
        try:
            if not context:
                return False
            
            # Simple condition evaluation (would be more sophisticated in practice)
            if 'user_segment=' in condition:
                expected_segment = condition.split('user_segment=')[1].strip()
                actual_segment = context.get('user_segment', '')
                return actual_segment == expected_segment
            
            if 'device_type=' in condition:
                expected_device = condition.split('device_type=')[1].strip()
                actual_device = context.get('device_type', '')
                return actual_device == expected_device
            
            if 'user_id_hash_mod=' in condition:
                # Rule based on user ID hash modulo
                parts = condition.split('user_id_hash_mod=')[1].split(',')
                mod_value = int(parts[0])
                expected_remainder = int(parts[1])
                
                user_hash = hashlib.md5(user_id.encode()).hexdigest()
                hash_number = int(user_hash[:8], 16)
                actual_remainder = hash_number % mod_value
                
                return actual_remainder == expected_remainder
            
            return False
            
        except Exception as e:
            logger.error(f"Rule condition evaluation failed: {str(e)}")
            return False
    
    def add_allocation_rule(
        self,
        experiment_id: str,
        condition: str,
        allocation: Dict[str, float],
        priority: int = 0
    ) -> bool:
        """Add custom allocation rule."""
        try:
            if experiment_id not in self.experiments:
                logger.error(f"Experiment {experiment_id} not found")
                return False
            
            # Validate allocation
            variants = list(self.experiments[experiment_id]['variants'].keys())
            if not self._validate_allocation(allocation, variants):
                logger.error("Invalid rule allocation")
                return False
            
            # Add rule
            rule = AllocationRule(condition=condition, allocation=allocation, priority=priority)
            
            if experiment_id not in self.allocation_rules:
                self.allocation_rules[experiment_id] = []
            
            self.allocation_rules[experiment_id].append(rule)
            
            logger.info(f"Added allocation rule for experiment {experiment_id}: {condition}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add allocation rule: {str(e)}")
            return False
    
    async def update_traffic_allocation(
        self,
        experiment_id: str,
        new_allocation: Dict[str, float]
    ) -> bool:
        """Update traffic allocation for an experiment."""
        try:
            if experiment_id not in self.experiments:
                logger.error(f"Experiment {experiment_id} not found")
                return False
            
            experiment = self.experiments[experiment_id]
            variants = list(experiment['variants'].keys())
            
            # Validate new allocation
            if not self._validate_allocation(new_allocation, variants):
                logger.error("Invalid new allocation")
                return False
            
            # Update allocation
            experiment['traffic_allocation'] = new_allocation.copy()
            
            logger.info(f"Updated traffic allocation for experiment {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update traffic allocation: {str(e)}")
            return False
    
    async def get_user_assignments(self, user_id: str) -> Dict[str, str]:
        """Get all experiment assignments for a user."""
        return self.user_assignments.get(user_id, {})
    
    async def get_experiment_assignments(self, experiment_id: str) -> Dict[str, int]:
        """Get assignment counts for an experiment."""
        try:
            if experiment_id not in self.experiments:
                return {}
            
            return self.experiments[experiment_id]['variant_assignments'].copy()
            
        except Exception as e:
            logger.error(f"Failed to get experiment assignments: {str(e)}")
            return {}
    
    async def remove_experiment(self, experiment_id: str) -> bool:
        """Remove an experiment and clean up assignments."""
        try:
            if experiment_id not in self.experiments:
                return True  # Already removed
            
            # Remove experiment
            del self.experiments[experiment_id]
            
            # Clean up allocation rules
            if experiment_id in self.allocation_rules:
                del self.allocation_rules[experiment_id]
            
            # Clean up user assignments
            for user_id in list(self.user_assignments.keys()):
                if experiment_id in self.user_assignments[user_id]:
                    del self.user_assignments[user_id][experiment_id]
                    
                    # Remove user entry if no more assignments
                    if not self.user_assignments[user_id]:
                        del self.user_assignments[user_id]
            
            # Clean up stats
            if experiment_id in self.assignment_stats['assignments_by_experiment']:
                del self.assignment_stats['assignments_by_experiment'][experiment_id]
            
            logger.info(f"Removed experiment {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove experiment {experiment_id}: {str(e)}")
            return False
    
    def _validate_allocation(self, allocation: Dict[str, float], variants: List[str]) -> bool:
        """Validate traffic allocation."""
        try:
            # Check if all variants have allocation
            if set(allocation.keys()) != set(variants):
                return False
            
            # Check if allocations sum to approximately 1.0
            total = sum(allocation.values())
            if abs(total - 1.0) > 0.001:
                return False
            
            # Check if all allocations are non-negative
            if any(alloc < 0 for alloc in allocation.values()):
                return False
            
            return True
            
        except Exception:
            return False
    
    async def get_splitter_stats(self) -> Dict[str, Any]:
        """Get comprehensive traffic splitter statistics."""
        try:
            # Calculate allocation balance for each experiment
            experiment_balance = {}
            
            for experiment_id, experiment in self.experiments.items():
                expected_allocation = experiment['traffic_allocation']
                actual_assignments = experiment['variant_assignments']
                total_assignments = sum(actual_assignments.values())
                
                if total_assignments > 0:
                    actual_allocation = {
                        variant: count / total_assignments 
                        for variant, count in actual_assignments.items()
                    }
                    
                    # Calculate allocation deviation
                    deviation = {}
                    for variant in expected_allocation:
                        expected = expected_allocation[variant]
                        actual = actual_allocation.get(variant, 0.0)
                        deviation[variant] = abs(actual - expected)
                    
                    experiment_balance[experiment_id] = {
                        'expected': expected_allocation,
                        'actual': actual_allocation,
                        'deviation': deviation,
                        'max_deviation': max(deviation.values()) if deviation else 0.0
                    }
            
            return {
                'assignment_stats': self.assignment_stats.copy(),
                'active_experiments': list(self.experiments.keys()),
                'total_users_assigned': len(self.user_assignments),
                'experiment_balance': experiment_balance,
                'allocation_rules_count': {
                    exp_id: len(rules) for exp_id, rules in self.allocation_rules.items()
                },
                'default_strategy': self.default_strategy.value
            }
            
        except Exception as e:
            logger.error(f"Failed to get splitter stats: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Clean up traffic splitter resources."""
        try:
            # Clear all data
            self.experiments.clear()
            self.user_assignments.clear()
            self.allocation_rules.clear()
            
            # Reset stats
            self.assignment_stats = {
                'total_assignments': 0,
                'assignments_by_experiment': {},
                'assignments_by_variant': {},
                'hash_collisions': 0,
                'context_overrides': 0
            }
            
            logger.info("TrafficSplitter cleanup completed")
            
        except Exception as e:
            logger.error(f"TrafficSplitter cleanup failed: {str(e)}")