"""
Batch Processor for efficient bulk operations and resource optimization.

This module provides intelligent batching capabilities for database operations,
API calls, ML predictions, and other resource-intensive tasks.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import heapq

from app.utils.logger import logger


class BatchStrategy(str, Enum):
    """Batch processing strategies."""
    SIZE_BASED = "size_based"           # Batch by size
    TIME_BASED = "time_based"           # Batch by time window
    ADAPTIVE = "adaptive"               # Adaptive based on load
    PRIORITY_BASED = "priority_based"   # Batch by priority
    RESOURCE_BASED = "resource_based"   # Batch by resource availability


class BatchStatus(str, Enum):
    """Batch processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchItem:
    """Individual item in a batch."""
    id: str
    data: Any
    priority: int = 0
    created_at: datetime = None
    context: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_id: str
    strategy: BatchStrategy
    max_batch_size: int = 100
    max_wait_time_seconds: float = 5.0
    min_batch_size: int = 1
    priority_threshold: int = 5
    retry_attempts: int = 3
    timeout_seconds: float = 300.0
    processor_function: Optional[Callable] = None


@dataclass
class BatchResult:
    """Result of batch processing."""
    batch_id: str
    item_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class BatchExecution:
    """Batch execution metadata."""
    batch_id: str
    execution_id: str
    items: List[BatchItem]
    status: BatchStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time_ms: float = 0.0
    success_count: int = 0
    error_count: int = 0
    results: List[BatchResult] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = []


class BatchProcessor:
    """
    Advanced batch processor for efficient bulk operations with
    multiple processing strategies and adaptive optimization.
    """
    
    def __init__(
        self,
        default_strategy: BatchStrategy = BatchStrategy.ADAPTIVE,
        max_concurrent_batches: int = 10,
        enable_metrics: bool = True
    ):
        """Initialize batch processor."""
        self.default_strategy = default_strategy
        self.max_concurrent_batches = max_concurrent_batches
        self.enable_metrics = enable_metrics
        
        # Batch configurations
        self.batch_configs: Dict[str, BatchConfig] = {}
        
        # Pending items by batch type
        self.pending_items: Dict[str, List[BatchItem]] = defaultdict(list)
        self.priority_queues: Dict[str, List[BatchItem]] = defaultdict(list)
        
        # Active batch executions
        self.active_executions: Dict[str, BatchExecution] = {}
        self.execution_history: List[BatchExecution] = []
        
        # Batch processing tasks
        self.batch_tasks: Dict[str, asyncio.Task] = {}
        self.scheduler_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self.metrics = {
            'total_items_processed': 0,
            'total_batches_executed': 0,
            'average_batch_size': 0.0,
            'average_processing_time_ms': 0.0,
            'success_rate': 0.0,
            'throughput_items_per_second': 0.0,
            'queue_depths': defaultdict(int),
            'strategy_performance': defaultdict(lambda: {'count': 0, 'avg_time': 0.0})
        }
        
        # Adaptive parameters
        self.adaptive_config = {
            'load_threshold_high': 0.8,
            'load_threshold_low': 0.2,
            'size_adjustment_factor': 0.1,
            'time_adjustment_factor': 0.1,
            'performance_window_size': 100
        }
        
        logger.info(f"BatchProcessor initialized with {default_strategy.value} strategy")
    
    async def initialize(self):
        """Initialize batch processor with scheduler."""
        try:
            # Start batch scheduler
            self.scheduler_task = asyncio.create_task(self._batch_scheduler())
            
            logger.info("BatchProcessor scheduler started")
            
        except Exception as e:
            logger.error(f"BatchProcessor initialization failed: {str(e)}")
            raise
    
    async def register_batch_type(
        self,
        batch_id: str,
        processor_function: Callable,
        config: Optional[BatchConfig] = None
    ) -> bool:
        """
        Register a new batch type with processing function.
        
        Args:
            batch_id: Unique identifier for batch type
            processor_function: Function to process batch items
            config: Batch configuration
            
        Returns:
            True if registered successfully
        """
        try:
            if config is None:
                config = BatchConfig(
                    batch_id=batch_id,
                    strategy=self.default_strategy,
                    processor_function=processor_function
                )
            else:
                config.batch_id = batch_id
                config.processor_function = processor_function
            
            self.batch_configs[batch_id] = config
            
            # Initialize data structures
            if batch_id not in self.pending_items:
                self.pending_items[batch_id] = []
            if batch_id not in self.priority_queues:
                self.priority_queues[batch_id] = []
            
            logger.info(f"Registered batch type: {batch_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register batch type {batch_id}: {str(e)}")
            return False
    
    async def add_item(
        self,
        batch_id: str,
        item_data: Any,
        priority: int = 0,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add item to batch queue.
        
        Args:
            batch_id: Batch type identifier
            item_data: Data to be processed
            priority: Item priority (higher = more urgent)
            context: Additional context for processing
            
        Returns:
            Item ID for tracking
        """
        try:
            if batch_id not in self.batch_configs:
                raise ValueError(f"Batch type {batch_id} not registered")
            
            # Generate item ID
            item_id = f"{batch_id}_{int(time.time() * 1000)}_{len(self.pending_items[batch_id])}"
            
            # Create batch item
            item = BatchItem(
                id=item_id,
                data=item_data,
                priority=priority,
                context=context
            )
            
            # Add to appropriate queue
            config = self.batch_configs[batch_id]
            
            if config.strategy == BatchStrategy.PRIORITY_BASED:
                heapq.heappush(self.priority_queues[batch_id], (-priority, item))
            else:
                self.pending_items[batch_id].append(item)
            
            # Update queue depth metric
            self.metrics['queue_depths'][batch_id] += 1
            
            logger.debug(f"Added item {item_id} to batch {batch_id}")
            
            return item_id
            
        except Exception as e:
            logger.error(f"Failed to add item to batch {batch_id}: {str(e)}")
            raise
    
    async def process_batch_immediately(
        self,
        batch_id: str,
        items: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[BatchResult]:
        """
        Process a batch immediately without queuing.
        
        Args:
            batch_id: Batch type identifier
            items: Items to process
            context: Processing context
            
        Returns:
            List of batch results
        """
        try:
            if batch_id not in self.batch_configs:
                raise ValueError(f"Batch type {batch_id} not registered")
            
            config = self.batch_configs[batch_id]
            
            # Create batch items
            batch_items = []
            for i, item_data in enumerate(items):
                item = BatchItem(
                    id=f"{batch_id}_immediate_{int(time.time() * 1000)}_{i}",
                    data=item_data,
                    context=context
                )
                batch_items.append(item)
            
            # Execute batch
            execution = await self._execute_batch(config, batch_items)
            
            return execution.results
            
        except Exception as e:
            logger.error(f"Immediate batch processing failed for {batch_id}: {str(e)}")
            return []
    
    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get status information for a batch type."""
        try:
            if batch_id not in self.batch_configs:
                return {'error': 'Batch type not found'}
            
            config = self.batch_configs[batch_id]
            
            # Count pending items
            pending_count = len(self.pending_items[batch_id])
            priority_count = len(self.priority_queues[batch_id])
            
            # Find active executions
            active_executions = [
                exec for exec in self.active_executions.values()
                if exec.batch_id == batch_id
            ]
            
            # Calculate recent performance
            recent_executions = [
                exec for exec in self.execution_history[-50:]
                if exec.batch_id == batch_id and exec.status == BatchStatus.COMPLETED
            ]
            
            avg_processing_time = 0.0
            avg_batch_size = 0.0
            success_rate = 0.0
            
            if recent_executions:
                avg_processing_time = sum(exec.processing_time_ms for exec in recent_executions) / len(recent_executions)
                avg_batch_size = sum(len(exec.items) for exec in recent_executions) / len(recent_executions)
                
                total_items = sum(len(exec.items) for exec in recent_executions)
                successful_items = sum(exec.success_count for exec in recent_executions)
                success_rate = successful_items / total_items if total_items > 0 else 0.0
            
            return {
                'batch_id': batch_id,
                'config': {
                    'strategy': config.strategy.value,
                    'max_batch_size': config.max_batch_size,
                    'max_wait_time_seconds': config.max_wait_time_seconds,
                    'min_batch_size': config.min_batch_size
                },
                'queue_status': {
                    'pending_items': pending_count,
                    'priority_items': priority_count,
                    'active_executions': len(active_executions)
                },
                'recent_performance': {
                    'average_processing_time_ms': avg_processing_time,
                    'average_batch_size': avg_batch_size,
                    'success_rate': success_rate,
                    'recent_batches': len(recent_executions)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get batch status for {batch_id}: {str(e)}")
            return {'error': str(e)}
    
    async def _batch_scheduler(self):
        """Background scheduler for batch processing."""
        try:
            while True:
                await asyncio.sleep(0.1)  # Check every 100ms
                
                # Process each batch type
                for batch_id, config in self.batch_configs.items():
                    try:
                        # Check if we should create a new batch
                        if await self._should_create_batch(batch_id, config):
                            # Limit concurrent batches
                            if len(self.active_executions) < self.max_concurrent_batches:
                                await self._create_and_execute_batch(batch_id, config)
                    
                    except Exception as e:
                        logger.error(f"Scheduler error for batch {batch_id}: {str(e)}")
                
                # Clean up completed executions
                await self._cleanup_completed_executions()
                
                # Update metrics
                if self.enable_metrics:
                    await self._update_metrics()
                
        except asyncio.CancelledError:
            logger.info("Batch scheduler cancelled")
        except Exception as e:
            logger.error(f"Batch scheduler error: {str(e)}")
    
    async def _should_create_batch(self, batch_id: str, config: BatchConfig) -> bool:
        """Determine if a batch should be created."""
        try:
            pending_items = self.pending_items[batch_id]
            priority_items = self.priority_queues[batch_id]
            
            total_items = len(pending_items) + len(priority_items)
            
            if total_items == 0:
                return False
            
            # Strategy-based decisions
            if config.strategy == BatchStrategy.SIZE_BASED:
                return total_items >= config.max_batch_size
            
            elif config.strategy == BatchStrategy.TIME_BASED:
                if total_items >= config.min_batch_size:
                    oldest_item_time = None
                    
                    if pending_items:
                        oldest_item_time = pending_items[0].created_at
                    
                    if priority_items:
                        priority_oldest = priority_items[0][1].created_at
                        if oldest_item_time is None or priority_oldest < oldest_item_time:
                            oldest_item_time = priority_oldest
                    
                    if oldest_item_time:
                        wait_time = (datetime.now() - oldest_item_time).total_seconds()
                        return wait_time >= config.max_wait_time_seconds
            
            elif config.strategy == BatchStrategy.PRIORITY_BASED:
                # Process if we have high priority items or enough regular items
                if priority_items:
                    highest_priority = -priority_items[0][0]
                    if highest_priority >= config.priority_threshold:
                        return True
                
                return total_items >= config.max_batch_size
            
            elif config.strategy == BatchStrategy.ADAPTIVE:
                return await self._adaptive_batch_decision(batch_id, config, total_items)
            
            elif config.strategy == BatchStrategy.RESOURCE_BASED:
                # Check system load and resource availability
                current_load = len(self.active_executions) / self.max_concurrent_batches
                
                if current_load < self.adaptive_config['load_threshold_low']:
                    return total_items >= config.min_batch_size
                elif current_load < self.adaptive_config['load_threshold_high']:
                    return total_items >= config.max_batch_size // 2
                else:
                    return total_items >= config.max_batch_size
            
            return False
            
        except Exception as e:
            logger.error(f"Batch decision error for {batch_id}: {str(e)}")
            return False
    
    async def _adaptive_batch_decision(
        self,
        batch_id: str,
        config: BatchConfig,
        total_items: int
    ) -> bool:
        """Make adaptive batch creation decision."""
        try:
            # Get recent performance data
            recent_executions = [
                exec for exec in self.execution_history[-self.adaptive_config['performance_window_size']:]
                if exec.batch_id == batch_id and exec.status == BatchStatus.COMPLETED
            ]
            
            if not recent_executions:
                # No history, use default behavior
                return total_items >= config.max_batch_size
            
            # Calculate performance metrics
            avg_processing_time = sum(exec.processing_time_ms for exec in recent_executions) / len(recent_executions)
            avg_batch_size = sum(len(exec.items) for exec in recent_executions) / len(recent_executions)
            
            # Calculate efficiency (items per second)
            avg_efficiency = (avg_batch_size / (avg_processing_time / 1000)) if avg_processing_time > 0 else 1.0
            
            # Adaptive thresholds
            current_load = len(self.active_executions) / self.max_concurrent_batches
            
            if current_load < self.adaptive_config['load_threshold_low']:
                # Low load: process smaller batches for lower latency
                adaptive_size = max(config.min_batch_size, int(avg_batch_size * 0.7))
            elif current_load > self.adaptive_config['load_threshold_high']:
                # High load: wait for larger batches for better efficiency
                adaptive_size = min(config.max_batch_size, int(avg_batch_size * 1.3))
            else:
                # Normal load: use historical average
                adaptive_size = int(avg_batch_size)
            
            # Time-based component
            if total_items >= config.min_batch_size:
                pending_items = self.pending_items[batch_id]
                if pending_items:
                    wait_time = (datetime.now() - pending_items[0].created_at).total_seconds()
                    adaptive_wait_time = config.max_wait_time_seconds * (1.0 + current_load)
                    
                    if wait_time >= adaptive_wait_time:
                        return True
            
            return total_items >= adaptive_size
            
        except Exception as e:
            logger.error(f"Adaptive batch decision error: {str(e)}")
            return total_items >= config.max_batch_size
    
    async def _create_and_execute_batch(self, batch_id: str, config: BatchConfig):
        """Create and execute a batch."""
        try:
            # Collect items for batch
            batch_items = []
            
            # Priority items first
            while (self.priority_queues[batch_id] and 
                   len(batch_items) < config.max_batch_size):
                priority, item = heapq.heappop(self.priority_queues[batch_id])
                batch_items.append(item)
                self.metrics['queue_depths'][batch_id] -= 1
            
            # Regular items
            while (self.pending_items[batch_id] and 
                   len(batch_items) < config.max_batch_size):
                item = self.pending_items[batch_id].pop(0)
                batch_items.append(item)
                self.metrics['queue_depths'][batch_id] -= 1
            
            if not batch_items:
                return
            
            # Create execution task
            execution_id = f"{batch_id}_{int(time.time() * 1000)}"
            execution_task = asyncio.create_task(
                self._execute_batch_with_tracking(config, batch_items, execution_id)
            )
            
            self.batch_tasks[execution_id] = execution_task
            
            logger.debug(f"Created batch execution {execution_id} with {len(batch_items)} items")
            
        except Exception as e:
            logger.error(f"Batch creation failed for {batch_id}: {str(e)}")
    
    async def _execute_batch_with_tracking(
        self,
        config: BatchConfig,
        batch_items: List[BatchItem],
        execution_id: str
    ):
        """Execute batch with full tracking."""
        try:
            # Create execution record
            execution = BatchExecution(
                batch_id=config.batch_id,
                execution_id=execution_id,
                items=batch_items,
                status=BatchStatus.PENDING,
                created_at=datetime.now()
            )
            
            self.active_executions[execution_id] = execution
            
            # Execute batch
            final_execution = await self._execute_batch(config, batch_items, execution)
            
            # Move to history
            self.execution_history.append(final_execution)
            
            # Remove from active
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            
            # Clean up task
            if execution_id in self.batch_tasks:
                del self.batch_tasks[execution_id]
            
        except Exception as e:
            logger.error(f"Batch execution tracking failed: {str(e)}")
    
    async def _execute_batch(
        self,
        config: BatchConfig,
        batch_items: List[BatchItem],
        execution: Optional[BatchExecution] = None
    ) -> BatchExecution:
        """Execute a batch of items."""
        try:
            start_time = time.time()
            
            if execution is None:
                execution = BatchExecution(
                    batch_id=config.batch_id,
                    execution_id=f"immediate_{int(time.time() * 1000)}",
                    items=batch_items,
                    status=BatchStatus.PENDING,
                    created_at=datetime.now()
                )
            
            execution.status = BatchStatus.PROCESSING
            execution.started_at = datetime.now()
            
            if not config.processor_function:
                raise ValueError(f"No processor function for batch {config.batch_id}")
            
            # Prepare items for processing
            items_data = [item.data for item in batch_items]
            context = {
                'batch_id': config.batch_id,
                'execution_id': execution.execution_id,
                'item_contexts': [item.context for item in batch_items]
            }
            
            # Execute processing function
            try:
                if asyncio.iscoroutinefunction(config.processor_function):
                    results = await config.processor_function(items_data, context)
                else:
                    results = config.processor_function(items_data, context)
                
                # Handle different result formats
                if not isinstance(results, list):
                    results = [results] * len(batch_items)
                
                # Ensure we have results for all items
                while len(results) < len(batch_items):
                    results.append(None)
                
                # Create batch results
                for i, (item, result) in enumerate(zip(batch_items, results)):
                    batch_result = BatchResult(
                        batch_id=config.batch_id,
                        item_id=item.id,
                        success=True,
                        result=result,
                        processing_time_ms=(time.time() - start_time) * 1000
                    )
                    execution.results.append(batch_result)
                    execution.success_count += 1
                
                execution.status = BatchStatus.COMPLETED
                
            except Exception as proc_error:
                logger.error(f"Batch processing function failed: {str(proc_error)}")
                
                # Create error results for all items
                for item in batch_items:
                    batch_result = BatchResult(
                        batch_id=config.batch_id,
                        item_id=item.id,
                        success=False,
                        error=str(proc_error),
                        processing_time_ms=(time.time() - start_time) * 1000
                    )
                    execution.results.append(batch_result)
                    execution.error_count += 1
                
                execution.status = BatchStatus.FAILED
            
            execution.completed_at = datetime.now()
            execution.processing_time_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self.metrics['total_items_processed'] += len(batch_items)
            self.metrics['total_batches_executed'] += 1
            
            strategy_stats = self.metrics['strategy_performance'][config.strategy.value]
            strategy_stats['count'] += 1
            strategy_stats['avg_time'] = (
                (strategy_stats['avg_time'] * (strategy_stats['count'] - 1) + 
                 execution.processing_time_ms) / strategy_stats['count']
            )
            
            logger.info(
                f"Batch {execution.execution_id} completed: "
                f"{execution.success_count} success, {execution.error_count} errors, "
                f"{execution.processing_time_ms:.2f}ms"
            )
            
            return execution
            
        except Exception as e:
            logger.error(f"Batch execution failed: {str(e)}")
            
            if execution:
                execution.status = BatchStatus.FAILED
                execution.completed_at = datetime.now()
                execution.processing_time_ms = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0.0
            
            return execution
    
    async def _cleanup_completed_executions(self):
        """Clean up old execution history."""
        try:
            # Keep only recent executions
            max_history = 1000
            if len(self.execution_history) > max_history:
                self.execution_history = self.execution_history[-max_history:]
            
            # Clean up old batch tasks
            completed_tasks = [
                exec_id for exec_id, task in self.batch_tasks.items()
                if task.done()
            ]
            
            for exec_id in completed_tasks:
                del self.batch_tasks[exec_id]
            
        except Exception as e:
            logger.error(f"Execution cleanup failed: {str(e)}")
    
    async def _update_metrics(self):
        """Update performance metrics."""
        try:
            if not self.execution_history:
                return
            
            # Calculate averages from recent history
            recent_executions = self.execution_history[-100:]
            
            if recent_executions:
                total_items = sum(len(exec.items) for exec in recent_executions)
                total_time = sum(exec.processing_time_ms for exec in recent_executions)
                successful_items = sum(exec.success_count for exec in recent_executions)
                
                self.metrics['average_batch_size'] = total_items / len(recent_executions)
                self.metrics['average_processing_time_ms'] = total_time / len(recent_executions)
                self.metrics['success_rate'] = successful_items / total_items if total_items > 0 else 0.0
                
                # Calculate throughput (items per second)
                if total_time > 0:
                    self.metrics['throughput_items_per_second'] = (total_items / (total_time / 1000))
                else:
                    self.metrics['throughput_items_per_second'] = 0.0
            
        except Exception as e:
            logger.error(f"Metrics update failed: {str(e)}")
    
    async def get_processor_stats(self) -> Dict[str, Any]:
        """Get comprehensive processor statistics."""
        try:
            await self._update_metrics()
            
            return {
                'performance_metrics': self.metrics.copy(),
                'active_batches': {
                    exec_id: {
                        'batch_id': exec.batch_id,
                        'status': exec.status.value,
                        'item_count': len(exec.items),
                        'started_at': exec.started_at.isoformat() if exec.started_at else None
                    }
                    for exec_id, exec in self.active_executions.items()
                },
                'queue_status': {
                    batch_id: {
                        'pending_items': len(self.pending_items[batch_id]),
                        'priority_items': len(self.priority_queues[batch_id])
                    }
                    for batch_id in self.batch_configs.keys()
                },
                'configuration': {
                    'default_strategy': self.default_strategy.value,
                    'max_concurrent_batches': self.max_concurrent_batches,
                    'batch_types': list(self.batch_configs.keys())
                },
                'recent_performance': {
                    exec.batch_id: {
                        'execution_count': len([e for e in self.execution_history[-50:] if e.batch_id == exec.batch_id]),
                        'avg_processing_time': sum(e.processing_time_ms for e in self.execution_history[-50:] if e.batch_id == exec.batch_id) / 
                                             max(1, len([e for e in self.execution_history[-50:] if e.batch_id == exec.batch_id]))
                    }
                    for exec in self.execution_history[-50:]
                } if self.execution_history else {}
            }
            
        except Exception as e:
            logger.error(f"Failed to get processor stats: {str(e)}")
            return {}
    
    async def pause_batch_type(self, batch_id: str) -> bool:
        """Pause processing for a specific batch type."""
        try:
            if batch_id not in self.batch_configs:
                return False
            
            # Mark config as paused (implement pause logic)
            # For now, we'll just log the action
            logger.info(f"Paused batch type: {batch_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause batch type {batch_id}: {str(e)}")
            return False
    
    async def resume_batch_type(self, batch_id: str) -> bool:
        """Resume processing for a specific batch type."""
        try:
            if batch_id not in self.batch_configs:
                return False
            
            # Mark config as resumed (implement resume logic)
            logger.info(f"Resumed batch type: {batch_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume batch type {batch_id}: {str(e)}")
            return False
    
    async def clear_queue(self, batch_id: str) -> int:
        """Clear pending items for a batch type."""
        try:
            if batch_id not in self.batch_configs:
                return 0
            
            cleared_count = len(self.pending_items[batch_id]) + len(self.priority_queues[batch_id])
            
            self.pending_items[batch_id].clear()
            self.priority_queues[batch_id].clear()
            self.metrics['queue_depths'][batch_id] = 0
            
            logger.info(f"Cleared {cleared_count} items from batch {batch_id}")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Failed to clear queue for {batch_id}: {str(e)}")
            return 0
    
    async def cleanup(self):
        """Clean up batch processor resources."""
        try:
            # Cancel scheduler
            if self.scheduler_task:
                self.scheduler_task.cancel()
                try:
                    await self.scheduler_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel all batch tasks
            for task in self.batch_tasks.values():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Clear data structures
            self.batch_configs.clear()
            self.pending_items.clear()
            self.priority_queues.clear()
            self.active_executions.clear()
            self.execution_history.clear()
            self.batch_tasks.clear()
            
            logger.info("BatchProcessor cleanup completed")
            
        except Exception as e:
            logger.error(f"BatchProcessor cleanup failed: {str(e)}")