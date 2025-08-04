"""
Cache Optimizer for advanced caching strategies and performance optimization.

This module implements intelligent caching with TTL management, cache warming,
invalidation strategies, and performance monitoring.
"""

import asyncio
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import zlib

from app.utils.logger import logger
from app.core.constants.ml_constants import PerformanceMetrics


class CacheStrategy(str, Enum):
    """Cache replacement strategies."""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    FIFO = "fifo"            # First In First Out
    TTL = "ttl"              # Time To Live
    ADAPTIVE = "adaptive"     # Adaptive based on usage patterns


class CacheScope(str, Enum):
    """Cache scope definitions."""
    GLOBAL = "global"         # Global cache
    USER = "user"            # User-specific cache
    SESSION = "session"      # Session-specific cache
    EXPERIMENT = "experiment" # Experiment-specific cache


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int]
    size_bytes: int
    scope: CacheScope
    tags: List[str]
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        
        expiry_time = self.created_at + timedelta(seconds=self.ttl_seconds)
        return datetime.now() > expiry_time
    
    def update_access(self):
        """Update access metadata."""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_hits: int
    total_misses: int
    hit_ratio: float
    total_entries: int
    total_size_bytes: int
    average_entry_size: float
    eviction_count: int
    invalidation_count: int


class CacheOptimizer:
    """
    Advanced cache optimizer with intelligent caching strategies,
    performance monitoring, and automatic optimization.
    """
    
    def __init__(
        self,
        max_size_mb: int = 500,
        default_ttl_seconds: int = 3600,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        enable_compression: bool = True
    ):
        """Initialize cache optimizer."""
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl_seconds = default_ttl_seconds
        self.strategy = strategy
        self.enable_compression = enable_compression
        
        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_by_scope: Dict[CacheScope, Dict[str, str]] = {
            scope: {} for scope in CacheScope
        }
        self.cache_by_tags: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.stats = CacheStats(
            total_hits=0,
            total_misses=0,
            hit_ratio=0.0,
            total_entries=0,
            total_size_bytes=0,
            average_entry_size=0.0,
            eviction_count=0,
            invalidation_count=0
        )
        
        # Cache warming configurations
        self.warming_configs: Dict[str, Dict[str, Any]] = {}
        self.warming_tasks: Dict[str, asyncio.Task] = {}
        
        # Optimization parameters
        self.optimization_config = {
            'min_access_count_for_warming': 5,
            'warming_batch_size': 10,
            'cleanup_interval_seconds': 300,
            'stats_update_interval_seconds': 60,
            'adaptive_threshold_adjustment': 0.1
        }
        
        # Start background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.stats_task: Optional[asyncio.Task] = None
        
        logger.info(f"CacheOptimizer initialized with {strategy.value} strategy, max size: {max_size_mb}MB")
    
    async def initialize(self):
        """Initialize cache optimizer with background tasks."""
        try:
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            # Start stats update task
            self.stats_task = asyncio.create_task(self._stats_update_loop())
            
            logger.info("CacheOptimizer background tasks started")
            
        except Exception as e:
            logger.error(f"CacheOptimizer initialization failed: {str(e)}")
            raise
    
    async def get(
        self,
        key: str,
        scope: CacheScope = CacheScope.GLOBAL,
        default: Any = None
    ) -> Any:
        """Get value from cache."""
        try:
            cache_key = self._build_cache_key(key, scope)
            
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                # Check expiration
                if entry.is_expired():
                    await self._remove_entry(cache_key)
                    self.stats.total_misses += 1
                    return default
                
                # Update access
                entry.update_access()
                self.stats.total_hits += 1
                
                # Decompress if needed
                value = entry.value
                if self.enable_compression and isinstance(value, bytes):
                    try:
                        value = pickle.loads(zlib.decompress(value))
                    except Exception:
                        pass  # Value might not be compressed
                
                logger.debug(f"Cache hit for key: {key}")
                return value
            
            self.stats.total_misses += 1
            return default
            
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {str(e)}")
            self.stats.total_misses += 1
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        scope: CacheScope = CacheScope.GLOBAL,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache."""
        try:
            cache_key = self._build_cache_key(key, scope)
            ttl = ttl_seconds or self.default_ttl_seconds
            tags = tags or []
            
            # Serialize and optionally compress value
            serialized_value = value
            if self.enable_compression:
                try:
                    serialized_data = pickle.dumps(value)
                    if len(serialized_data) > 1024:  # Only compress larger objects
                        serialized_value = zlib.compress(serialized_data)
                    else:
                        serialized_value = value
                except Exception:
                    serialized_value = value
            
            # Calculate size
            size_bytes = self._calculate_size(serialized_value)
            
            # Check if we need to make space
            await self._ensure_cache_space(size_bytes)
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                value=serialized_value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl_seconds=ttl,
                size_bytes=size_bytes,
                scope=scope,
                tags=tags
            )
            
            # Store entry
            self.cache[cache_key] = entry
            self.cache_by_scope[scope][key] = cache_key
            
            # Update tags
            for tag in tags:
                if tag not in self.cache_by_tags:
                    self.cache_by_tags[tag] = []
                self.cache_by_tags[tag].append(cache_key)
            
            # Update stats
            self.stats.total_entries += 1
            self.stats.total_size_bytes += size_bytes
            
            logger.debug(f"Cache set for key: {key}, size: {size_bytes} bytes")
            return True
            
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {str(e)}")
            return False
    
    async def delete(
        self,
        key: str,
        scope: CacheScope = CacheScope.GLOBAL
    ) -> bool:
        """Delete value from cache."""
        try:
            cache_key = self._build_cache_key(key, scope)
            
            if cache_key in self.cache:
                await self._remove_entry(cache_key)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {str(e)}")
            return False
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate cache entries by tags."""
        try:
            invalidated_count = 0
            
            for tag in tags:
                if tag in self.cache_by_tags:
                    cache_keys = self.cache_by_tags[tag].copy()
                    
                    for cache_key in cache_keys:
                        if cache_key in self.cache:
                            await self._remove_entry(cache_key)
                            invalidated_count += 1
            
            self.stats.invalidation_count += invalidated_count
            logger.info(f"Invalidated {invalidated_count} cache entries by tags: {tags}")
            
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Cache invalidation by tags failed: {str(e)}")
            return 0
    
    async def invalidate_by_scope(self, scope: CacheScope) -> int:
        """Invalidate all cache entries in a scope."""
        try:
            invalidated_count = 0
            
            if scope in self.cache_by_scope:
                cache_keys = list(self.cache_by_scope[scope].values())
                
                for cache_key in cache_keys:
                    if cache_key in self.cache:
                        await self._remove_entry(cache_key)
                        invalidated_count += 1
            
            self.stats.invalidation_count += invalidated_count
            logger.info(f"Invalidated {invalidated_count} cache entries in scope: {scope.value}")
            
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Cache invalidation by scope failed: {str(e)}")
            return 0
    
    async def get_or_set(
        self,
        key: str,
        factory_function: Callable,
        ttl_seconds: Optional[int] = None,
        scope: CacheScope = CacheScope.GLOBAL,
        tags: Optional[List[str]] = None,
        *args,
        **kwargs
    ) -> Any:
        """Get value from cache or set it using factory function."""
        try:
            # Try to get from cache first
            cached_value = await self.get(key, scope)
            
            if cached_value is not None:
                return cached_value
            
            # Generate value using factory function
            if asyncio.iscoroutinefunction(factory_function):
                value = await factory_function(*args, **kwargs)
            else:
                value = factory_function(*args, **kwargs)
            
            # Store in cache
            await self.set(key, value, ttl_seconds, scope, tags)
            
            return value
            
        except Exception as e:
            logger.error(f"Cache get_or_set failed for key {key}: {str(e)}")
            
            # Fallback to factory function
            try:
                if asyncio.iscoroutinefunction(factory_function):
                    return await factory_function(*args, **kwargs)
                else:
                    return factory_function(*args, **kwargs)
            except Exception as factory_error:
                logger.error(f"Factory function failed: {str(factory_error)}")
                return None
    
    async def warm_cache(
        self,
        warming_config: Dict[str, Any]
    ) -> bool:
        """Warm cache with predefined data."""
        try:
            config_id = warming_config.get('id', 'default')
            factory_function = warming_config.get('factory_function')
            key_generator = warming_config.get('key_generator')
            batch_size = warming_config.get('batch_size', self.optimization_config['warming_batch_size'])
            
            if not factory_function or not key_generator:
                logger.error("Invalid warming configuration")
                return False
            
            # Store warming config
            self.warming_configs[config_id] = warming_config
            
            # Generate keys to warm
            if asyncio.iscoroutinefunction(key_generator):
                keys = await key_generator()
            else:
                keys = key_generator()
            
            # Warm cache in batches
            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i:i + batch_size]
                
                warming_tasks = []
                for key in batch_keys:
                    task = asyncio.create_task(
                        self.get_or_set(
                            key=key,
                            factory_function=factory_function,
                            scope=warming_config.get('scope', CacheScope.GLOBAL),
                            tags=warming_config.get('tags', [])
                        )
                    )
                    warming_tasks.append(task)
                
                # Wait for batch completion
                await asyncio.gather(*warming_tasks, return_exceptions=True)
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            logger.info(f"Cache warming completed for config: {config_id}, warmed {len(keys)} keys")
            return True
            
        except Exception as e:
            logger.error(f"Cache warming failed: {str(e)}")
            return False
    
    async def _ensure_cache_space(self, required_bytes: int):
        """Ensure sufficient cache space by evicting entries if needed."""
        try:
            current_size = sum(entry.size_bytes for entry in self.cache.values())
            
            if current_size + required_bytes <= self.max_size_bytes:
                return
            
            # Calculate how much space we need to free
            space_to_free = (current_size + required_bytes) - self.max_size_bytes
            
            # Get entries sorted by eviction strategy
            entries_to_evict = self._get_eviction_candidates(space_to_free)
            
            # Evict entries
            freed_space = 0
            for cache_key in entries_to_evict:
                if cache_key in self.cache:
                    freed_space += self.cache[cache_key].size_bytes
                    await self._remove_entry(cache_key)
                    self.stats.eviction_count += 1
                    
                    if freed_space >= space_to_free:
                        break
            
            logger.debug(f"Evicted entries to free {freed_space} bytes")
            
        except Exception as e:
            logger.error(f"Cache space management failed: {str(e)}")
    
    def _get_eviction_candidates(self, space_needed: int) -> List[str]:
        """Get list of cache keys to evict based on strategy."""
        try:
            entries = list(self.cache.values())
            
            if self.strategy == CacheStrategy.LRU:
                # Sort by last accessed time (oldest first)
                entries.sort(key=lambda x: x.last_accessed)
            
            elif self.strategy == CacheStrategy.LFU:
                # Sort by access count (least frequent first)
                entries.sort(key=lambda x: x.access_count)
            
            elif self.strategy == CacheStrategy.FIFO:
                # Sort by creation time (oldest first)
                entries.sort(key=lambda x: x.created_at)
            
            elif self.strategy == CacheStrategy.TTL:
                # Sort by expiration time (soonest to expire first)
                entries.sort(key=lambda x: x.created_at + timedelta(seconds=x.ttl_seconds or 0))
            
            elif self.strategy == CacheStrategy.ADAPTIVE:
                # Adaptive strategy considering multiple factors
                now = datetime.now()
                
                def adaptive_score(entry):
                    age_factor = (now - entry.last_accessed).total_seconds() / 3600  # Hours since last access
                    frequency_factor = 1.0 / max(entry.access_count, 1)  # Inverse of access count
                    size_factor = entry.size_bytes / 1024  # Size in KB
                    
                    return age_factor + frequency_factor + (size_factor * 0.001)
                
                entries.sort(key=adaptive_score, reverse=True)
            
            # Return keys of entries to evict
            return [entry.key for entry in entries]
            
        except Exception as e:
            logger.error(f"Eviction candidate selection failed: {str(e)}")
            return []
    
    async def _remove_entry(self, cache_key: str):
        """Remove cache entry and update indexes."""
        try:
            if cache_key not in self.cache:
                return
            
            entry = self.cache[cache_key]
            
            # Remove from main cache
            del self.cache[cache_key]
            
            # Remove from scope index
            for key, stored_cache_key in self.cache_by_scope[entry.scope].items():
                if stored_cache_key == cache_key:
                    del self.cache_by_scope[entry.scope][key]
                    break
            
            # Remove from tag indexes
            for tag in entry.tags:
                if tag in self.cache_by_tags:
                    if cache_key in self.cache_by_tags[tag]:
                        self.cache_by_tags[tag].remove(cache_key)
                    
                    # Clean up empty tag lists
                    if not self.cache_by_tags[tag]:
                        del self.cache_by_tags[tag]
            
            # Update stats
            self.stats.total_entries -= 1
            self.stats.total_size_bytes -= entry.size_bytes
            
        except Exception as e:
            logger.error(f"Cache entry removal failed: {str(e)}")
    
    def _build_cache_key(self, key: str, scope: CacheScope) -> str:
        """Build full cache key with scope."""
        return f"{scope.value}:{key}"
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, bytes):
                return len(value)
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, bool):
                return 1
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(
                    self._calculate_size(k) + self._calculate_size(v)
                    for k, v in value.items()
                )
            else:
                # Fallback to pickle size
                return len(pickle.dumps(value))
                
        except Exception:
            return 1024  # Default size estimate
    
    async def _cleanup_loop(self):
        """Background task for cache cleanup."""
        try:
            while True:
                await asyncio.sleep(self.optimization_config['cleanup_interval_seconds'])
                await self._cleanup_expired_entries()
                
        except asyncio.CancelledError:
            logger.info("Cache cleanup loop cancelled")
        except Exception as e:
            logger.error(f"Cache cleanup loop error: {str(e)}")
    
    async def _cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        try:
            expired_keys = []
            
            for cache_key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(cache_key)
            
            for cache_key in expired_keys:
                await self._remove_entry(cache_key)
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            logger.error(f"Expired entries cleanup failed: {str(e)}")
    
    async def _stats_update_loop(self):
        """Background task for updating cache statistics."""
        try:
            while True:
                await asyncio.sleep(self.optimization_config['stats_update_interval_seconds'])
                await self._update_cache_stats()
                
        except asyncio.CancelledError:
            logger.info("Cache stats update loop cancelled")
        except Exception as e:
            logger.error(f"Cache stats update loop error: {str(e)}")
    
    async def _update_cache_stats(self):
        """Update cache performance statistics."""
        try:
            total_requests = self.stats.total_hits + self.stats.total_misses
            
            if total_requests > 0:
                self.stats.hit_ratio = self.stats.total_hits / total_requests
            else:
                self.stats.hit_ratio = 0.0
            
            self.stats.total_entries = len(self.cache)
            self.stats.total_size_bytes = sum(entry.size_bytes for entry in self.cache.values())
            
            if self.stats.total_entries > 0:
                self.stats.average_entry_size = self.stats.total_size_bytes / self.stats.total_entries
            else:
                self.stats.average_entry_size = 0.0
            
        except Exception as e:
            logger.error(f"Cache stats update failed: {str(e)}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            await self._update_cache_stats()
            
            return {
                'performance_stats': asdict(self.stats),
                'configuration': {
                    'max_size_mb': self.max_size_bytes // (1024 * 1024),
                    'default_ttl_seconds': self.default_ttl_seconds,
                    'strategy': self.strategy.value,
                    'compression_enabled': self.enable_compression
                },
                'cache_distribution': {
                    scope.value: len(entries) 
                    for scope, entries in self.cache_by_scope.items()
                },
                'tag_distribution': {
                    tag: len(keys) for tag, keys in self.cache_by_tags.items()
                },
                'warming_configs': list(self.warming_configs.keys()),
                'memory_usage': {
                    'current_size_mb': self.stats.total_size_bytes / (1024 * 1024),
                    'utilization_percentage': (self.stats.total_size_bytes / self.max_size_bytes) * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {}
    
    async def optimize_cache(self) -> Dict[str, Any]:
        """Run cache optimization and return optimization results."""
        try:
            optimization_results = {
                'actions_taken': [],
                'before_stats': await self.get_cache_stats(),
                'optimization_timestamp': datetime.now().isoformat()
            }
            
            # Clean up expired entries
            await self._cleanup_expired_entries()
            optimization_results['actions_taken'].append('cleaned_expired_entries')
            
            # Optimize strategy if adaptive
            if self.strategy == CacheStrategy.ADAPTIVE:
                await self._optimize_adaptive_strategy()
                optimization_results['actions_taken'].append('optimized_adaptive_strategy')
            
            # Warm high-value cache entries
            await self._warm_high_value_entries()
            optimization_results['actions_taken'].append('warmed_high_value_entries')
            
            optimization_results['after_stats'] = await self.get_cache_stats()
            
            logger.info(f"Cache optimization completed: {optimization_results['actions_taken']}")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Cache optimization failed: {str(e)}")
            return {'error': str(e)}
    
    async def _optimize_adaptive_strategy(self):
        """Optimize adaptive caching strategy based on usage patterns."""
        try:
            # Analyze access patterns
            high_frequency_entries = [
                entry for entry in self.cache.values()
                if entry.access_count >= self.optimization_config['min_access_count_for_warming']
            ]
            
            # Extend TTL for high-frequency entries
            for entry in high_frequency_entries:
                if entry.ttl_seconds and entry.access_count > 10:
                    entry.ttl_seconds = int(entry.ttl_seconds * 1.5)  # Extend TTL by 50%
            
            logger.debug(f"Optimized TTL for {len(high_frequency_entries)} high-frequency entries")
            
        except Exception as e:
            logger.error(f"Adaptive strategy optimization failed: {str(e)}")
    
    async def _warm_high_value_entries(self):
        """Warm cache with high-value entries that were recently evicted."""
        try:
            # This would typically analyze eviction logs and re-warm valuable entries
            # For now, we'll just trigger any configured warming tasks
            
            for config_id, config in self.warming_configs.items():
                if config_id not in self.warming_tasks:
                    warming_task = asyncio.create_task(self.warm_cache(config))
                    self.warming_tasks[config_id] = warming_task
            
        except Exception as e:
            logger.error(f"High-value entry warming failed: {str(e)}")
    
    async def cleanup(self):
        """Clean up cache optimizer resources."""
        try:
            # Cancel background tasks
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            if self.stats_task:
                self.stats_task.cancel()
                try:
                    await self.stats_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel warming tasks
            for task in self.warming_tasks.values():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Clear cache
            self.cache.clear()
            for scope_cache in self.cache_by_scope.values():
                scope_cache.clear()
            self.cache_by_tags.clear()
            
            logger.info("CacheOptimizer cleanup completed")
            
        except Exception as e:
            logger.error(f"CacheOptimizer cleanup failed: {str(e)}")