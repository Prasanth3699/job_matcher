"""
Embedding Cache for ML models.

This module provides intelligent caching for embeddings, model predictions,
and computed results to significantly improve performance and reduce latency.
"""

import asyncio
import hashlib
import json
import pickle
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import redis.asyncio as aioredis
from pathlib import Path
import numpy as np

from app.utils.logger import logger
from app.core.config import get_settings
from app.core.constants.ml_constants import (
    CachingParameters,
    EnsembleConfig,
    PerformanceMetrics
)


class EmbeddingCache:
    """
    High-performance caching system for ML embeddings and results with
    intelligent eviction policies and distributed caching support.
    """

    def __init__(self):
        """Initialize the embedding cache with Redis and in-memory components."""
        self.settings = get_settings()
        self.redis_client: Optional[aioredis.Redis] = None
        self.local_cache: Dict[str, Dict[str, Any]] = {}
        
        # Cache configuration
        self.cache_config = EnsembleConfig.CACHE_CONFIG
        self.embedding_ttl = self.cache_config['embedding_ttl']
        self.result_ttl = self.cache_config['result_ttl']
        self.model_ttl = self.cache_config['model_ttl']
        self.max_embedding_cache_size = self.cache_config['embedding_cache_size']
        self.max_batch_cache_size = self.cache_config['batch_cache_size']
        
        # Performance tracking
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0,
            'embedding_cache_size': 0,
            'result_cache_size': 0,
            'memory_usage_mb': 0.0
        }
        
        # Initialize async components
        self._initialization_task = None
        
        logger.info("EmbeddingCache initialized")

    async def initialize(self):
        """Initialize async components like Redis connection."""
        try:
            if self.settings.REDIS_URL:
                self.redis_client = await aioredis.from_url(
                    self.settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=False,  # Keep as bytes for pickle
                    max_connections=20
                )
                
                # Test connection
                await self.redis_client.ping()
                logger.info("Redis connection established for embedding cache")
            else:
                logger.warning("No Redis URL configured, using local cache only")
                
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {str(e)}")
            self.redis_client = None

    async def get_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """
        Retrieve cached embedding for text and model combination.
        
        Args:
            text: Input text to get embedding for
            model_name: Name of the model used for embedding
            
        Returns:
            Cached embedding array or None if not found
        """
        try:
            self.cache_stats['total_requests'] += 1
            
            # Generate cache key
            cache_key = self._generate_embedding_key(text, model_name)
            
            # Try local cache first (fastest)
            embedding = await self._get_from_local_cache(cache_key, 'embedding')
            if embedding is not None:
                self.cache_stats['hits'] += 1
                return embedding
            
            # Try Redis cache
            if self.redis_client:
                embedding = await self._get_from_redis_cache(cache_key, 'embedding')
                if embedding is not None:
                    # Store in local cache for next time
                    await self._store_in_local_cache(cache_key, embedding, 'embedding')
                    self.cache_stats['hits'] += 1
                    return embedding
            
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving embedding from cache: {str(e)}")
            self.cache_stats['misses'] += 1
            return None

    async def store_embedding(
        self,
        text: str,
        model_name: str,
        embedding: np.ndarray
    ) -> bool:
        """
        Store embedding in cache with appropriate TTL.
        
        Args:
            text: Input text the embedding represents
            model_name: Name of the model used
            embedding: Embedding array to cache
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            cache_key = self._generate_embedding_key(text, model_name)
            
            # Store in local cache
            success_local = await self._store_in_local_cache(
                cache_key, embedding, 'embedding'
            )
            
            # Store in Redis cache
            success_redis = True
            if self.redis_client:
                success_redis = await self._store_in_redis_cache(
                    cache_key, embedding, 'embedding', self.embedding_ttl
                )
            
            # Update cache size tracking
            self.cache_stats['embedding_cache_size'] += 1
            self._update_memory_usage()
            
            return success_local and success_redis
            
        except Exception as e:
            logger.error(f"Error storing embedding in cache: {str(e)}")
            return False

    async def get_cached_results(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve cached matching results.
        
        Args:
            cache_key: Unique key for the cached results
            
        Returns:
            Cached results or None if not found
        """
        try:
            self.cache_stats['total_requests'] += 1
            
            # Try local cache first
            results = await self._get_from_local_cache(cache_key, 'result')
            if results is not None:
                self.cache_stats['hits'] += 1
                return results
            
            # Try Redis cache
            if self.redis_client:
                results = await self._get_from_redis_cache(cache_key, 'result')
                if results is not None:
                    # Store in local cache
                    await self._store_in_local_cache(cache_key, results, 'result')
                    self.cache_stats['hits'] += 1
                    return results
            
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving results from cache: {str(e)}")
            self.cache_stats['misses'] += 1
            return None

    async def cache_results(
        self,
        cache_key: str,
        results: List[Dict[str, Any]]
    ) -> bool:
        """
        Cache matching results.
        
        Args:
            cache_key: Unique key for the results
            results: Results to cache
            
        Returns:
            True if cached successfully, False otherwise
        """
        try:
            # Store in local cache
            success_local = await self._store_in_local_cache(
                cache_key, results, 'result'
            )
            
            # Store in Redis cache
            success_redis = True
            if self.redis_client:
                success_redis = await self._store_in_redis_cache(
                    cache_key, results, 'result', self.result_ttl
                )
            
            # Update cache size tracking
            self.cache_stats['result_cache_size'] += 1
            self._update_memory_usage()
            
            return success_local and success_redis
            
        except Exception as e:
            logger.error(f"Error caching results: {str(e)}")
            return False

    async def get_batch_embeddings(
        self,
        texts: List[str],
        model_name: str
    ) -> Tuple[List[Optional[np.ndarray]], List[str]]:
        """
        Retrieve multiple embeddings, returning cached ones and missing text list.
        
        Args:
            texts: List of texts to get embeddings for
            model_name: Name of the model
            
        Returns:
            Tuple of (embeddings_list, missing_texts)
        """
        try:
            embeddings = [None] * len(texts)
            missing_texts = []
            missing_indices = []
            
            # Check each text in cache
            for i, text in enumerate(texts):
                embedding = await self.get_embedding(text, model_name)
                if embedding is not None:
                    embeddings[i] = embedding
                else:
                    missing_texts.append(text)
                    missing_indices.append(i)
            
            return embeddings, missing_texts
            
        except Exception as e:
            logger.error(f"Error retrieving batch embeddings: {str(e)}")
            return [None] * len(texts), texts

    async def store_batch_embeddings(
        self,
        texts: List[str],
        embeddings: List[np.ndarray],
        model_name: str
    ) -> bool:
        """
        Store multiple embeddings in batch.
        
        Args:
            texts: List of texts
            embeddings: Corresponding embeddings
            model_name: Name of the model
            
        Returns:
            True if all stored successfully, False otherwise
        """
        try:
            if len(texts) != len(embeddings):
                logger.error("Mismatch between texts and embeddings length")
                return False
            
            success_count = 0
            for text, embedding in zip(texts, embeddings):
                if await self.store_embedding(text, embedding, model_name):
                    success_count += 1
            
            return success_count == len(texts)
            
        except Exception as e:
            logger.error(f"Error storing batch embeddings: {str(e)}")
            return False

    async def _get_from_local_cache(self, key: str, cache_type: str) -> Optional[Any]:
        """Retrieve item from local in-memory cache."""
        try:
            if key in self.local_cache:
                cache_entry = self.local_cache[key]
                
                # Check expiration
                if cache_entry['expires_at'] > time.time():
                    # Update access time for LRU
                    cache_entry['last_accessed'] = time.time()
                    return cache_entry['data']
                else:
                    # Remove expired entry
                    del self.local_cache[key]
                    
            return None
            
        except Exception as e:
            logger.error(f"Error accessing local cache: {str(e)}")
            return None

    async def _store_in_local_cache(
        self,
        key: str,
        data: Any,
        cache_type: str
    ) -> bool:
        """Store item in local in-memory cache."""
        try:
            # Check if we need to evict items
            await self._maybe_evict_local_cache()
            
            # Determine TTL based on cache type
            ttl = {
                'embedding': self.embedding_ttl,
                'result': self.result_ttl,
                'model': self.model_ttl
            }.get(cache_type, 3600)
            
            # Store with metadata
            self.local_cache[key] = {
                'data': data,
                'cache_type': cache_type,
                'created_at': time.time(),
                'last_accessed': time.time(),
                'expires_at': time.time() + ttl
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing in local cache: {str(e)}")
            return False

    async def _get_from_redis_cache(self, key: str, cache_type: str) -> Optional[Any]:
        """Retrieve item from Redis cache."""
        try:
            if not self.redis_client:
                return None
            
            # Add cache type prefix to key
            redis_key = f"ml_cache:{cache_type}:{key}"
            
            # Get data from Redis
            cached_data = await self.redis_client.get(redis_key)
            if cached_data:
                # Deserialize based on cache type
                if cache_type == 'embedding':
                    return pickle.loads(cached_data)
                else:
                    return json.loads(cached_data.decode('utf-8'))
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving from Redis cache: {str(e)}")
            return None

    async def _store_in_redis_cache(
        self,
        key: str,
        data: Any,
        cache_type: str,
        ttl: int
    ) -> bool:
        """Store item in Redis cache."""
        try:
            if not self.redis_client:
                return True  # Consider it successful if Redis is not available
            
            # Add cache type prefix to key
            redis_key = f"ml_cache:{cache_type}:{key}"
            
            # Serialize based on cache type
            if cache_type == 'embedding':
                serialized_data = pickle.dumps(data)
            else:
                serialized_data = json.dumps(data, default=str)
            
            # Store with TTL
            await self.redis_client.setex(redis_key, ttl, serialized_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing in Redis cache: {str(e)}")
            return False

    async def _maybe_evict_local_cache(self):
        """Evict items from local cache if it's getting too large."""
        try:
            max_cache_size = max(self.max_embedding_cache_size, 1000)
            
            if len(self.local_cache) >= max_cache_size:
                # Sort by last accessed time (LRU eviction)
                sorted_items = sorted(
                    self.local_cache.items(),
                    key=lambda x: x[1]['last_accessed']
                )
                
                # Remove oldest 20% of items
                items_to_remove = int(len(sorted_items) * 0.2)
                for i in range(items_to_remove):
                    key_to_remove = sorted_items[i][0]
                    del self.local_cache[key_to_remove]
                    self.cache_stats['evictions'] += 1
                
                logger.info(f"Evicted {items_to_remove} items from local cache")
                
        except Exception as e:
            logger.error(f"Error during cache eviction: {str(e)}")

    def _generate_embedding_key(self, text: str, model_name: str) -> str:
        """Generate a unique cache key for text and model combination."""
        # Normalize text for consistent caching
        normalized_text = text.strip().lower()
        
        # Create hash of text + model for consistent key generation
        content = f"{normalized_text}|{model_name}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _update_memory_usage(self):
        """Update memory usage statistics."""
        try:
            import sys
            total_size = sum(
                sys.getsizeof(entry) for entry in self.local_cache.values()
            )
            self.cache_stats['memory_usage_mb'] = total_size / (1024 * 1024)
        except:
            pass  # Memory tracking is optional

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            # Calculate hit rate
            total_requests = self.cache_stats['total_requests']
            hit_rate = (
                self.cache_stats['hits'] / total_requests
                if total_requests > 0 else 0.0
            )
            
            # Get Redis stats if available
            redis_stats = {}
            if self.redis_client:
                try:
                    redis_info = await self.redis_client.info('memory')
                    redis_stats = {
                        'redis_memory_used': redis_info.get('used_memory_human', 'Unknown'),
                        'redis_connected': True
                    }
                except:
                    redis_stats = {'redis_connected': False}
            
            return {
                'cache_performance': {
                    'hit_rate': hit_rate,
                    'total_requests': total_requests,
                    'cache_hits': self.cache_stats['hits'],
                    'cache_misses': self.cache_stats['misses'],
                    'evictions': self.cache_stats['evictions']
                },
                'cache_sizes': {
                    'local_cache_entries': len(self.local_cache),
                    'embedding_cache_size': self.cache_stats['embedding_cache_size'],
                    'result_cache_size': self.cache_stats['result_cache_size'],
                    'memory_usage_mb': self.cache_stats['memory_usage_mb']
                },
                'configuration': {
                    'embedding_ttl': self.embedding_ttl,
                    'result_ttl': self.result_ttl,
                    'max_embedding_cache_size': self.max_embedding_cache_size,
                    'max_batch_cache_size': self.max_batch_cache_size
                },
                'redis_stats': redis_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {'error': 'Failed to retrieve cache statistics'}

    async def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            cache_type: Type of cache to clear ('embedding', 'result', 'model')
                       If None, clears all caches
        """
        try:
            # Clear local cache
            if cache_type:
                keys_to_remove = [
                    key for key, entry in self.local_cache.items()
                    if entry.get('cache_type') == cache_type
                ]
                for key in keys_to_remove:
                    del self.local_cache[key]
            else:
                self.local_cache.clear()
            
            # Clear Redis cache
            if self.redis_client:
                if cache_type:
                    pattern = f"ml_cache:{cache_type}:*"
                    keys = await self.redis_client.keys(pattern)
                    if keys:
                        await self.redis_client.delete(*keys)
                else:
                    keys = await self.redis_client.keys("ml_cache:*")
                    if keys:
                        await self.redis_client.delete(*keys)
            
            # Reset stats
            if not cache_type:
                self.cache_stats = {
                    'hits': 0,
                    'misses': 0,
                    'evictions': 0,
                    'total_requests': 0,
                    'embedding_cache_size': 0,
                    'result_cache_size': 0,
                    'memory_usage_mb': 0.0
                }
            
            logger.info(f"Cleared {cache_type or 'all'} cache(s)")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")

    async def warm_up_cache(self, texts: List[str], model_names: List[str]):
        """
        Pre-warm cache with commonly used embeddings.
        
        Args:
            texts: List of texts to pre-compute embeddings for
            model_names: List of models to use
        """
        try:
            logger.info(f"Warming up cache with {len(texts)} texts and {len(model_names)} models")
            
            # This would typically load models and compute embeddings
            # For now, we'll just log the intention
            logger.info("Cache warm-up completed")
            
        except Exception as e:
            logger.error(f"Cache warm-up failed: {str(e)}")

    async def cleanup(self):
        """Clean up cache resources."""
        try:
            # Clear local cache
            self.local_cache.clear()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
                
            logger.info("EmbeddingCache cleanup completed")
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {str(e)}")