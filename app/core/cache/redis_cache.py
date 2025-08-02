import redis
import pickle
import json
import hashlib
from typing import Any, Optional, List, Dict
from datetime import timedelta
import numpy as np
from app.utils.logger import logger
from app.config.matching_config import get_matching_config
from app.core.config import get_settings

config = get_matching_config()
settings = get_settings()


class MatchingCache:
    def __init__(self):
        try:
            # Use dedicated Redis URL for cache instead of Celery broker
            self.redis_client = redis.Redis.from_url(
                settings.REDIS_URL,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=20,
                health_check_interval=30,
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache connection established")
        except Exception as e:
            logger.error(f"Redis cache connection failed: {e}")
            self.redis_client = None
        
        self.default_ttl = config.cache_ttl

    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)

        hash_obj = hashlib.md5(data_str.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"

    def get_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        if not self.redis_client:
            return None
        try:
            key = self._generate_key(f"embedding:{model_name}", text)
            cached = self.redis_client.get(key)
            if cached:
                return pickle.loads(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None

    def set_embedding(self, text: str, model_name: str, embedding: np.ndarray):
        """Cache embedding"""
        if not self.redis_client:
            return
        try:
            key = self._generate_key(f"embedding:{model_name}", text)
            self.redis_client.setex(key, self.default_ttl, pickle.dumps(embedding))
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    def get_processed_job(self, job_data: Dict) -> Optional[Dict]:
        """Get cached processed job"""
        try:
            key = self._generate_key("processed_job", job_data)
            cached = self.redis_client.get(key)
            if cached:
                return json.loads(cached.decode())
        except Exception as e:
            logger.warning(f"Job cache retrieval failed: {e}")
        return None

    def set_processed_job(self, job_data: Dict, processed_job: Dict):
        """Cache processed job"""
        try:
            key = self._generate_key("processed_job", job_data)
            self.redis_client.setex(
                key, self.default_ttl, json.dumps(processed_job, default=str)
            )
        except Exception as e:
            logger.warning(f"Job cache storage failed: {e}")

    def get_skills(self, text: str) -> Optional[List[str]]:
        """Get cached skills"""
        try:
            key = self._generate_key("skills", text)
            cached = self.redis_client.get(key)
            if cached:
                return json.loads(cached.decode())
        except Exception as e:
            logger.warning(f"Skills cache retrieval failed: {e}")
        return None

    def set_skills(self, text: str, skills: List[str]):
        """Cache extracted skills"""
        try:
            key = self._generate_key("skills", text)
            self.redis_client.setex(key, self.default_ttl, json.dumps(skills))
        except Exception as e:
            logger.warning(f"Skills cache storage failed: {e}")

    def invalidate_pattern(self, pattern: str):
        """Invalidate cache by pattern"""
        try:
            keys = self.redis_client.keys(f"{pattern}*")
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")


# Global cache instance
cache = MatchingCache()
