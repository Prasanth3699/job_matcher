"""
Caching module for ML models and embeddings.

This module provides high-performance caching capabilities for the ensemble
matching system, including embedding caching, result caching, and model caching.
"""

from .embedding_cache import EmbeddingCache

__all__ = [
    "EmbeddingCache"
]