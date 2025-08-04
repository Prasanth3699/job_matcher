"""
Integration module for connecting ensemble scoring with neural ranking.

This module provides seamless integration between the ensemble matching engine
and the neural ranking system for optimal matching performance.
"""

from .ensemble_ranking_bridge import EnsembleRankingBridge
from .unified_matching_service import UnifiedMatchingService

__all__ = [
    "EnsembleRankingBridge",
    "UnifiedMatchingService"
]