"""
Ensemble matching module for advanced ML pipeline.

This module provides state-of-the-art ensemble learning capabilities for
resume-job matching with multiple specialized models and intelligent
score combination strategies.
"""

from .ensemble_manager import EnsembleMatchingEngine
from .model_registry import ModelRegistry
from .weight_optimizer import WeightOptimizer
from .ensemble_scorer import EnsembleScorer

__all__ = [
    "EnsembleMatchingEngine",
    "ModelRegistry",
    "WeightOptimizer", 
    "EnsembleScorer"
]