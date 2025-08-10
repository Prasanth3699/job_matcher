"""
Model Registry for ensemble matching pipeline.

This module handles loading, caching, and management of multiple ML models
used in the ensemble matching system.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import pickle
import threading

from app.utils.logger import logger
from app.core.constants.ml_constants import (
    ModelParameters,
    EnsembleConfig,
    CachingParameters,
    MLConstants
)


class ModelRegistry:
    """
    Centralized registry for managing multiple ML models with intelligent
    loading, caching, and lifecycle management.
    """

    def __init__(self):
        """Initialize the model registry with caching and loading capabilities."""
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.model_load_times: Dict[str, datetime] = {}
        self.access_counts: Dict[str, int] = {}
        self.loading_locks: Dict[str, threading.Lock] = {}
        
        # Configuration
        self.cache_config = EnsembleConfig.CACHE_CONFIG
        self.model_config = EnsembleConfig.MODEL_CONFIG
        self.max_cached_models = CachingParameters.MODEL_CACHE_SIZE
        
        # Initialize available models
        self._initialize_model_definitions()
        
        logger.info("ModelRegistry initialized successfully")

    def _initialize_model_definitions(self):
        """Initialize definitions of all available models."""
        self.model_definitions = {
            'primary_semantic': {
                'type': 'sentence_transformer',
                'model_name': ModelParameters.PRIMARY_MODELS['all-mpnet-base-v2']['name'],
                'config': ModelParameters.PRIMARY_MODELS['all-mpnet-base-v2'],
                'priority': 1,
                'description': 'High-quality semantic embeddings for primary matching'
            },
            'fast_semantic': {
                'type': 'sentence_transformer',
                'model_name': ModelParameters.PRIMARY_MODELS['all-MiniLM-L6-v2']['name'],
                'config': ModelParameters.PRIMARY_MODELS['all-MiniLM-L6-v2'],
                'priority': 2,
                'description': 'Fast semantic embeddings for speed optimization'
            },
            'domain_specific': {
                'type': 'custom_semantic',
                'model_name': 'tech_domain_fine_tuned',
                'config': {
                    'dimensions': 768,
                    'max_length': 512,
                    'batch_size': 16
                },
                'priority': 3,
                'description': 'Domain-specific model fine-tuned for tech industry'
            },
            'feature_based': {
                'type': 'feature_matcher',
                'model_name': 'traditional_ml_features',
                'config': {
                    'feature_types': ['skills', 'experience', 'education', 'location'],
                    'weighting_strategy': 'learned'
                },
                'priority': 4,
                'description': 'Traditional ML feature-based matching'
            },
            'feedback_learned': {
                'type': 'neural_feedback',
                'model_name': 'user_feedback_neural',
                'config': {
                    'input_dim': 512,
                    'hidden_layers': [256, 128, 64],
                    'output_dim': 1
                },
                'priority': 5,
                'description': 'Neural network trained on user feedback patterns'
            }
        }

    async def get_model(self, model_name: str) -> Optional[Any]:
        """
        Get a model by name, loading it if necessary.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            The loaded model instance or None if loading fails
        """
        try:
            # Check if model is already loaded and valid
            if model_name in self.loaded_models:
                if self._is_model_valid(model_name):
                    self._update_access_stats(model_name)
                    return self.loaded_models[model_name]
                else:
                    # Remove invalid model
                    await self._unload_model(model_name)
            
            # Load model if not cached
            return await self._load_model(model_name)
            
        except Exception as e:
            logger.error(f"Failed to get model {model_name}: {str(e)}")
            return None

    async def _load_model(self, model_name: str) -> Optional[Any]:
        """Load a specific model with thread safety and caching."""
        # Ensure thread safety for model loading
        if model_name not in self.loading_locks:
            self.loading_locks[model_name] = threading.Lock()
        
        with self.loading_locks[model_name]:
            # Double-check if model was loaded by another thread
            if model_name in self.loaded_models and self._is_model_valid(model_name):
                return self.loaded_models[model_name]
            
            try:
                # Check memory before loading
                if not self._check_memory_availability():
                    await self._evict_least_used_model()
                
                # Get model definition
                if model_name not in self.model_definitions:
                    logger.error(f"Unknown model: {model_name}")
                    return None
                
                model_def = self.model_definitions[model_name]
                
                logger.info(f"Loading model: {model_name} ({model_def['type']})")
                start_time = time.time()
                
                # Load based on model type
                model = await self._load_model_by_type(model_name, model_def)
                
                if model:
                    # Cache the loaded model
                    self.loaded_models[model_name] = model
                    self.model_load_times[model_name] = datetime.now()
                    self.access_counts[model_name] = 1
                    
                    # Store metadata
                    self.model_metadata[model_name] = {
                        'load_time': time.time() - start_time,
                        'memory_usage': self._estimate_model_memory(model),
                        'last_accessed': datetime.now(),
                        'access_count': 1,
                        'model_type': model_def['type'],
                        'description': model_def['description']
                    }
                    
                    load_time = time.time() - start_time
                    logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s")
                    
                    return model
                else:
                    logger.error(f"Failed to load model {model_name}")
                    return None
                
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}", exc_info=True)
                return None

    async def _load_model_by_type(self, model_name: str, model_def: Dict[str, Any]) -> Optional[Any]:
        """Load a model based on its type."""
        model_type = model_def['type']
        
        try:
            if model_type == 'sentence_transformer':
                return await self._load_sentence_transformer(model_def)
            elif model_type == 'custom_semantic':
                return await self._load_custom_semantic_model(model_def)
            elif model_type == 'feature_matcher':
                return await self._load_feature_matcher(model_def)
            elif model_type == 'neural_feedback':
                return await self._load_neural_feedback_model(model_def)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {str(e)}")
            return None

    async def _load_sentence_transformer(self, model_def: Dict[str, Any]) -> Optional[Any]:
        """Load a SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = model_def['model_name']
            config = model_def['config']
            
            # Load with configuration
            model = SentenceTransformer(
                model_name,
                cache_folder=ModelParameters.MODEL_CACHE_DIR,
                trust_remote_code=ModelParameters.TRUST_REMOTE_CODE
            )
            
            # Add custom methods for consistent interface
            model.match_single = self._create_semantic_matcher(model, config)
            model.match_batch = self._create_semantic_batch_matcher(model, config)
            
            return model
            
        except ImportError:
            logger.error("sentence-transformers not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {str(e)}")
            return None

    async def _load_custom_semantic_model(self, model_def: Dict[str, Any]) -> Optional[Any]:
        """Load a custom semantic model (placeholder for domain-specific models)."""
        try:
            # This would load a custom fine-tuned model
            # For now, we'll use a sentence transformer as a placeholder
            from sentence_transformers import SentenceTransformer
            
            # Fall back to a general model for now
            model = SentenceTransformer(
                ModelParameters.DEFAULT_PRIMARY_MODEL,
                cache_folder=ModelParameters.MODEL_CACHE_DIR
            )
            
            config = model_def['config']
            model.match_single = self._create_semantic_matcher(model, config)
            model.match_batch = self._create_semantic_batch_matcher(model, config)
            
            logger.info("Using fallback semantic model for domain-specific model")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load custom semantic model: {str(e)}")
            return None

    async def _load_feature_matcher(self, model_def: Dict[str, Any]) -> Optional[Any]:
        """Load a traditional feature-based matcher."""
        try:
            from app.core.matching.feature_matcher import FeatureMatcher
            
            # FeatureMatcher doesn't accept parameters in its constructor
            model = FeatureMatcher()
            
            # Add configuration as attributes if needed
            config = model_def['config']
            model.config = config
            
            return model
            
        except ImportError:
            logger.warning("FeatureMatcher not available, creating placeholder")
            return self._create_feature_matcher_placeholder(model_def['config'])
        except Exception as e:
            logger.error(f"Failed to load feature matcher: {str(e)}")
            return None

    async def _load_neural_feedback_model(self, model_def: Dict[str, Any]) -> Optional[Any]:
        """Load a neural network trained on user feedback."""
        try:
            # This would load a pre-trained neural network
            # For now, we'll create a placeholder that returns reasonable scores
            return self._create_feedback_model_placeholder(model_def['config'])
            
        except Exception as e:
            logger.error(f"Failed to load neural feedback model: {str(e)}")
            return None

    def _create_semantic_matcher(self, model: Any, config: Dict[str, Any]):
        """Create a semantic matching function for a model."""
        def match_single(resume_features: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Extract text content
                resume_text = resume_features.get('text_content', '')
                job_text = self._extract_job_text(job)
                
                # Get embeddings
                resume_embedding = model.encode([resume_text])
                job_embedding = model.encode([job_text])
                
                # Calculate similarity
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np
                
                similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
                
                # Normalize similarity to [0, 1] range
                # Cosine similarity is in [-1, 1], so we convert to [0, 1]
                normalized_score = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
                
                return {
                    'score': float(normalized_score),
                    'confidence': min(1.0, max(0.0, float(normalized_score) + 0.1)),
                    'explanation': f'Semantic similarity: {similarity:.3f}',
                    'model_type': 'semantic'
                }
                
            except Exception as e:
                logger.error(f"Semantic matching error: {str(e)}")
                return {'score': 0.0, 'confidence': 0.0, 'explanation': 'Error in semantic matching'}
        
        return match_single

    def _create_semantic_batch_matcher(self, model: Any, config: Dict[str, Any]):
        """Create a batch semantic matching function."""
        def match_batch(resume_features: Dict[str, Any], jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            try:
                resume_text = resume_features.get('text_content', '')
                job_texts = [self._extract_job_text(job) for job in jobs]
                
                # Batch encode
                resume_embedding = model.encode([resume_text])
                job_embeddings = model.encode(job_texts)
                
                # Calculate similarities
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(resume_embedding, job_embeddings)[0]
                
                results = []
                for i, similarity in enumerate(similarities):
                    # Normalize similarity to [0, 1] range
                    # Cosine similarity is in [-1, 1], so we convert to [0, 1]
                    normalized_score = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
                    
                    results.append({
                        'score': float(normalized_score),
                        'confidence': min(1.0, max(0.0, float(normalized_score) + 0.1)),
                        'explanation': f'Semantic similarity: {similarity:.3f}',
                        'model_type': 'semantic'
                    })
                
                return results
                
            except Exception as e:
                logger.error(f"Batch semantic matching error: {str(e)}")
                return [{'score': 0.0, 'confidence': 0.0} for _ in jobs]
        
        return match_batch

    def _create_feature_matcher_placeholder(self, config: Dict[str, Any]) -> Any:
        """Create a placeholder feature matcher."""
        class FeatureMatcherPlaceholder:
            def __init__(self, config):
                self.config = config
            
            def match_single(self, resume_features: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
                # Simple skill matching as placeholder
                resume_skills = set(s.lower() for s in resume_features.get('skills', []))
                job_skills = set(s.lower() for s in job.get('skills', []))
                
                if resume_skills and job_skills:
                    intersection = len(resume_skills.intersection(job_skills))
                    union = len(resume_skills.union(job_skills))
                    jaccard_score = intersection / union if union > 0 else 0.0
                else:
                    jaccard_score = 0.0
                
                return {
                    'score': jaccard_score,
                    'confidence': 0.8,
                    'explanation': f'Skill match score: {jaccard_score:.3f}',
                    'model_type': 'feature'
                }
            
            def match_batch(self, resume_features: Dict[str, Any], jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                return [self.match_single(resume_features, job) for job in jobs]
        
        return FeatureMatcherPlaceholder(config)

    def _create_feedback_model_placeholder(self, config: Dict[str, Any]) -> Any:
        """Create a placeholder feedback model."""
        class FeedbackModelPlaceholder:
            def __init__(self, config):
                self.config = config
            
            def match_single(self, resume_features: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
                # Placeholder logic based on career level and job requirements
                career_level = resume_features.get('metadata', {}).get('career_level', 'entry')
                
                # Simple scoring based on career progression
                level_scores = {
                    'entry': 0.6,
                    'junior': 0.65,
                    'mid': 0.7,
                    'senior': 0.75,
                    'executive': 0.8
                }
                
                base_score = level_scores.get(career_level, 0.6)
                
                return {
                    'score': base_score,
                    'confidence': 0.6,
                    'explanation': f'Feedback-based score for {career_level} level',
                    'model_type': 'feedback'
                }
            
            def match_batch(self, resume_features: Dict[str, Any], jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                return [self.match_single(resume_features, job) for job in jobs]
        
        return FeedbackModelPlaceholder(config)

    def _extract_job_text(self, job: Dict[str, Any]) -> str:
        """Extract text content from job for semantic analysis."""
        text_parts = []
        
        # Add job details
        if job.get('job_title'):
            text_parts.append(job['job_title'])
        if job.get('description'):
            text_parts.append(job['description'])
        
        # Add requirements and responsibilities
        for field in ['requirements', 'responsibilities', 'qualifications']:
            if job.get(field):
                if isinstance(job[field], list):
                    text_parts.extend(job[field])
                else:
                    text_parts.append(str(job[field]))
        
        # Add skills
        if job.get('skills'):
            if isinstance(job['skills'], list):
                text_parts.extend(job['skills'])
            else:
                text_parts.append(str(job['skills']))
        
        return ' '.join(text_parts)

    def _is_model_valid(self, model_name: str) -> bool:
        """Check if a cached model is still valid."""
        if model_name not in self.model_load_times:
            return False
        
        # Check if model has expired
        load_time = self.model_load_times[model_name]
        ttl = timedelta(seconds=self.cache_config['model_ttl'])
        
        return datetime.now() - load_time < ttl

    def _update_access_stats(self, model_name: str):
        """Update access statistics for a model."""
        self.access_counts[model_name] = self.access_counts.get(model_name, 0) + 1
        if model_name in self.model_metadata:
            self.model_metadata[model_name]['last_accessed'] = datetime.now()
            self.model_metadata[model_name]['access_count'] = self.access_counts[model_name]

    def _check_memory_availability(self) -> bool:
        """Check if there's enough memory to load another model."""
        # Simple heuristic - allow up to max_cached_models
        return len(self.loaded_models) < self.max_cached_models

    async def _evict_least_used_model(self):
        """Evict the least recently used model to free memory."""
        if not self.loaded_models:
            return
        
        # Find the least recently used model
        lru_model = min(
            self.model_metadata.keys(),
            key=lambda name: self.model_metadata[name]['last_accessed']
        )
        
        logger.info(f"Evicting least used model: {lru_model}")
        await self._unload_model(lru_model)

    async def _unload_model(self, model_name: str):
        """Unload a model from memory."""
        try:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
            if model_name in self.model_load_times:
                del self.model_load_times[model_name]
            if model_name in self.model_metadata:
                del self.model_metadata[model_name]
            if model_name in self.access_counts:
                del self.access_counts[model_name]
            
            logger.info(f"Model {model_name} unloaded from memory")
            
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {str(e)}")

    def _estimate_model_memory(self, model: Any) -> float:
        """Estimate memory usage of a model in MB."""
        try:
            import sys
            return sys.getsizeof(model) / (1024 * 1024)  # Convert to MB
        except:
            return 100.0  # Default estimate

    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the model registry."""
        return {
            'loaded_models': list(self.loaded_models.keys()),
            'model_metadata': self.model_metadata.copy(),
            'access_counts': self.access_counts.copy(),
            'available_models': list(self.model_definitions.keys()),
            'cache_config': self.cache_config.copy()
        }

    async def preload_models(self, model_names: Optional[List[str]] = None):
        """Preload specified models or all high-priority models."""
        if model_names is None:
            # Load models by priority
            sorted_models = sorted(
                self.model_definitions.items(),
                key=lambda x: x[1]['priority']
            )
            model_names = [name for name, _ in sorted_models[:3]]  # Load top 3
        
        logger.info(f"Preloading models: {model_names}")
        
        for model_name in model_names:
            try:
                await self.get_model(model_name)
            except Exception as e:
                logger.error(f"Failed to preload model {model_name}: {str(e)}")

    async def cleanup(self):
        """Clean up all loaded models and resources."""
        try:
            model_names = list(self.loaded_models.keys())
            for model_name in model_names:
                await self._unload_model(model_name)
            
            logger.info("ModelRegistry cleanup completed")
            
        except Exception as e:
            logger.error(f"Registry cleanup failed: {str(e)}")