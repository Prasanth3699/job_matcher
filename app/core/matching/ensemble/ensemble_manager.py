"""
Ensemble Matching Engine for advanced ML pipeline.

This module orchestrates multiple specialized models to provide superior
resume-job matching accuracy through ensemble learning techniques.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path

from app.utils.logger import logger
from app.core.constants.ml_constants import (
    EnsembleConfig,
    ModelParameters,
    PerformanceMetrics,
    CachingParameters,
)
from app.core.matching.ensemble.model_registry import ModelRegistry
from app.core.matching.ensemble.weight_optimizer import WeightOptimizer
from app.core.matching.ensemble.ensemble_scorer import EnsembleScorer
from app.core.cache.embedding_cache import EmbeddingCache
from app.utils.serialization import make_json_serializable


class EnsembleMatchingEngine:
    """
    Advanced ensemble matching engine that combines multiple specialized models
    for superior resume-job matching accuracy and performance.
    """

    def __init__(self):
        """Initialize the ensemble matching engine with all required components."""
        self.model_registry = ModelRegistry()
        self.weight_optimizer = WeightOptimizer()
        self.ensemble_scorer = EnsembleScorer()
        self.embedding_cache = EmbeddingCache()

        # Performance tracking
        self.performance_metrics = {
            "total_matches": 0,
            "cache_hits": 0,
            "average_latency": 0.0,
            "model_usage_stats": {},
        }

        # Model weights (can be dynamically optimized)
        self.current_weights = EnsembleConfig.MODEL_WEIGHTS.copy()

        logger.info("EnsembleMatchingEngine initialized successfully")

    async def initialize(self):
        """
        Async initialization method for interface compatibility.
        EnsembleMatchingEngine initialization is already completed in __init__.
        """
        logger.info("EnsembleMatchingEngine async initialization completed")

    def match_resume_to_jobs_sync(
        self,
        resume_data: Dict[str, Any],
        jobs: List[Dict[str, Any]],
        top_n: int = 5,
        include_explanations: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Blocking wrapper around the async match_resume_to_jobs for use in Celery worker threads.
        Spins up a temporary event loop, executes the coroutine, and closes the loop safely.
        """
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                self.match_resume_to_jobs(
                    resume_data=resume_data,
                    jobs=jobs,
                    top_n=top_n,
                    include_explanations=include_explanations,
                )
            )
        finally:
            try:
                loop.run_until_complete(asyncio.sleep(0))
            except Exception:
                pass
            loop.close()

    async def match_resume_to_jobs(
        self,
        resume_data: Dict[str, Any],
        jobs: List[Dict[str, Any]],
        top_n: int = 5,
        include_explanations: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Match a resume against multiple jobs using ensemble learning.

        Args:
            resume_data: Parsed resume data with skills, experience, etc.
            jobs: List of job dictionaries with requirements and details
            top_n: Number of top matches to return
            include_explanations: Whether to include match explanations

        Returns:
            List of match results with scores and explanations
        """
        start_time = time.time()

        try:
            # Validate inputs
            if not resume_data or not jobs:
                logger.warning("Empty resume data or jobs list provided")
                return []

            # Generate unique cache key for this matching request
            cache_key = self._generate_cache_key(resume_data, jobs)

            # Try to get cached results first
            cached_results = await self.embedding_cache.get_cached_results(cache_key)
            if cached_results:
                self.performance_metrics["cache_hits"] += 1
                logger.info(
                    f"Cache hit for matching request - {len(cached_results)} results"
                )
                return cached_results[:top_n]

            # Load and prepare models
            models = await self._prepare_models()
            if not models:
                logger.error("No models available for ensemble matching")
                return []

            # Extract resume features once for all models
            resume_features = await self._extract_resume_features(resume_data)

            # Process jobs in batches for better performance
            job_batches = self._create_job_batches(jobs)
            all_match_results = []

            for batch in job_batches:
                batch_results = await self._process_job_batch(
                    resume_features, batch, models, include_explanations
                )
                all_match_results.extend(batch_results)

            # Sort by overall score and take top N
            sorted_results = sorted(
                all_match_results,
                key=lambda x: x.get("overall_score", 0.0),
                reverse=True,
            )[:top_n]

            # Cache the results for future requests
            await self.embedding_cache.cache_results(cache_key, sorted_results)

            # Update performance metrics
            self._update_performance_metrics(start_time, len(sorted_results))

            logger.info(
                f"Ensemble matching completed: {len(sorted_results)} matches in {time.time() - start_time:.2f}s"
            )

            return sorted_results

        except Exception as e:
            logger.error(f"Ensemble matching failed: {str(e)}", exc_info=True)
            # Fallback to single model if ensemble fails
            return await self._fallback_matching(resume_data, jobs, top_n)

    async def _prepare_models(self) -> Dict[str, Any]:
        """Load and prepare all ensemble models."""
        try:
            models = {}

            # Load primary semantic model
            primary_model = await self.model_registry.get_model("primary_semantic")
            if primary_model:
                models["primary_semantic"] = primary_model

            # Load fast semantic model
            fast_model = await self.model_registry.get_model("fast_semantic")
            if fast_model:
                models["fast_semantic"] = fast_model

            # Load domain-specific model
            domain_model = await self.model_registry.get_model("domain_specific")
            if domain_model:
                models["domain_specific"] = domain_model

            # Load feature-based model
            feature_model = await self.model_registry.get_model("feature_based")
            if feature_model:
                models["feature_based"] = feature_model

            # Load feedback-learned model
            feedback_model = await self.model_registry.get_model("feedback_learned")
            if feedback_model:
                models["feedback_learned"] = feedback_model

            logger.info(f"Loaded {len(models)} models for ensemble matching")
            return models

        except Exception as e:
            logger.error(f"Failed to prepare models: {str(e)}")
            return {}

    async def _extract_resume_features(
        self, resume_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract comprehensive features from resume data."""
        try:
            features = {
                "text_content": self._extract_text_content(resume_data),
                "skills": resume_data.get("skills", []),
                "experience": resume_data.get("experiences", []),
                "education": resume_data.get("education", []),
                "certifications": resume_data.get("certifications", []),
                "normalized_features": resume_data.get("normalized_features", {}),
                "metadata": {
                    "total_experience_years": self._calculate_total_experience(
                        resume_data
                    ),
                    "skill_count": len(resume_data.get("skills", [])),
                    "education_level": self._determine_education_level(resume_data),
                    "career_level": self._determine_career_level(resume_data),
                },
            }

            return features

        except Exception as e:
            logger.error(f"Failed to extract resume features: {str(e)}")
            return {}

    def _extract_text_content(self, resume_data: Dict[str, Any]) -> str:
        """Extract all text content from resume for semantic analysis."""
        text_parts = []

        # Add basic info
        if resume_data.get("summary"):
            text_parts.append(resume_data["summary"])

        # Add experience descriptions
        for exp in resume_data.get("experiences", []):
            if exp.get("description"):
                text_parts.append(exp["description"])
            if exp.get("responsibilities"):
                text_parts.extend(exp["responsibilities"])

        # Add education details
        for edu in resume_data.get("education", []):
            if edu.get("degree"):
                text_parts.append(edu["degree"])
            if edu.get("field_of_study"):
                text_parts.append(edu["field_of_study"])

        # Add skills
        text_parts.extend(resume_data.get("skills", []))

        return " ".join(text_parts)

    def _calculate_total_experience(self, resume_data: Dict[str, Any]) -> float:
        """Calculate total years of experience from resume."""
        try:
            total_months = 0
            for exp in resume_data.get("experiences", []):
                if exp.get("duration_months"):
                    total_months += exp["duration_months"]
            return total_months / 12.0
        except:
            return 0.0

    def _determine_education_level(self, resume_data: Dict[str, Any]) -> str:
        """Determine the highest education level."""
        education_levels = {
            "phd": 5,
            "doctorate": 5,
            "doctoral": 5,
            "master": 4,
            "masters": 4,
            "mba": 4,
            "ms": 4,
            "ma": 4,
            "bachelor": 3,
            "bachelors": 3,
            "bs": 3,
            "ba": 3,
            "be": 3,
            "associate": 2,
            "diploma": 2,
            "certificate": 1,
            "certification": 1,
        }

        highest_level = 0
        highest_degree = "none"

        for edu in resume_data.get("education", []):
            degree = edu.get("degree", "").lower()
            for level_name, level_value in education_levels.items():
                if level_name in degree and level_value > highest_level:
                    highest_level = level_value
                    highest_degree = level_name

        return highest_degree

    def _determine_career_level(self, resume_data: Dict[str, Any]) -> str:
        """Determine career level based on experience and roles."""
        total_years = self._calculate_total_experience(resume_data)

        # Check for leadership roles
        leadership_keywords = [
            "manager",
            "director",
            "lead",
            "senior",
            "principal",
            "architect",
            "vp",
            "cto",
            "ceo",
        ]
        has_leadership = any(
            any(
                keyword in exp.get("job_title", "").lower()
                for keyword in leadership_keywords
            )
            for exp in resume_data.get("experiences", [])
        )

        if total_years >= 10 and has_leadership:
            return "executive"
        elif total_years >= 7:
            return "senior"
        elif total_years >= 3:
            return "mid"
        elif total_years >= 1:
            return "junior"
        else:
            return "entry"

    def _create_job_batches(
        self, jobs: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Create batches of jobs for efficient processing."""
        batch_size = EnsembleConfig.CACHE_CONFIG.get("batch_cache_size", 100)
        batches = []

        for i in range(0, len(jobs), batch_size):
            batch = jobs[i : i + batch_size]
            batches.append(batch)

        return batches

    async def _process_job_batch(
        self,
        resume_features: Dict[str, Any],
        job_batch: List[Dict[str, Any]],
        models: Dict[str, Any],
        include_explanations: bool,
    ) -> List[Dict[str, Any]]:
        """Process a batch of jobs against the resume using all models."""
        batch_results = []

        # Use ThreadPoolExecutor for parallel model execution
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            # Create tasks for each model
            model_tasks = []
            for model_name, model in models.items():
                if model_name in self.current_weights:
                    task = executor.submit(
                        self._run_model_batch,
                        model_name,
                        model,
                        resume_features,
                        job_batch,
                    )
                    model_tasks.append((model_name, task))

            # Collect results from all models
            model_results = {}
            for model_name, task in model_tasks:
                try:
                    results = task.result(timeout=30)  # 30 second timeout per model
                    model_results[model_name] = results
                    self.performance_metrics["model_usage_stats"][model_name] = (
                        self.performance_metrics["model_usage_stats"].get(model_name, 0)
                        + 1
                    )
                except Exception as e:
                    logger.error(f"Model {model_name} failed: {str(e)}")
                    model_results[model_name] = []

        # Combine results using ensemble scoring
        for i, job in enumerate(job_batch):
            job_scores = {}
            for model_name, results in model_results.items():
                if i < len(results):
                    job_scores[model_name] = results[i]

            # Calculate ensemble score
            ensemble_result = self.ensemble_scorer.calculate_ensemble_score(
                job_scores, self.current_weights, include_explanations
            )

            # Add job details
            ensemble_result["job_details"] = job
            ensemble_result["job_id"] = job.get("job_id", job.get("id", ""))

            batch_results.append(ensemble_result)

        return batch_results

    def _run_model_batch(
        self,
        model_name: str,
        model: Any,
        resume_features: Dict[str, Any],
        job_batch: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run a specific model on a batch of jobs."""
        try:
            if hasattr(model, "match_batch"):
                # Use batch processing if available
                return model.match_batch(resume_features, job_batch)
            else:
                # Fall back to individual matching
                results = []
                for job in job_batch:
                    result = model.match_single(resume_features, job)
                    results.append(result)
                return results
        except Exception as e:
            logger.error(f"Model {model_name} batch processing failed: {str(e)}")
            return [{"score": 0.0, "confidence": 0.0} for _ in job_batch]

    async def _fallback_matching(
        self, resume_data: Dict[str, Any], jobs: List[Dict[str, Any]], top_n: int
    ) -> List[Dict[str, Any]]:
        """Fallback to single model matching if ensemble fails."""
        try:
            logger.info("Using fallback matching with primary model")

            # Try to load just the primary semantic model
            primary_model = await self.model_registry.get_model("primary_semantic")
            if not primary_model:
                logger.error("No fallback model available")
                return []

            resume_features = await self._extract_resume_features(resume_data)
            results = []

            for job in jobs:
                try:
                    result = primary_model.match_single(resume_features, job)
                    result["job_details"] = job
                    result["job_id"] = job.get("job_id", job.get("id", ""))
                    result["overall_score"] = result.get("score", 0.0)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Fallback matching failed for job: {str(e)}")
                    continue

            # Sort and return top N
            sorted_results = sorted(
                results, key=lambda x: x.get("overall_score", 0.0), reverse=True
            )

            return sorted_results[:top_n]

        except Exception as e:
            logger.error(f"Fallback matching failed completely: {str(e)}")
            return []

    def _generate_cache_key(
        self, resume_data: Dict[str, Any], jobs: List[Dict[str, Any]]
    ) -> str:
        """Generate a unique cache key for the matching request."""
        import hashlib

        # Create a hash based on resume skills and job IDs
        resume_skills = str(sorted(resume_data.get("skills", [])))
        job_ids = str(sorted([job.get("job_id", job.get("id", "")) for job in jobs]))

        cache_string = f"{resume_skills}_{job_ids}_{len(jobs)}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _update_performance_metrics(self, start_time: float, result_count: int):
        """Update internal performance metrics."""
        processing_time = time.time() - start_time

        self.performance_metrics["total_matches"] += 1

        # Update average latency (rolling average)
        current_avg = self.performance_metrics["average_latency"]
        total_matches = self.performance_metrics["total_matches"]

        self.performance_metrics["average_latency"] = (
            current_avg * (total_matches - 1) + processing_time
        ) / total_matches

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            "ensemble_metrics": self.performance_metrics.copy(),
            "cache_metrics": await self.embedding_cache.get_cache_stats(),
            "model_weights": self.current_weights.copy(),
            "cache_hit_rate": (
                self.performance_metrics["cache_hits"]
                / max(self.performance_metrics["total_matches"], 1)
            ),
        }

    async def optimize_weights(self, feedback_data: List[Dict[str, Any]]):
        """Optimize ensemble weights based on user feedback."""
        try:
            if not feedback_data:
                logger.warning("No feedback data provided for weight optimization")
                return

            # Use weight optimizer to adjust weights
            new_weights = await self.weight_optimizer.optimize_weights(
                current_weights=self.current_weights,
                feedback_data=feedback_data,
                performance_metrics=self.performance_metrics,
            )

            if new_weights:
                self.current_weights = new_weights
                logger.info(f"Updated ensemble weights: {new_weights}")

        except Exception as e:
            logger.error(f"Weight optimization failed: {str(e)}")

    async def cleanup(self):
        """Clean up resources and cache."""
        try:
            await self.embedding_cache.cleanup()
            await self.model_registry.cleanup()
            logger.info("EnsembleMatchingEngine cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
