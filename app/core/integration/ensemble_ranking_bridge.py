"""
Ensemble Ranking Bridge for integrating ensemble scoring with neural ranking.

This module provides seamless integration between ensemble matching scores
and neural ranking predictions for optimal job matching performance.
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from app.utils.logger import logger
from app.core.constants.ml_constants import (
    EnsembleConfig,
    RankingConfig,
    PerformanceMetrics
)
from app.core.matching.ensemble.ensemble_manager import EnsembleMatchingEngine
from app.core.ranking.neural_ranker import NeuralRanker, RankingPrediction
from app.core.ranking.rank_explainer import RankExplainer


class IntegrationStrategy(str, Enum):
    """Strategies for integrating ensemble and ranking scores."""
    ENSEMBLE_FIRST = "ensemble_first"      # Ensemble scoring then ranking
    RANKING_FIRST = "ranking_first"        # Ranking first then ensemble
    PARALLEL = "parallel"                  # Parallel processing and combination
    ADAPTIVE = "adaptive"                  # Adaptive based on performance
    HYBRID = "hybrid"                      # Advanced hybrid approach


@dataclass
class IntegratedResult:
    """Result combining ensemble and ranking predictions."""
    job_id: str
    final_score: float
    ensemble_score: float
    ranking_score: float
    confidence: float
    explanation: Dict[str, Any]
    job_details: Dict[str, Any]
    ranking_position: int


class EnsembleRankingBridge:
    """
    Advanced bridge connecting ensemble matching with neural ranking
    for superior job matching accuracy and user satisfaction.
    """
    
    def __init__(self, integration_strategy: IntegrationStrategy = IntegrationStrategy.HYBRID):
        """Initialize the ensemble ranking bridge."""
        self.integration_strategy = integration_strategy
        
        # Core components
        self.ensemble_engine: Optional[EnsembleMatchingEngine] = None
        self.neural_ranker: Optional[NeuralRanker] = None
        self.rank_explainer: Optional[RankExplainer] = None
        
        # Integration parameters
        self.integration_weights = {
            'ensemble_weight': 0.6,      # Ensemble contribution
            'ranking_weight': 0.4,       # Ranking contribution
            'confidence_boost': 0.1,     # Confidence bonus for agreement
            'diversity_factor': 0.05     # Diversity consideration
        }
        
        # Performance tracking
        self.integration_stats = {
            'total_integrations': 0,
            'ensemble_wins': 0,
            'ranking_wins': 0,
            'score_agreements': 0,
            'average_score_difference': 0.0,
            'processing_times': []
        }
        
        # Adaptive parameters
        self.adaptive_config = {
            'performance_window': 1000,  # Recent predictions to consider
            'adaptation_threshold': 0.1,  # Threshold for weight adjustment
            'max_weight_change': 0.05,   # Maximum weight change per adaptation
            'min_weight': 0.2,           # Minimum weight for either system
            'max_weight': 0.8            # Maximum weight for either system
        }
        
        logger.info(f"EnsembleRankingBridge initialized with strategy: {integration_strategy}")
    
    async def initialize(self):
        """Initialize all components of the bridge."""
        try:
            # Initialize ensemble engine
            self.ensemble_engine = EnsembleMatchingEngine()
            await self.ensemble_engine.initialize()
            
            # Initialize neural ranker
            self.neural_ranker = NeuralRanker()
            await self.neural_ranker.initialize()
            
            # Initialize rank explainer
            self.rank_explainer = RankExplainer()
            
            logger.info("EnsembleRankingBridge components initialized successfully")
            
        except Exception as e:
            logger.error(f"Bridge initialization failed: {str(e)}")
            raise
    
    async def integrated_match(
        self,
        resume_data: Dict[str, Any],
        jobs: List[Dict[str, Any]],
        top_n: int = 5,
        include_explanations: bool = True,
        strategy_override: Optional[IntegrationStrategy] = None
    ) -> List[IntegratedResult]:
        """
        Perform integrated matching using both ensemble and ranking.
        
        Args:
            resume_data: Parsed resume data
            jobs: List of job descriptions
            top_n: Number of top matches to return
            include_explanations: Whether to include detailed explanations
            strategy_override: Override default integration strategy
            
        Returns:
            List of integrated matching results
        """
        try:
            start_time = datetime.now()
            
            if not self.ensemble_engine or not self.neural_ranker:
                raise ValueError("Bridge components not initialized")
            
            strategy = strategy_override or self.integration_strategy
            
            # Execute based on strategy
            if strategy == IntegrationStrategy.ENSEMBLE_FIRST:
                results = await self._ensemble_first_integration(
                    resume_data, jobs, top_n, include_explanations
                )
            elif strategy == IntegrationStrategy.RANKING_FIRST:
                results = await self._ranking_first_integration(
                    resume_data, jobs, top_n, include_explanations
                )
            elif strategy == IntegrationStrategy.PARALLEL:
                results = await self._parallel_integration(
                    resume_data, jobs, top_n, include_explanations
                )
            elif strategy == IntegrationStrategy.ADAPTIVE:
                results = await self._adaptive_integration(
                    resume_data, jobs, top_n, include_explanations
                )
            elif strategy == IntegrationStrategy.HYBRID:
                results = await self._hybrid_integration(
                    resume_data, jobs, top_n, include_explanations
                )
            else:
                raise ValueError(f"Unknown integration strategy: {strategy}")
            
            # Update performance tracking
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_integration_stats(results, processing_time)
            
            # Adaptive weight adjustment
            if strategy == IntegrationStrategy.ADAPTIVE:
                await self._adjust_adaptive_weights(results)
            
            logger.info(f"Integrated matching completed: {len(results)} results in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Integrated matching failed: {str(e)}")
            return []
    
    async def _ensemble_first_integration(
        self,
        resume_data: Dict[str, Any],
        jobs: List[Dict[str, Any]],
        top_n: int,
        include_explanations: bool
    ) -> List[IntegratedResult]:
        """Integration strategy: Ensemble scoring first, then ranking refinement."""
        try:
            # Step 1: Get ensemble scores for all jobs
            ensemble_results = await self.ensemble_engine.match_resume_to_jobs(
                resume_data, jobs, top_n=len(jobs), include_explanations=False
            )
            
            if not ensemble_results:
                return []
            
            # Step 2: Pre-filter to top candidates (2x top_n for ranking)
            pre_filter_count = min(top_n * 2, len(ensemble_results))
            top_ensemble_results = ensemble_results[:pre_filter_count]
            
            # Extract jobs for ranking
            ranking_jobs = []
            ensemble_scores = {}
            
            for result in top_ensemble_results:
                job_details = result.get('job_details', {})
                job_id = result.get('job_id', '')
                
                ranking_jobs.append(job_details)
                ensemble_scores[job_id] = result.get('overall_score', 0.0)
            
            # Step 3: Apply neural ranking to pre-filtered results
            ranking_prediction = await self.neural_ranker.predict_rankings(
                resume_data, ranking_jobs
            )
            
            # Step 4: Combine scores
            integrated_results = []
            
            for i, (ensemble_result, ranking_score) in enumerate(
                zip(top_ensemble_results, ranking_prediction.scores)
            ):
                job_id = ensemble_result.get('job_id', '')
                ensemble_score = ensemble_result.get('overall_score', 0.0)
                
                # Weighted combination
                final_score = (
                    self.integration_weights['ensemble_weight'] * ensemble_score +
                    self.integration_weights['ranking_weight'] * ranking_score
                )
                
                # Confidence calculation
                score_agreement = 1.0 - abs(ensemble_score - ranking_score)
                confidence = (
                    ensemble_result.get('confidence', 0.5) * 0.5 +
                    ranking_prediction.confidence * 0.5 +
                    score_agreement * self.integration_weights['confidence_boost']
                )
                
                # Generate explanation if needed
                explanation = {}
                if include_explanations:
                    explanation = await self._generate_integration_explanation(
                        ensemble_result, ranking_score, ranking_prediction, i
                    )
                
                integrated_results.append(IntegratedResult(
                    job_id=job_id,
                    final_score=final_score,
                    ensemble_score=ensemble_score,
                    ranking_score=ranking_score,
                    confidence=min(1.0, confidence),
                    explanation=explanation,
                    job_details=ensemble_result.get('job_details', {}),
                    ranking_position=i + 1
                ))
            
            # Final ranking by integrated score
            integrated_results.sort(key=lambda x: x.final_score, reverse=True)
            
            return integrated_results[:top_n]
            
        except Exception as e:
            logger.error(f"Ensemble-first integration failed: {str(e)}")
            return []
    
    async def _ranking_first_integration(
        self,
        resume_data: Dict[str, Any],
        jobs: List[Dict[str, Any]],
        top_n: int,
        include_explanations: bool
    ) -> List[IntegratedResult]:
        """Integration strategy: Neural ranking first, then ensemble refinement."""
        try:
            # Step 1: Get neural ranking predictions
            ranking_prediction = await self.neural_ranker.predict_rankings(
                resume_data, jobs
            )
            
            if not ranking_prediction.scores:
                return []
            
            # Step 2: Sort jobs by ranking scores
            job_ranking_pairs = list(zip(jobs, ranking_prediction.scores))
            job_ranking_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Step 3: Take top candidates for ensemble scoring
            pre_filter_count = min(top_n * 2, len(job_ranking_pairs))
            top_ranked_jobs = [pair[0] for pair in job_ranking_pairs[:pre_filter_count]]
            top_ranking_scores = [pair[1] for pair in job_ranking_pairs[:pre_filter_count]]
            
            # Step 4: Apply ensemble scoring to top-ranked jobs
            ensemble_results = await self.ensemble_engine.match_resume_to_jobs(
                resume_data, top_ranked_jobs, top_n=len(top_ranked_jobs), include_explanations=False
            )
            
            # Step 5: Combine scores
            integrated_results = []
            
            for i, (ensemble_result, ranking_score) in enumerate(
                zip(ensemble_results, top_ranking_scores)
            ):
                job_id = ensemble_result.get('job_id', '')
                ensemble_score = ensemble_result.get('overall_score', 0.0)
                
                # Weighted combination (ranking gets more weight in this strategy)
                final_score = (
                    self.integration_weights['ranking_weight'] * ranking_score +
                    self.integration_weights['ensemble_weight'] * ensemble_score
                )
                
                # Confidence calculation
                score_agreement = 1.0 - abs(ensemble_score - ranking_score)
                confidence = (
                    ranking_prediction.confidence * 0.6 +
                    ensemble_result.get('confidence', 0.5) * 0.4 +
                    score_agreement * self.integration_weights['confidence_boost']
                )
                
                # Generate explanation if needed
                explanation = {}
                if include_explanations:
                    explanation = await self._generate_integration_explanation(
                        ensemble_result, ranking_score, ranking_prediction, i
                    )
                
                integrated_results.append(IntegratedResult(
                    job_id=job_id,
                    final_score=final_score,
                    ensemble_score=ensemble_score,
                    ranking_score=ranking_score,
                    confidence=min(1.0, confidence),
                    explanation=explanation,
                    job_details=ensemble_result.get('job_details', {}),
                    ranking_position=i + 1
                ))
            
            # Final ranking by integrated score
            integrated_results.sort(key=lambda x: x.final_score, reverse=True)
            
            return integrated_results[:top_n]
            
        except Exception as e:
            logger.error(f"Ranking-first integration failed: {str(e)}")
            return []
    
    async def _parallel_integration(
        self,
        resume_data: Dict[str, Any],
        jobs: List[Dict[str, Any]],
        top_n: int,
        include_explanations: bool
    ) -> List[IntegratedResult]:
        """Integration strategy: Parallel execution and score combination."""
        try:
            # Execute both systems in parallel
            ensemble_task = asyncio.create_task(
                self.ensemble_engine.match_resume_to_jobs(
                    resume_data, jobs, top_n=len(jobs), include_explanations=False
                )
            )
            
            ranking_task = asyncio.create_task(
                self.neural_ranker.predict_rankings(resume_data, jobs)
            )
            
            # Wait for both results
            ensemble_results, ranking_prediction = await asyncio.gather(
                ensemble_task, ranking_task
            )
            
            if not ensemble_results or not ranking_prediction.scores:
                return []
            
            # Combine results
            integrated_results = []
            
            for i, (ensemble_result, ranking_score) in enumerate(
                zip(ensemble_results, ranking_prediction.scores)
            ):
                job_id = ensemble_result.get('job_id', '')
                ensemble_score = ensemble_result.get('overall_score', 0.0)
                
                # Equal weight combination
                final_score = (ensemble_score + ranking_score) / 2.0
                
                # Confidence based on agreement
                score_agreement = 1.0 - abs(ensemble_score - ranking_score)
                confidence = (
                    ensemble_result.get('confidence', 0.5) +
                    ranking_prediction.confidence +
                    score_agreement * self.integration_weights['confidence_boost']
                ) / 2.0
                
                # Generate explanation if needed
                explanation = {}
                if include_explanations:
                    explanation = await self._generate_integration_explanation(
                        ensemble_result, ranking_score, ranking_prediction, i
                    )
                
                integrated_results.append(IntegratedResult(
                    job_id=job_id,
                    final_score=final_score,
                    ensemble_score=ensemble_score,
                    ranking_score=ranking_score,
                    confidence=min(1.0, confidence),
                    explanation=explanation,
                    job_details=ensemble_result.get('job_details', {}),
                    ranking_position=i + 1
                ))
            
            # Final ranking by integrated score
            integrated_results.sort(key=lambda x: x.final_score, reverse=True)
            
            return integrated_results[:top_n]
            
        except Exception as e:
            logger.error(f"Parallel integration failed: {str(e)}")
            return []
    
    async def _adaptive_integration(
        self,
        resume_data: Dict[str, Any],
        jobs: List[Dict[str, Any]],
        top_n: int,
        include_explanations: bool
    ) -> List[IntegratedResult]:
        """Integration strategy: Adaptive based on recent performance."""
        try:
            # Use current adaptive weights
            current_weights = self._get_current_adaptive_weights()
            
            # Execute both systems in parallel
            ensemble_results, ranking_prediction = await asyncio.gather(
                self.ensemble_engine.match_resume_to_jobs(
                    resume_data, jobs, top_n=len(jobs), include_explanations=False
                ),
                self.neural_ranker.predict_rankings(resume_data, jobs)
            )
            
            if not ensemble_results or not ranking_prediction.scores:
                return []
            
            # Combine with adaptive weights
            integrated_results = []
            
            for i, (ensemble_result, ranking_score) in enumerate(
                zip(ensemble_results, ranking_prediction.scores)
            ):
                job_id = ensemble_result.get('job_id', '')
                ensemble_score = ensemble_result.get('overall_score', 0.0)
                
                # Adaptive weighted combination
                final_score = (
                    current_weights['ensemble'] * ensemble_score +
                    current_weights['ranking'] * ranking_score
                )
                
                # Confidence calculation
                score_agreement = 1.0 - abs(ensemble_score - ranking_score)
                confidence = (
                    ensemble_result.get('confidence', 0.5) * current_weights['ensemble'] +
                    ranking_prediction.confidence * current_weights['ranking'] +
                    score_agreement * self.integration_weights['confidence_boost']
                )
                
                # Generate explanation if needed
                explanation = {}
                if include_explanations:
                    explanation = await self._generate_integration_explanation(
                        ensemble_result, ranking_score, ranking_prediction, i
                    )
                    explanation['adaptive_weights'] = current_weights
                
                integrated_results.append(IntegratedResult(
                    job_id=job_id,
                    final_score=final_score,
                    ensemble_score=ensemble_score,
                    ranking_score=ranking_score,
                    confidence=min(1.0, confidence),
                    explanation=explanation,
                    job_details=ensemble_result.get('job_details', {}),
                    ranking_position=i + 1
                ))
            
            # Final ranking by integrated score
            integrated_results.sort(key=lambda x: x.final_score, reverse=True)
            
            return integrated_results[:top_n]
            
        except Exception as e:
            logger.error(f"Adaptive integration failed: {str(e)}")
            return []
    
    async def _hybrid_integration(
        self,
        resume_data: Dict[str, Any],
        jobs: List[Dict[str, Any]],
        top_n: int,
        include_explanations: bool
    ) -> List[IntegratedResult]:
        """Integration strategy: Advanced hybrid approach with multiple techniques."""
        try:
            # Execute both systems in parallel
            ensemble_results, ranking_prediction = await asyncio.gather(
                self.ensemble_engine.match_resume_to_jobs(
                    resume_data, jobs, top_n=len(jobs), include_explanations=False
                ),
                self.neural_ranker.predict_rankings(resume_data, jobs)
            )
            
            if not ensemble_results or not ranking_prediction.scores:
                return []
            
            # Advanced hybrid combination
            integrated_results = []
            
            for i, (ensemble_result, ranking_score) in enumerate(
                zip(ensemble_results, ranking_prediction.scores)
            ):
                job_id = ensemble_result.get('job_id', '')
                ensemble_score = ensemble_result.get('overall_score', 0.0)
                
                # Multi-factor combination
                base_score = (
                    self.integration_weights['ensemble_weight'] * ensemble_score +
                    self.integration_weights['ranking_weight'] * ranking_score
                )
                
                # Agreement bonus
                score_agreement = 1.0 - abs(ensemble_score - ranking_score)
                agreement_bonus = score_agreement * self.integration_weights['confidence_boost']
                
                # Diversity factor (position-based)
                diversity_bonus = (1.0 - i / len(jobs)) * self.integration_weights['diversity_factor']
                
                # Confidence-weighted boost
                ensemble_confidence = ensemble_result.get('confidence', 0.5)
                ranking_confidence = ranking_prediction.confidence
                confidence_weight = (ensemble_confidence + ranking_confidence) / 2.0
                
                # Final score calculation
                final_score = (
                    base_score +
                    agreement_bonus +
                    diversity_bonus
                ) * (0.8 + 0.2 * confidence_weight)  # Confidence multiplier
                
                # Overall confidence
                confidence = (
                    ensemble_confidence * 0.4 +
                    ranking_confidence * 0.4 +
                    score_agreement * 0.2
                )
                
                # Generate explanation if needed
                explanation = {}
                if include_explanations:
                    explanation = await self._generate_hybrid_explanation(
                        ensemble_result, ranking_score, ranking_prediction, i,
                        base_score, agreement_bonus, diversity_bonus, confidence_weight
                    )
                
                integrated_results.append(IntegratedResult(
                    job_id=job_id,
                    final_score=min(1.0, final_score),  # Cap at 1.0
                    ensemble_score=ensemble_score,
                    ranking_score=ranking_score,
                    confidence=min(1.0, confidence),
                    explanation=explanation,
                    job_details=ensemble_result.get('job_details', {}),
                    ranking_position=i + 1
                ))
            
            # Final ranking by integrated score
            integrated_results.sort(key=lambda x: x.final_score, reverse=True)
            
            return integrated_results[:top_n]
            
        except Exception as e:
            logger.error(f"Hybrid integration failed: {str(e)}")
            return []
    
    async def _generate_integration_explanation(
        self,
        ensemble_result: Dict[str, Any],
        ranking_score: float,
        ranking_prediction: RankingPrediction,
        position: int
    ) -> Dict[str, Any]:
        """Generate explanation for integrated result."""
        try:
            ensemble_score = ensemble_result.get('overall_score', 0.0)
            
            explanation = {
                'integration_method': self.integration_strategy.value,
                'score_breakdown': {
                    'ensemble_score': ensemble_score,
                    'ranking_score': ranking_score,
                    'score_difference': abs(ensemble_score - ranking_score),
                    'score_agreement': 1.0 - abs(ensemble_score - ranking_score)
                },
                'component_explanations': {
                    'ensemble': ensemble_result.get('explanation', ''),
                    'ranking': f"Neural ranking score: {ranking_score:.3f}"
                },
                'integration_weights': self.integration_weights.copy(),
                'position_in_ranking': position + 1
            }
            
            # Add detailed explanation from rank explainer if available
            if self.rank_explainer:
                job_details = ensemble_result.get('job_details', {})
                # Note: This would need resume_data passed to work properly
                # For now, we'll add a placeholder
                explanation['detailed_ranking_explanation'] = 'Available on request'
            
            return explanation
            
        except Exception as e:
            logger.error(f"Integration explanation generation failed: {str(e)}")
            return {'error': 'Failed to generate explanation'}
    
    async def _generate_hybrid_explanation(
        self,
        ensemble_result: Dict[str, Any],
        ranking_score: float,
        ranking_prediction: RankingPrediction,
        position: int,
        base_score: float,
        agreement_bonus: float,
        diversity_bonus: float,
        confidence_weight: float
    ) -> Dict[str, Any]:
        """Generate detailed explanation for hybrid integration."""
        try:
            ensemble_score = ensemble_result.get('overall_score', 0.0)
            
            explanation = {
                'integration_method': 'hybrid',
                'score_composition': {
                    'base_score': base_score,
                    'agreement_bonus': agreement_bonus,
                    'diversity_bonus': diversity_bonus,
                    'confidence_multiplier': 0.8 + 0.2 * confidence_weight
                },
                'component_scores': {
                    'ensemble_score': ensemble_score,
                    'ranking_score': ranking_score,
                    'ensemble_confidence': ensemble_result.get('confidence', 0.5),
                    'ranking_confidence': ranking_prediction.confidence
                },
                'calculation_details': {
                    'score_agreement': 1.0 - abs(ensemble_score - ranking_score),
                    'position_factor': 1.0 - position / 10,  # Position diversity
                    'weights_used': self.integration_weights.copy()
                },
                'interpretation': self._interpret_hybrid_score(
                    base_score, agreement_bonus, diversity_bonus
                )
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Hybrid explanation generation failed: {str(e)}")
            return {'error': 'Failed to generate hybrid explanation'}
    
    def _interpret_hybrid_score(
        self,
        base_score: float,
        agreement_bonus: float,
        diversity_bonus: float
    ) -> str:
        """Interpret hybrid score components."""
        interpretations = []
        
        if agreement_bonus > 0.05:
            interpretations.append("Strong agreement between ensemble and ranking models")
        elif agreement_bonus < 0.01:
            interpretations.append("Some disagreement between ensemble and ranking models")
        
        if diversity_bonus > 0.02:
            interpretations.append("Benefits from ranking diversity consideration")
        
        if base_score > 0.8:
            interpretations.append("Excellent base matching score")
        elif base_score > 0.6:
            interpretations.append("Good base matching score")
        else:
            interpretations.append("Moderate base matching score")
        
        return "; ".join(interpretations) if interpretations else "Standard hybrid scoring applied"
    
    def _get_current_adaptive_weights(self) -> Dict[str, float]:
        """Get current adaptive weights based on recent performance."""
        # For now, use configured weights
        # In production, this would analyze recent performance
        return {
            'ensemble': self.integration_weights['ensemble_weight'],
            'ranking': self.integration_weights['ranking_weight']
        }
    
    async def _adjust_adaptive_weights(self, results: List[IntegratedResult]):
        """Adjust adaptive weights based on result quality."""
        try:
            if len(results) < 5:  # Need sufficient data
                return
            
            # Analyze score agreement
            score_agreements = [
                1.0 - abs(r.ensemble_score - r.ranking_score) for r in results
            ]
            avg_agreement = np.mean(score_agreements)
            
            # Adjust weights based on agreement
            if avg_agreement > 0.8:  # High agreement - maintain current weights
                pass
            elif avg_agreement < 0.5:  # Low agreement - investigate further
                # This would trigger more sophisticated analysis
                logger.info(f"Low score agreement detected: {avg_agreement:.3f}")
            
            # Update adaptive configuration if needed
            # This is a placeholder for more sophisticated adaptation logic
            
        except Exception as e:
            logger.error(f"Adaptive weight adjustment failed: {str(e)}")
    
    def _update_integration_stats(self, results: List[IntegratedResult], processing_time: float):
        """Update integration performance statistics."""
        try:
            self.integration_stats['total_integrations'] += 1
            self.integration_stats['processing_times'].append(processing_time)
            
            # Keep only recent processing times
            if len(self.integration_stats['processing_times']) > 1000:
                self.integration_stats['processing_times'] = \
                    self.integration_stats['processing_times'][-1000:]
            
            # Analyze score patterns
            if results:
                score_differences = [
                    abs(r.ensemble_score - r.ranking_score) for r in results
                ]
                avg_difference = np.mean(score_differences)
                
                # Update rolling average
                current_avg = self.integration_stats['average_score_difference']
                total_integrations = self.integration_stats['total_integrations']
                
                self.integration_stats['average_score_difference'] = (
                    (current_avg * (total_integrations - 1) + avg_difference) / total_integrations
                )
                
                # Count agreements and wins
                for result in results:
                    if abs(result.ensemble_score - result.ranking_score) < 0.1:
                        self.integration_stats['score_agreements'] += 1
                    
                    if result.ensemble_score > result.ranking_score:
                        self.integration_stats['ensemble_wins'] += 1
                    else:
                        self.integration_stats['ranking_wins'] += 1
            
        except Exception as e:
            logger.error(f"Stats update failed: {str(e)}")
    
    async def update_integration_weights(self, new_weights: Dict[str, float]) -> bool:
        """Update integration weights configuration."""
        try:
            # Validate weights
            required_keys = ['ensemble_weight', 'ranking_weight']
            if not all(key in new_weights for key in required_keys):
                logger.error("Missing required weight keys")
                return False
            
            if abs(new_weights['ensemble_weight'] + new_weights['ranking_weight'] - 1.0) > 0.001:
                logger.error("Ensemble and ranking weights must sum to 1.0")
                return False
            
            # Update weights
            self.integration_weights.update(new_weights)
            
            logger.info(f"Updated integration weights: {new_weights}")
            return True
            
        except Exception as e:
            logger.error(f"Weight update failed: {str(e)}")
            return False
    
    async def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics."""
        try:
            # Calculate derived statistics
            total_integrations = self.integration_stats['total_integrations']
            
            if total_integrations > 0:
                ensemble_win_rate = self.integration_stats['ensemble_wins'] / total_integrations
                ranking_win_rate = self.integration_stats['ranking_wins'] / total_integrations
                agreement_rate = self.integration_stats['score_agreements'] / total_integrations
            else:
                ensemble_win_rate = ranking_win_rate = agreement_rate = 0.0
            
            # Processing time statistics
            processing_times = self.integration_stats['processing_times']
            if processing_times:
                avg_processing_time = np.mean(processing_times)
                p95_processing_time = np.percentile(processing_times, 95)
            else:
                avg_processing_time = p95_processing_time = 0.0
            
            return {
                'integration_stats': self.integration_stats.copy(),
                'derived_metrics': {
                    'ensemble_win_rate': ensemble_win_rate,
                    'ranking_win_rate': ranking_win_rate,
                    'score_agreement_rate': agreement_rate,
                    'avg_processing_time_s': avg_processing_time,
                    'p95_processing_time_s': p95_processing_time
                },
                'current_configuration': {
                    'integration_strategy': self.integration_strategy.value,
                    'integration_weights': self.integration_weights.copy(),
                    'adaptive_config': self.adaptive_config.copy()
                },
                'component_status': {
                    'ensemble_engine_ready': self.ensemble_engine is not None,
                    'neural_ranker_ready': self.neural_ranker is not None,
                    'rank_explainer_ready': self.rank_explainer is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get integration stats: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Clean up bridge resources."""
        try:
            # Clean up components
            if self.ensemble_engine:
                await self.ensemble_engine.cleanup()
            
            if self.neural_ranker:
                await self.neural_ranker.cleanup()
            
            # Reset stats
            self.integration_stats = {
                'total_integrations': 0,
                'ensemble_wins': 0,
                'ranking_wins': 0,
                'score_agreements': 0,
                'average_score_difference': 0.0,
                'processing_times': []
            }
            
            logger.info("EnsembleRankingBridge cleanup completed")
            
        except Exception as e:
            logger.error(f"Bridge cleanup failed: {str(e)}")