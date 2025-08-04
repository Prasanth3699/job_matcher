"""
Ensemble Scorer for combining multiple model predictions.

This module handles intelligent combination of scores from multiple models
with confidence weighting, explanation generation, and score calibration.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math

from app.utils.logger import logger
from app.core.constants.ml_constants import (
    EnsembleConfig,
    ScoringThresholds,
    PerformanceMetrics,
    MatchingWeights
)


@dataclass
class ScoredResult:
    """Container for a scored matching result."""
    overall_score: float
    confidence: float
    model_scores: Dict[str, float]
    model_confidences: Dict[str, float]
    explanation: Optional[str] = None
    score_breakdown: Optional[Dict[str, Any]] = None


class EnsembleScorer:
    """
    Advanced ensemble scoring engine that combines predictions from multiple
    models with intelligent weighting and confidence estimation.
    """

    def __init__(self):
        """Initialize the ensemble scorer with configuration."""
        self.default_weights = EnsembleConfig.MODEL_WEIGHTS.copy()
        self.score_thresholds = {
            'excellent': ScoringThresholds.EXCELLENT_MATCH,
            'very_good': ScoringThresholds.VERY_GOOD_MATCH,
            'good': ScoringThresholds.GOOD_MATCH,
            'fair': ScoringThresholds.FAIR_MATCH,
            'poor': ScoringThresholds.POOR_MATCH
        }
        
        # Calibration parameters
        self.calibration_params = {
            'temperature': 1.5,  # Temperature scaling for confidence calibration
            'bias_correction': 0.05,  # Bias correction factor
            'confidence_floor': 0.1,  # Minimum confidence level
            'confidence_ceiling': 0.95  # Maximum confidence level
        }
        
        logger.info("EnsembleScorer initialized successfully")

    def calculate_ensemble_score(
        self,
        model_scores: Dict[str, Dict[str, Any]],
        model_weights: Dict[str, float],
        include_explanations: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate ensemble score from multiple model predictions.
        
        Args:
            model_scores: Dictionary of model names to their score dictionaries
            model_weights: Weights for each model in the ensemble
            include_explanations: Whether to generate detailed explanations
            
        Returns:
            Dictionary containing ensemble score and metadata
        """
        try:
            if not model_scores:
                logger.warning("No model scores provided for ensemble calculation")
                return self._create_empty_result()
            
            # Extract scores and confidences from each model
            scores = {}
            confidences = {}
            valid_models = []
            
            for model_name, result in model_scores.items():
                if model_name in model_weights and result:
                    score = result.get('score', 0.0)
                    confidence = result.get('confidence', 0.0)
                    
                    # Validate score and confidence
                    if self._is_valid_score(score) and self._is_valid_score(confidence):
                        scores[model_name] = float(score)
                        confidences[model_name] = float(confidence)
                        valid_models.append(model_name)
                    else:
                        logger.warning(f"Invalid score/confidence from model {model_name}: {result}")
            
            if not valid_models:
                logger.error("No valid model scores available for ensemble calculation")
                return self._create_empty_result()
            
            # Calculate weighted ensemble score
            ensemble_result = self._calculate_weighted_score(
                scores, confidences, model_weights, valid_models
            )
            
            # Apply score calibration
            calibrated_result = self._calibrate_scores(ensemble_result)
            
            # Generate explanation if requested
            explanation = None
            if include_explanations:
                explanation = self._generate_explanation(
                    scores, confidences, model_weights, calibrated_result
                )
            
            # Create detailed score breakdown
            score_breakdown = self._create_score_breakdown(
                scores, confidences, model_weights, valid_models
            )
            
            return {
                'overall_score': calibrated_result.overall_score,
                'confidence': calibrated_result.confidence,
                'model_scores': scores,
                'model_confidences': confidences,
                'explanation': explanation,
                'score_breakdown': score_breakdown,
                'matching_quality': self._determine_match_quality(calibrated_result.overall_score),
                'ensemble_metadata': {
                    'valid_models': valid_models,
                    'total_models': len(model_scores),
                    'confidence_weighted': True,
                    'calibrated': True
                }
            }
            
        except Exception as e:
            logger.error(f"Ensemble scoring failed: {str(e)}", exc_info=True)
            return self._create_empty_result()

    def _calculate_weighted_score(
        self,
        scores: Dict[str, float],
        confidences: Dict[str, float],
        model_weights: Dict[str, float],
        valid_models: List[str]
    ) -> ScoredResult:
        """Calculate weighted ensemble score with confidence weighting."""
        try:
            # Normalize weights for valid models only
            valid_weights = {name: model_weights[name] for name in valid_models}
            total_weight = sum(valid_weights.values())
            
            if total_weight <= 0:
                logger.error("Total weight is zero or negative")
                return ScoredResult(0.0, 0.0, scores, confidences)
            
            normalized_weights = {name: w / total_weight for name, w in valid_weights.items()}
            
            # Calculate confidence-weighted ensemble score
            weighted_score = 0.0
            weighted_confidence = 0.0
            total_confidence_weight = 0.0
            
            for model_name in valid_models:
                model_score = scores[model_name]
                model_confidence = confidences[model_name]
                model_weight = normalized_weights[model_name]
                
                # Combine model weight with confidence
                effective_weight = model_weight * (0.5 + 0.5 * model_confidence)
                
                weighted_score += effective_weight * model_score
                weighted_confidence += effective_weight * model_confidence
                total_confidence_weight += effective_weight
            
            # Normalize by total effective weight
            if total_confidence_weight > 0:
                final_score = weighted_score / total_confidence_weight
                final_confidence = weighted_confidence / total_confidence_weight
            else:
                final_score = 0.0
                final_confidence = 0.0
            
            # Apply consensus bonus (higher confidence when models agree)
            consensus_bonus = self._calculate_consensus_bonus(scores, valid_models)
            final_confidence = min(1.0, final_confidence + consensus_bonus)
            
            return ScoredResult(
                overall_score=final_score,
                confidence=final_confidence,
                model_scores=scores,
                model_confidences=confidences
            )
            
        except Exception as e:
            logger.error(f"Weighted score calculation failed: {str(e)}")
            return ScoredResult(0.0, 0.0, scores, confidences)

    def _calculate_consensus_bonus(self, scores: Dict[str, float], valid_models: List[str]) -> float:
        """Calculate bonus confidence based on model consensus."""
        try:
            if len(valid_models) < 2:
                return 0.0
            
            score_values = [scores[model] for model in valid_models]
            
            # Calculate standard deviation of scores
            mean_score = np.mean(score_values)
            std_score = np.std(score_values)
            
            # Lower standard deviation = higher consensus = bonus confidence
            max_std = 0.3  # Maximum expected standard deviation
            consensus_ratio = max(0, (max_std - std_score) / max_std)
            
            # Bonus is proportional to consensus and number of models
            model_count_factor = min(1.0, len(valid_models) / 5.0)  # Max bonus at 5+ models
            consensus_bonus = consensus_ratio * model_count_factor * 0.1  # Max 10% bonus
            
            return consensus_bonus
            
        except Exception as e:
            logger.error(f"Consensus bonus calculation failed: {str(e)}")
            return 0.0

    def _calibrate_scores(self, result: ScoredResult) -> ScoredResult:
        """Apply score calibration to improve confidence reliability."""
        try:
            # Temperature scaling for score calibration
            temperature = self.calibration_params['temperature']
            calibrated_score = 1.0 / (1.0 + math.exp(-result.overall_score / temperature))
            
            # Bias correction
            bias_correction = self.calibration_params['bias_correction']
            calibrated_score = max(0.0, min(1.0, calibrated_score - bias_correction))
            
            # Confidence calibration with floor and ceiling
            confidence_floor = self.calibration_params['confidence_floor']
            confidence_ceiling = self.calibration_params['confidence_ceiling']
            
            calibrated_confidence = max(
                confidence_floor,
                min(confidence_ceiling, result.confidence)
            )
            
            # Adjust confidence based on score magnitude
            if calibrated_score < 0.3:
                calibrated_confidence *= 0.8  # Lower confidence for low scores
            elif calibrated_score > 0.8:
                calibrated_confidence = min(confidence_ceiling, calibrated_confidence * 1.1)
            
            return ScoredResult(
                overall_score=calibrated_score,
                confidence=calibrated_confidence,
                model_scores=result.model_scores,
                model_confidences=result.model_confidences
            )
            
        except Exception as e:
            logger.error(f"Score calibration failed: {str(e)}")
            return result

    def _generate_explanation(
        self,
        scores: Dict[str, float],
        confidences: Dict[str, float],
        model_weights: Dict[str, float],
        result: ScoredResult
    ) -> str:
        """Generate human-readable explanation for the ensemble score."""
        try:
            explanation_parts = []
            
            # Overall score interpretation
            quality = self._determine_match_quality(result.overall_score)
            explanation_parts.append(f"Overall Match Quality: {quality.title()} ({result.overall_score:.3f})")
            
            # Model contributions
            explanation_parts.append("\nModel Contributions:")
            
            # Sort models by their effective contribution
            model_contributions = []
            for model_name, score in scores.items():
                weight = model_weights.get(model_name, 0)
                confidence = confidences.get(model_name, 0)
                contribution = weight * score * (0.5 + 0.5 * confidence)
                model_contributions.append((model_name, score, confidence, contribution))
            
            # Sort by contribution (highest first)
            model_contributions.sort(key=lambda x: x[3], reverse=True)
            
            for model_name, score, confidence, contribution in model_contributions:
                model_desc = self._get_model_description(model_name)
                explanation_parts.append(
                    f"  • {model_desc}: {score:.3f} (confidence: {confidence:.3f})"
                )
            
            # Confidence explanation
            if result.confidence > 0.8:
                conf_desc = "very high"
            elif result.confidence > 0.6:
                conf_desc = "high"
            elif result.confidence > 0.4:
                conf_desc = "moderate"
            else:
                conf_desc = "low"
            
            explanation_parts.append(f"\nOverall Confidence: {conf_desc} ({result.confidence:.3f})")
            
            # Add interpretation guidance
            if result.overall_score > 0.8:
                explanation_parts.append("This is an excellent match with strong alignment across multiple factors.")
            elif result.overall_score > 0.6:
                explanation_parts.append("This is a good match with reasonable alignment in key areas.")
            elif result.overall_score > 0.4:
                explanation_parts.append("This is a fair match with some alignment but room for improvement.")
            else:
                explanation_parts.append("This match has limited alignment and may not be suitable.")
            
            return "\n".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {str(e)}")
            return f"Match score: {result.overall_score:.3f} (confidence: {result.confidence:.3f})"

    def _get_model_description(self, model_name: str) -> str:
        """Get human-readable description for a model."""
        descriptions = {
            'primary_semantic': 'Semantic Analysis',
            'fast_semantic': 'Quick Semantic Match',
            'domain_specific': 'Tech Domain Expert',
            'feature_based': 'Skills & Experience',
            'feedback_learned': 'User Preference Model'
        }
        
        return descriptions.get(model_name, model_name.replace('_', ' ').title())

    def _create_score_breakdown(
        self,
        scores: Dict[str, float],
        confidences: Dict[str, float],
        model_weights: Dict[str, float],
        valid_models: List[str]
    ) -> Dict[str, Any]:
        """Create detailed score breakdown for analysis."""
        try:
            breakdown = {
                'individual_models': {},
                'weighted_contributions': {},
                'aggregation_method': 'confidence_weighted_average',
                'normalization_applied': True,
                'calibration_applied': True
            }
            
            # Individual model details
            for model_name in valid_models:
                breakdown['individual_models'][model_name] = {
                    'raw_score': scores[model_name],
                    'confidence': confidences[model_name],
                    'model_weight': model_weights.get(model_name, 0),
                    'description': self._get_model_description(model_name)
                }
            
            # Calculate weighted contributions
            total_weight = sum(model_weights.get(name, 0) for name in valid_models)
            
            for model_name in valid_models:
                weight = model_weights.get(model_name, 0)
                confidence = confidences[model_name]
                effective_weight = (weight / total_weight) * (0.5 + 0.5 * confidence) if total_weight > 0 else 0
                
                breakdown['weighted_contributions'][model_name] = {
                    'normalized_weight': weight / total_weight if total_weight > 0 else 0,
                    'confidence_multiplier': 0.5 + 0.5 * confidence,
                    'effective_weight': effective_weight,
                    'score_contribution': effective_weight * scores[model_name]
                }
            
            return breakdown
            
        except Exception as e:
            logger.error(f"Score breakdown creation failed: {str(e)}")
            return {'error': 'Failed to create score breakdown'}

    def _determine_match_quality(self, score: float) -> str:
        """Determine qualitative match quality from numerical score."""
        if score >= self.score_thresholds['excellent']:
            return 'excellent'
        elif score >= self.score_thresholds['very_good']:
            return 'very_good'
        elif score >= self.score_thresholds['good']:
            return 'good'
        elif score >= self.score_thresholds['fair']:
            return 'fair'
        else:
            return 'poor'

    def _is_valid_score(self, score: Any) -> bool:
        """Validate that a score is a valid number between 0 and 1."""
        try:
            score_float = float(score)
            return 0.0 <= score_float <= 1.0 and not math.isnan(score_float)
        except (TypeError, ValueError):
            return False

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create an empty result for error cases."""
        return {
            'overall_score': 0.0,
            'confidence': 0.0,
            'model_scores': {},
            'model_confidences': {},
            'explanation': 'No valid model scores available',
            'score_breakdown': {},
            'matching_quality': 'poor',
            'ensemble_metadata': {
                'valid_models': [],
                'total_models': 0,
                'error': 'No valid model predictions'
            }
        }

    def calculate_ranking_scores(
        self,
        results: List[Dict[str, Any]],
        diversity_weight: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Calculate ranking scores with diversity consideration.
        
        Args:
            results: List of ensemble results to rank
            diversity_weight: Weight for diversity in ranking (0-1)
            
        Returns:
            Ranked list of results with ranking scores
        """
        try:
            if not results:
                return []
            
            # Calculate diversity scores
            diversity_scores = self._calculate_diversity_scores(results)
            
            # Combine relevance and diversity
            for i, result in enumerate(results):
                relevance_score = result.get('overall_score', 0.0)
                diversity_score = diversity_scores[i]
                
                # Weighted combination
                ranking_score = (
                    (1 - diversity_weight) * relevance_score +
                    diversity_weight * diversity_score
                )
                
                result['ranking_score'] = ranking_score
                result['diversity_score'] = diversity_score
                result['relevance_score'] = relevance_score
            
            # Sort by ranking score
            ranked_results = sorted(
                results,
                key=lambda x: x.get('ranking_score', 0.0),
                reverse=True
            )
            
            # Add rank positions
            for i, result in enumerate(ranked_results):
                result['rank'] = i + 1
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Ranking calculation failed: {str(e)}")
            return results

    def _calculate_diversity_scores(self, results: List[Dict[str, Any]]) -> List[float]:
        """Calculate diversity scores to promote result variety."""
        try:
            if len(results) <= 1:
                return [1.0] * len(results)
            
            diversity_scores = []
            
            for i, result in enumerate(results):
                # Extract job details for comparison
                job_details = result.get('job_details', {})
                
                # Calculate uniqueness compared to other results
                uniqueness_score = 0.0
                comparison_count = 0
                
                for j, other_result in enumerate(results):
                    if i != j:
                        other_job = other_result.get('job_details', {})
                        similarity = self._calculate_job_similarity(job_details, other_job)
                        uniqueness_score += (1.0 - similarity)
                        comparison_count += 1
                
                if comparison_count > 0:
                    avg_uniqueness = uniqueness_score / comparison_count
                else:
                    avg_uniqueness = 1.0
                
                diversity_scores.append(avg_uniqueness)
            
            return diversity_scores
            
        except Exception as e:
            logger.error(f"Diversity score calculation failed: {str(e)}")
            return [1.0] * len(results)

    def _calculate_job_similarity(self, job1: Dict[str, Any], job2: Dict[str, Any]) -> float:
        """Calculate similarity between two jobs."""
        try:
            similarity_factors = []
            
            # Company similarity
            company1 = job1.get('company_name', '').lower()
            company2 = job2.get('company_name', '').lower()
            company_sim = 1.0 if company1 == company2 else 0.0
            similarity_factors.append(company_sim)
            
            # Job type similarity
            type1 = job1.get('job_type', '').lower()
            type2 = job2.get('job_type', '').lower()
            type_sim = 1.0 if type1 == type2 else 0.0
            similarity_factors.append(type_sim)
            
            # Location similarity
            loc1 = job1.get('location', '').lower()
            loc2 = job2.get('location', '').lower()
            loc_sim = 1.0 if loc1 == loc2 else 0.0
            similarity_factors.append(loc_sim)
            
            # Skills similarity (Jaccard index)
            skills1 = set(s.lower() for s in job1.get('skills', []))
            skills2 = set(s.lower() for s in job2.get('skills', []))
            
            if skills1 or skills2:
                intersection = len(skills1.intersection(skills2))
                union = len(skills1.union(skills2))
                skills_sim = intersection / union if union > 0 else 0.0
            else:
                skills_sim = 0.0
            
            similarity_factors.append(skills_sim)
            
            # Calculate weighted average
            weights = [0.3, 0.2, 0.2, 0.3]  # company, type, location, skills
            weighted_similarity = sum(w * s for w, s in zip(weights, similarity_factors))
            
            return weighted_similarity
            
        except Exception as e:
            logger.error(f"Job similarity calculation failed: {str(e)}")
            return 0.0

    async def get_scorer_stats(self) -> Dict[str, Any]:
        """Get statistics about the ensemble scorer."""
        return {
            'default_weights': self.default_weights.copy(),
            'score_thresholds': self.score_thresholds.copy(),
            'calibration_params': self.calibration_params.copy(),
            'supported_models': list(self.default_weights.keys())
        }