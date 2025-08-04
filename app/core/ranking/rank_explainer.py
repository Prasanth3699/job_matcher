"""
Rank Explainer for interpretable ranking decisions.

This module provides comprehensive explanations for ranking decisions,
helping users understand why certain jobs were ranked higher than others.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import json

from app.utils.logger import logger
from app.core.constants.ml_constants import (
    RankingConfig,
    ScoringThresholds,
    PerformanceMetrics
)


@dataclass
class FeatureContribution:
    """Individual feature contribution to ranking score."""
    feature_name: str
    value: float
    contribution: float
    importance: float
    interpretation: str


@dataclass
class RankingExplanation:
    """Comprehensive explanation for a ranking decision."""
    job_id: str
    rank_position: int
    overall_score: float
    confidence: float
    feature_contributions: List[FeatureContribution]
    model_explanations: Dict[str, Any]
    comparative_analysis: Optional[Dict[str, Any]]
    recommendations: List[str]
    explanation_confidence: float


@dataclass
class ComparisonExplanation:
    """Explanation comparing two ranked items."""
    higher_ranked_job: str
    lower_ranked_job: str
    score_difference: float
    key_differentiators: List[Dict[str, Any]]
    similarity_score: float
    explanation_text: str


class RankExplainer:
    """
    Advanced ranking explanation system that provides interpretable insights
    into ranking decisions using feature attribution and model interpretation.
    """
    
    def __init__(self):
        """Initialize the rank explainer."""
        self.feature_names = self._initialize_feature_names()
        self.feature_categories = self._initialize_feature_categories()
        self.scoring_thresholds = {
            'excellent': ScoringThresholds.EXCELLENT_MATCH,
            'very_good': ScoringThresholds.VERY_GOOD_MATCH,
            'good': ScoringThresholds.GOOD_MATCH,
            'fair': ScoringThresholds.FAIR_MATCH,
            'poor': ScoringThresholds.POOR_MATCH
        }
        
        # Explanation templates
        self.explanation_templates = self._initialize_explanation_templates()
        
        logger.info("RankExplainer initialized successfully")
    
    def _initialize_feature_names(self) -> List[str]:
        """Initialize comprehensive feature names for explanation."""
        return [
            # Semantic features (4)
            'semantic_cosine_similarity', 'semantic_l2_distance', 
            'semantic_max_product', 'semantic_mean_product',
            
            # Statistical features (10)
            'skill_jaccard', 'skill_coverage', 'skill_overlap_ratio',
            'experience_match', 'education_match', 'location_match',
            'resume_skill_count', 'job_skill_count', 
            'resume_experience_years', 'required_experience_years',
            
            # Interaction features (6)
            'technical_skill_match', 'soft_skill_match', 
            'domain_skill_match', 'tool_skill_match',
            'level_alignment', 'complexity_match',
            
            # Contextual features (5)
            'posting_freshness', 'market_demand', 
            'company_score', 'salary_competitiveness', 'normalized_posting_age'
        ]
    
    def _initialize_feature_categories(self) -> Dict[str, List[str]]:
        """Categorize features for better explanation organization."""
        return {
            'skills_and_experience': [
                'skill_jaccard', 'skill_coverage', 'skill_overlap_ratio',
                'technical_skill_match', 'soft_skill_match', 'domain_skill_match', 'tool_skill_match',
                'experience_match', 'resume_experience_years', 'required_experience_years'
            ],
            'education_and_qualifications': [
                'education_match', 'level_alignment', 'complexity_match'
            ],
            'semantic_matching': [
                'semantic_cosine_similarity', 'semantic_l2_distance',
                'semantic_max_product', 'semantic_mean_product'
            ],
            'job_context': [
                'posting_freshness', 'market_demand', 'company_score',
                'salary_competitiveness', 'normalized_posting_age'
            ],
            'location_and_logistics': [
                'location_match'
            ],
            'quantitative_metrics': [
                'resume_skill_count', 'job_skill_count'
            ]
        }
    
    def _initialize_explanation_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize explanation templates for different scenarios."""
        return {
            'high_skill_match': {
                'condition': 'skill_jaccard > 0.7',
                'template': "Strong skill alignment with {percentage}% of required skills matching your background.",
                'recommendation': "This role leverages your existing expertise well."
            },
            'experience_overqualified': {
                'condition': 'experience_match > 1.2',
                'template': "You have {years} more years of experience than typically required.",
                'recommendation': "Consider how this role fits your career progression goals."
            },
            'experience_underqualified': {
                'condition': 'experience_match < 0.8',
                'template': "This role requires {gap} more years of experience than you currently have.",
                'recommendation': "Consider gaining additional experience or highlighting transferable skills."
            },
            'semantic_strong_match': {
                'condition': 'semantic_cosine_similarity > 0.8',
                'template': "Excellent semantic match between your background and job requirements.",
                'recommendation': "Your experience description aligns very well with this role."
            },
            'education_mismatch': {
                'condition': 'education_match < 0.5',
                'template': "Educational requirements may not fully align with your background.",
                'recommendation': "Consider highlighting relevant certifications or equivalent experience."
            },
            'fresh_posting': {
                'condition': 'posting_freshness > 0.8',
                'template': "This is a recently posted position with high visibility.",
                'recommendation': "Apply soon to increase your chances of being noticed."
            },
            'high_market_demand': {
                'condition': 'market_demand > 0.7',
                'template': "This role is in high demand in the current job market.",
                'recommendation': "Strong market demand may mean competitive salaries and growth opportunities."
            }
        }
    
    async def explain_ranking(
        self,
        job_data: Dict[str, Any],
        resume_data: Dict[str, Any],
        ranking_features: np.ndarray,
        model_outputs: Dict[str, Any],
        rank_position: int,
        comparative_jobs: Optional[List[Dict[str, Any]]] = None
    ) -> RankingExplanation:
        """
        Generate comprehensive explanation for a ranking decision.
        
        Args:
            job_data: Job posting data
            resume_data: Resume data
            ranking_features: Feature vector used for ranking
            model_outputs: Model prediction outputs
            rank_position: Position in ranking list
            comparative_jobs: Other jobs for comparison
            
        Returns:
            Comprehensive ranking explanation
        """
        try:
            job_id = job_data.get('job_id', job_data.get('id', 'unknown'))
            overall_score = model_outputs.get('relevance', [0.0])[0]
            confidence = model_outputs.get('confidence', [0.0])[0]
            
            # Calculate feature contributions
            feature_contributions = await self._calculate_feature_contributions(
                ranking_features, model_outputs, job_data, resume_data
            )
            
            # Generate model-specific explanations
            model_explanations = self._generate_model_explanations(
                model_outputs, feature_contributions
            )
            
            # Comparative analysis if other jobs provided
            comparative_analysis = None
            if comparative_jobs:
                comparative_analysis = await self._generate_comparative_analysis(
                    job_data, comparative_jobs, ranking_features, rank_position
                )
            
            # Generate actionable recommendations
            recommendations = self._generate_recommendations(
                feature_contributions, job_data, resume_data
            )
            
            # Calculate explanation confidence
            explanation_confidence = self._calculate_explanation_confidence(
                feature_contributions, confidence, len(comparative_jobs) if comparative_jobs else 0
            )
            
            explanation = RankingExplanation(
                job_id=job_id,
                rank_position=rank_position,
                overall_score=float(overall_score),
                confidence=float(confidence),
                feature_contributions=feature_contributions,
                model_explanations=model_explanations,
                comparative_analysis=comparative_analysis,
                recommendations=recommendations,
                explanation_confidence=explanation_confidence
            )
            
            logger.debug(f"Generated ranking explanation for job {job_id} at position {rank_position}")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Ranking explanation generation failed: {str(e)}")
            return self._create_fallback_explanation(job_data, rank_position)
    
    async def _calculate_feature_contributions(
        self,
        features: np.ndarray,
        model_outputs: Dict[str, Any],
        job_data: Dict[str, Any],
        resume_data: Dict[str, Any]
    ) -> List[FeatureContribution]:
        """Calculate individual feature contributions to the ranking score."""
        try:
            contributions = []
            
            # Simple feature importance calculation (would use SHAP/LIME in production)
            feature_importances = self._estimate_feature_importance(features, model_outputs)
            
            for i, (feature_name, importance) in enumerate(zip(self.feature_names, feature_importances)):
                if i < len(features):
                    feature_value = float(features[i])
                    contribution = feature_value * importance
                    
                    # Generate interpretation
                    interpretation = self._interpret_feature(
                        feature_name, feature_value, job_data, resume_data
                    )
                    
                    contributions.append(FeatureContribution(
                        feature_name=feature_name,
                        value=feature_value,
                        contribution=contribution,
                        importance=importance,
                        interpretation=interpretation
                    ))
            
            # Sort by absolute contribution
            contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
            
            return contributions
            
        except Exception as e:
            logger.error(f"Feature contribution calculation failed: {str(e)}")
            return []
    
    def _estimate_feature_importance(
        self,
        features: np.ndarray,
        model_outputs: Dict[str, Any]
    ) -> List[float]:
        """Estimate feature importance (simplified approach)."""
        try:
            # Simplified importance calculation
            # In production, this would use model-specific attribution methods
            
            num_features = len(features)
            base_importance = 1.0 / num_features
            
            # Assign higher importance to different feature categories
            importances = []
            
            for i, feature_name in enumerate(self.feature_names[:num_features]):
                if 'skill' in feature_name:
                    importance = base_importance * 1.5  # Skills are important
                elif 'semantic' in feature_name:
                    importance = base_importance * 1.3  # Semantic matching is important
                elif 'experience' in feature_name:
                    importance = base_importance * 1.4  # Experience is crucial
                elif 'education' in feature_name:
                    importance = base_importance * 1.2  # Education matters
                elif 'location' in feature_name:
                    importance = base_importance * 0.8  # Location less critical for remote
                else:
                    importance = base_importance
                
                # Adjust based on feature value magnitude
                if i < len(features):
                    feature_value = abs(features[i])
                    importance *= (1.0 + feature_value * 0.2)  # Higher values get more importance
                
                importances.append(importance)
            
            # Normalize importances
            total_importance = sum(importances)
            if total_importance > 0:
                importances = [imp / total_importance for imp in importances]
            
            return importances
            
        except Exception as e:
            logger.error(f"Feature importance estimation failed: {str(e)}")
            return [1.0 / len(features)] * len(features)
    
    def _interpret_feature(
        self,
        feature_name: str,
        feature_value: float,
        job_data: Dict[str, Any],
        resume_data: Dict[str, Any]
    ) -> str:
        """Generate human-readable interpretation for a feature."""
        try:
            # Feature-specific interpretations
            if feature_name == 'skill_jaccard':
                percentage = int(feature_value * 100)
                return f"{percentage}% skill overlap between your background and job requirements"
            
            elif feature_name == 'skill_coverage':
                percentage = int(feature_value * 100)
                return f"You possess {percentage}% of the skills required for this role"
            
            elif feature_name == 'experience_match':
                if feature_value > 1.0:
                    excess = (feature_value - 1.0) * 100
                    return f"You have {excess:.0f}% more experience than typically required"
                elif feature_value < 1.0:
                    deficit = (1.0 - feature_value) * 100
                    return f"You have {deficit:.0f}% less experience than typically required"
                else:
                    return "Your experience level matches the job requirements perfectly"
            
            elif feature_name == 'education_match':
                if feature_value >= 1.0:
                    return "Your education meets or exceeds the requirements"
                elif feature_value >= 0.7:
                    return "Your education partially meets the requirements"
                else:
                    return "Your education may not fully align with requirements"
            
            elif feature_name == 'semantic_cosine_similarity':
                if feature_value > 0.8:
                    return "Excellent semantic match between your profile and job description"
                elif feature_value > 0.6:
                    return "Good semantic alignment with job requirements"
                elif feature_value > 0.4:
                    return "Moderate semantic match with the role"
                else:
                    return "Limited semantic alignment with job description"
            
            elif feature_name == 'location_match':
                if feature_value > 0.8:
                    return "Excellent location compatibility"
                elif feature_value > 0.5:
                    return "Good location match or remote-friendly role"
                else:
                    return "Location may require consideration"
            
            elif feature_name == 'posting_freshness':
                if feature_value > 0.8:
                    return "Very recently posted job with high visibility"
                elif feature_value > 0.5:
                    return "Recently posted position"
                else:
                    return "Older job posting"
            
            elif feature_name == 'market_demand':
                if feature_value > 0.7:
                    return "High market demand for this role type"
                elif feature_value > 0.4:
                    return "Moderate market demand"
                else:
                    return "Lower market demand for this role"
            
            elif feature_name == 'company_score':
                if feature_value > 0.8:
                    return "Highly attractive company with good benefits"
                elif feature_value > 0.6:
                    return "Good company with competitive offerings"
                else:
                    return "Company attractiveness score is moderate"
            
            elif 'technical_skill' in feature_name:
                percentage = int(feature_value * 100)
                return f"{percentage}% match on technical skills"
            
            elif 'soft_skill' in feature_name:
                percentage = int(feature_value * 100)
                return f"{percentage}% match on soft skills and communication"
            
            elif feature_name == 'level_alignment':
                if feature_value > 0.8:
                    return "Excellent career level alignment with the role"
                elif feature_value > 0.6:
                    return "Good career level match"
                else:
                    return "Career level may not fully align"
            
            elif feature_name == 'complexity_match':
                if feature_value > 0.8:
                    return "Job complexity matches your experience level well"
                elif feature_value > 0.6:
                    return "Reasonable complexity match"
                else:
                    return "Job complexity may not align with your background"
            
            else:
                # Generic interpretation for unspecified features
                if feature_value > 0.7:
                    return f"Strong {feature_name.replace('_', ' ')} score"
                elif feature_value > 0.5:
                    return f"Good {feature_name.replace('_', ' ')} alignment"
                elif feature_value > 0.3:
                    return f"Moderate {feature_name.replace('_', ' ')} match"
                else:
                    return f"Limited {feature_name.replace('_', ' ')} alignment"
            
        except Exception as e:
            logger.error(f"Feature interpretation failed for {feature_name}: {str(e)}")
            return f"Feature value: {feature_value:.3f}"
    
    def _generate_model_explanations(
        self,
        model_outputs: Dict[str, Any],
        feature_contributions: List[FeatureContribution]
    ) -> Dict[str, Any]:
        """Generate model-specific explanations."""
        try:
            explanations = {}
            
            # Relevance score explanation
            relevance_score = model_outputs.get('relevance', [0.0])[0]
            if relevance_score > 0.8:
                relevance_explanation = "High relevance - this job strongly matches your profile"
            elif relevance_score > 0.6:
                relevance_explanation = "Good relevance - this job is a solid match for your background"
            elif relevance_score > 0.4:
                relevance_explanation = "Moderate relevance - partial alignment with your profile"
            else:
                relevance_explanation = "Lower relevance - limited match with your background"
            
            explanations['relevance'] = {
                'score': float(relevance_score),
                'explanation': relevance_explanation
            }
            
            # Diversity score explanation
            diversity_score = model_outputs.get('diversity', [0.0])[0]
            if diversity_score > 0.7:
                diversity_explanation = "Adds good variety to your job recommendations"
            elif diversity_score > 0.4:
                diversity_explanation = "Provides moderate diversity in opportunities"
            else:
                diversity_explanation = "Similar to other recommendations"
            
            explanations['diversity'] = {
                'score': float(diversity_score),
                'explanation': diversity_explanation
            }
            
            # Confidence explanation
            confidence_score = model_outputs.get('confidence', [0.0])[0]
            if confidence_score > 0.8:
                confidence_explanation = "High confidence in this ranking prediction"
            elif confidence_score > 0.6:
                confidence_explanation = "Good confidence in ranking accuracy"
            else:
                confidence_explanation = "Moderate confidence - consider additional factors"
            
            explanations['confidence'] = {
                'score': float(confidence_score),
                'explanation': confidence_explanation
            }
            
            # Top contributing factors
            top_factors = feature_contributions[:3]
            explanations['top_factors'] = [
                {
                    'factor': factor.feature_name.replace('_', ' ').title(),
                    'contribution': factor.contribution,
                    'interpretation': factor.interpretation
                }
                for factor in top_factors
            ]
            
            return explanations
            
        except Exception as e:
            logger.error(f"Model explanation generation failed: {str(e)}")
            return {}
    
    async def _generate_comparative_analysis(
        self,
        current_job: Dict[str, Any],
        other_jobs: List[Dict[str, Any]],
        current_features: np.ndarray,
        current_rank: int
    ) -> Dict[str, Any]:
        """Generate comparative analysis with other ranked jobs."""
        try:
            analysis = {
                'rank_position': current_rank,
                'total_jobs': len(other_jobs) + 1,
                'percentile': (len(other_jobs) - current_rank + 1) / (len(other_jobs) + 1),
                'comparisons': []
            }
            
            # Compare with jobs ranked immediately above and below
            comparison_indices = []
            
            # Job ranked above (if exists)
            if current_rank > 1:
                comparison_indices.append(current_rank - 2)  # 0-indexed
            
            # Job ranked below (if exists)
            if current_rank <= len(other_jobs):
                comparison_indices.append(current_rank)  # 0-indexed, next job
            
            for idx in comparison_indices:
                if 0 <= idx < len(other_jobs):
                    comparison = await self._compare_jobs(
                        current_job, other_jobs[idx], current_features
                    )
                    if comparison:
                        analysis['comparisons'].append(comparison)
            
            # Overall positioning
            if analysis['percentile'] > 0.8:
                analysis['positioning'] = "Top tier recommendation - excellent match"
            elif analysis['percentile'] > 0.6:
                analysis['positioning'] = "Upper tier recommendation - very good match"
            elif analysis['percentile'] > 0.4:
                analysis['positioning'] = "Middle tier recommendation - good potential"
            else:
                analysis['positioning'] = "Lower tier recommendation - consider carefully"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Comparative analysis generation failed: {str(e)}")
            return {}
    
    async def _compare_jobs(
        self,
        job1: Dict[str, Any],
        job2: Dict[str, Any],
        job1_features: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Compare two jobs and identify key differentiators."""
        try:
            job1_id = job1.get('job_id', job1.get('id', 'current'))
            job2_id = job2.get('job_id', job2.get('id', 'other'))
            
            # Key comparison factors
            comparisons = []
            
            # Skills comparison
            job1_skills = set(s.lower() for s in job1.get('skills', []))
            job2_skills = set(s.lower() for s in job2.get('skills', []))
            
            skill_overlap = len(job1_skills.intersection(job2_skills))
            skill_difference = len(job1_skills.symmetric_difference(job2_skills))
            
            if skill_difference > 0:
                comparisons.append({
                    'factor': 'Skills Requirements',
                    'description': f"Different skill focus - {skill_difference} unique skills between roles",
                    'impact': 'medium' if skill_difference > 3 else 'low'
                })
            
            # Company comparison
            job1_company = job1.get('company_name', '').lower()
            job2_company = job2.get('company_name', '').lower()
            
            if job1_company != job2_company:
                comparisons.append({
                    'factor': 'Company',
                    'description': f"Different companies: {job1.get('company_name', 'N/A')} vs {job2.get('company_name', 'N/A')}",
                    'impact': 'medium'
                })
            
            # Location comparison
            job1_location = job1.get('location', '').lower()
            job2_location = job2.get('location', '').lower()
            
            if job1_location != job2_location:
                comparisons.append({
                    'factor': 'Location',
                    'description': f"Different locations: {job1.get('location', 'N/A')} vs {job2.get('location', 'N/A')}",
                    'impact': 'low' if 'remote' in f"{job1_location} {job2_location}" else 'medium'
                })
            
            # Job level comparison
            job1_title = job1.get('job_title', '').lower()
            job2_title = job2.get('job_title', '').lower()
            
            level_keywords = ['senior', 'junior', 'lead', 'principal', 'manager', 'director']
            job1_level = any(keyword in job1_title for keyword in level_keywords)
            job2_level = any(keyword in job2_title for keyword in level_keywords)
            
            if job1_level != job2_level:
                comparisons.append({
                    'factor': 'Career Level',
                    'description': "Different seniority levels between the roles",
                    'impact': 'high'
                })
            
            return {
                'job1_id': job1_id,
                'job2_id': job2_id,
                'job1_title': job1.get('job_title', 'N/A'),
                'job2_title': job2.get('job_title', 'N/A'),
                'key_differences': comparisons,
                'similarity_score': skill_overlap / max(len(job1_skills.union(job2_skills)), 1)
            }
            
        except Exception as e:
            logger.error(f"Job comparison failed: {str(e)}")
            return None
    
    def _generate_recommendations(
        self,
        feature_contributions: List[FeatureContribution],
        job_data: Dict[str, Any],
        resume_data: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on feature analysis."""
        try:
            recommendations = []
            
            # Analyze top contributing features for recommendations
            for contribution in feature_contributions[:5]:
                feature_name = contribution.feature_name
                feature_value = contribution.value
                
                # Skill-related recommendations
                if 'skill' in feature_name and feature_value < 0.7:
                    if feature_value < 0.5:
                        recommendations.append(
                            "Consider highlighting transferable skills or gaining experience in key areas mentioned in the job requirements"
                        )
                    else:
                        recommendations.append(
                            "Emphasize your relevant skills and consider obtaining certifications in missing areas"
                        )
                
                # Experience-related recommendations
                if 'experience' in feature_name:
                    if feature_value < 0.8:
                        recommendations.append(
                            "Focus on demonstrating how your experience translates to this role's requirements"
                        )
                    elif feature_value > 1.2:
                        recommendations.append(
                            "Consider how this role aligns with your career progression and growth goals"
                        )
                
                # Education recommendations
                if 'education' in feature_name and feature_value < 0.6:
                    recommendations.append(
                        "Highlight relevant certifications, courses, or equivalent experience that compensates for educational requirements"
                    )
                
                # Semantic match recommendations
                if 'semantic' in feature_name and feature_value < 0.6:
                    recommendations.append(
                        "Tailor your application to better align with the job description language and requirements"
                    )
            
            # General recommendations based on overall patterns
            avg_skill_score = np.mean([
                c.value for c in feature_contributions 
                if 'skill' in c.feature_name
            ])
            
            if avg_skill_score > 0.8:
                recommendations.append(
                    "Excellent skill match - emphasize your expertise in the application"
                )
            elif avg_skill_score < 0.4:
                recommendations.append(
                    "Consider gaining additional skills or experience before applying, or focus on transferable competencies"
                )
            
            # Remove duplicates and limit to top recommendations
            unique_recommendations = list(dict.fromkeys(recommendations))
            return unique_recommendations[:4]  # Top 4 recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            return ["Review the job requirements carefully and tailor your application accordingly"]
    
    def _calculate_explanation_confidence(
        self,
        feature_contributions: List[FeatureContribution],
        model_confidence: float,
        num_comparisons: int
    ) -> float:
        """Calculate confidence in the explanation quality."""
        try:
            # Base confidence from model
            explanation_confidence = model_confidence * 0.6
            
            # Feature contribution clarity
            if feature_contributions:
                top_contributions = feature_contributions[:3]
                contribution_variance = np.var([abs(c.contribution) for c in top_contributions])
                
                # Higher variance in top contributions = clearer explanation
                clarity_bonus = min(0.2, contribution_variance * 0.5)
                explanation_confidence += clarity_bonus
            
            # Number of comparisons available
            comparison_bonus = min(0.2, num_comparisons * 0.05)
            explanation_confidence += comparison_bonus
            
            return min(1.0, explanation_confidence)
            
        except Exception as e:
            logger.error(f"Explanation confidence calculation failed: {str(e)}")
            return 0.5
    
    def _create_fallback_explanation(
        self,
        job_data: Dict[str, Any],
        rank_position: int
    ) -> RankingExplanation:
        """Create a basic explanation when detailed analysis fails."""
        job_id = job_data.get('job_id', job_data.get('id', 'unknown'))
        
        return RankingExplanation(
            job_id=job_id,
            rank_position=rank_position,
            overall_score=0.5,
            confidence=0.3,
            feature_contributions=[],
            model_explanations={
                'status': 'fallback_explanation',
                'message': 'Detailed analysis unavailable - basic ranking applied'
            },
            comparative_analysis=None,
            recommendations=[
                "Review job requirements carefully",
                "Consider how your background aligns with this role"
            ],
            explanation_confidence=0.3
        )
    
    async def explain_ranking_difference(
        self,
        higher_job: Dict[str, Any],
        lower_job: Dict[str, Any],
        score_difference: float
    ) -> ComparisonExplanation:
        """
        Explain why one job is ranked higher than another.
        
        Args:
            higher_job: Job ranked higher
            lower_job: Job ranked lower
            score_difference: Difference in ranking scores
            
        Returns:
            Detailed comparison explanation
        """
        try:
            higher_id = higher_job.get('job_id', higher_job.get('id', 'unknown'))
            lower_id = lower_job.get('job_id', lower_job.get('id', 'unknown'))
            
            # Identify key differentiators
            differentiators = []
            
            # Skills comparison
            higher_skills = set(s.lower() for s in higher_job.get('skills', []))
            lower_skills = set(s.lower() for s in lower_job.get('skills', []))
            
            skill_overlap_higher = len(higher_skills)  # Simplified
            skill_overlap_lower = len(lower_skills)
            
            if skill_overlap_higher != skill_overlap_lower:
                differentiators.append({
                    'factor': 'Skills Match',
                    'higher_value': skill_overlap_higher,
                    'lower_value': skill_overlap_lower,
                    'impact': abs(skill_overlap_higher - skill_overlap_lower) * 0.1
                })
            
            # Company attractiveness
            higher_company = higher_job.get('company_name', '')
            lower_company = lower_job.get('company_name', '')
            
            if higher_company != lower_company:
                differentiators.append({
                    'factor': 'Company Profile',
                    'higher_value': higher_company,
                    'lower_value': lower_company,
                    'impact': 0.1  # Simplified impact
                })
            
            # Calculate similarity
            all_skills = higher_skills.union(lower_skills)
            common_skills = higher_skills.intersection(lower_skills)
            similarity_score = len(common_skills) / len(all_skills) if all_skills else 0.0
            
            # Generate explanation text
            if score_difference > 0.2:
                explanation_text = f"Significantly higher match due to better alignment in key areas"
            elif score_difference > 0.1:
                explanation_text = f"Moderately higher match with some advantages"
            else:
                explanation_text = f"Slightly higher match with minor advantages"
            
            return ComparisonExplanation(
                higher_ranked_job=higher_id,
                lower_ranked_job=lower_id,
                score_difference=score_difference,
                key_differentiators=differentiators,
                similarity_score=similarity_score,
                explanation_text=explanation_text
            )
            
        except Exception as e:
            logger.error(f"Ranking difference explanation failed: {str(e)}")
            return ComparisonExplanation(
                higher_ranked_job="unknown",
                lower_ranked_job="unknown",
                score_difference=0.0,
                key_differentiators=[],
                similarity_score=0.0,
                explanation_text="Comparison analysis unavailable"
            )
    
    async def generate_summary_explanation(
        self,
        explanations: List[RankingExplanation]
    ) -> Dict[str, Any]:
        """Generate a summary explanation for multiple ranking decisions."""
        try:
            if not explanations:
                return {'summary': 'No explanations available'}
            
            # Overall patterns
            avg_confidence = np.mean([exp.confidence for exp in explanations])
            avg_score = np.mean([exp.overall_score for exp in explanations])
            
            # Most common factors
            all_factors = []
            for exp in explanations:
                for contribution in exp.feature_contributions[:3]:
                    all_factors.append(contribution.feature_name)
            
            from collections import Counter
            common_factors = Counter(all_factors).most_common(3)
            
            # Quality assessment
            if avg_score > 0.7:
                quality_assessment = "Strong overall matches found"
            elif avg_score > 0.5:
                quality_assessment = "Good potential matches identified"
            else:
                quality_assessment = "Limited matches - consider expanding criteria"
            
            summary = {
                'total_jobs_explained': len(explanations),
                'average_confidence': avg_confidence,
                'average_score': avg_score,
                'quality_assessment': quality_assessment,
                'key_factors': [factor for factor, count in common_factors],
                'recommendations': [
                    "Focus on positions with higher confidence scores",
                    "Review common success factors across recommendations",
                    "Consider additional skill development based on gaps identified"
                ]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary explanation generation failed: {str(e)}")
            return {'summary': 'Summary generation failed'}
    
    def get_explainer_info(self) -> Dict[str, Any]:
        """Get information about the explainer capabilities."""
        return {
            'supported_features': len(self.feature_names),
            'feature_categories': list(self.feature_categories.keys()),
            'explanation_types': [
                'individual_ranking',
                'comparative_analysis',
                'feature_contributions',
                'actionable_recommendations'
            ],
            'confidence_factors': [
                'model_confidence',
                'feature_clarity',
                'comparison_availability'
            ]
        }