from typing import List, Dict, Any
from dataclasses import asdict

import numpy as np
from ...utils.logger import logger
from .models import MatchResult, MatchingConstants
from .semantic_matcher import SemanticMatcher
from .feature_matcher import FeatureMatcher


def coerce_list_of_str(seq):
    if seq is None:
        return []
    if isinstance(seq, str):
        return [seq.strip()]
    return [str(x).strip() for x in seq if x is not None and str(x).strip() != ""]


class HybridScorer:
    """
    Combines semantic and feature-based matching to produce final scores.
    Handles result ranking and explanation generation.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.semantic_matcher = SemanticMatcher(model_name)
        self.feature_matcher = FeatureMatcher()
        self.constants = MatchingConstants()

    def match_resume_to_jobs(
        self,
        resume_data: Dict[str, Any],
        jobs_data: List[Dict[str, Any]],
        top_n: int = 5,
    ) -> List[MatchResult]:
        """
        Enhanced matching using all SemanticMatcher capabilities while
        preserving the original return structure.
        """
        if not jobs_data:
            return []

        results = []

        for job in jobs_data:
            try:
                resume_skills = coerce_list_of_str(resume_data.get("skills", []))
                job_skills = coerce_list_of_str(job.get("skills", []))
                # Get job description for context-aware matching
                job_context = job.get("description", "")[
                    :1000
                ]  # Use first 1000 chars as context

                # Enhanced skill matching with context
                skill_score, matching_skills, missing_skills = (
                    self.semantic_matcher.enhanced_skill_match(
                        resume_skills,
                        job_skills,
                        context=job_context,
                    )
                )

                # Prepare job features with enhanced skill score
                job_features = job["normalized_features"].copy()
                job_features["skill_match_score"] = skill_score

                # Feature-based matching (unchanged)
                feature_scores = self.feature_matcher.calculate_feature_similarity(
                    resume_data["normalized_features"], job_features
                )

                # Calculate weighted score (using enhanced version)
                overall_score = self._calculate_weighted_score(feature_scores)

                # Generate explanation - same format as before
                explanation = self._generate_explanation(
                    overall_score,
                    feature_scores,
                    matching_skills[:5],  # Keep top 5 for display
                    missing_skills[:5],  # Keep top 5 for display
                    job,
                )

                # Create result with original structure
                result = MatchResult(
                    job_id=job["job_id"],
                    job_title=job["job_title"],
                    company_name=job["company_name"],
                    location=job["location"],
                    job_type=job["job_type"],
                    apply_link=job["apply_link"],
                    overall_score=overall_score,
                    score_breakdown=feature_scores,
                    missing_skills=missing_skills[:5],  # Preserve original limit
                    matching_skills=matching_skills[:5],  # Preserve original limit
                    experience_match=feature_scores["experience"],
                    salary_match=feature_scores["salary"],
                    location_match=feature_scores["location"],
                    job_type_match=feature_scores["job_type"],
                    explanation=explanation,
                )

                results.append(result)

            except Exception as e:
                logger.warning(f"Failed to process job {job.get('job_id')}: {str(e)}")
                continue

        # Sort and return top_n results (unchanged)
        results.sort(key=lambda x: x.overall_score, reverse=True)
        return results[:top_n]

    def _calculate_match(
        self, resume_data: Dict[str, Any], job_data: Dict[str, Any]
    ) -> MatchResult:
        """Calculate match between resume and a single job"""
        # Semantic skill matching
        skill_score, matching_skills, missing_skills = (
            self.semantic_matcher.match_skills(
                resume_data.get("skills", []), job_data.get("skills", [])
            )
        )

        # Add skill match score to job features for feature matching
        job_features = job_data["normalized_features"].copy()
        job_features["skill_match_score"] = skill_score

        # Feature-based matching
        feature_scores = self.feature_matcher.calculate_feature_similarity(
            resume_data["normalized_features"], job_features
        )

        # Calculate weighted overall score
        overall_score = self._calculate_weighted_score(feature_scores)

        # Generate explanation
        explanation = self._generate_explanation(
            overall_score, feature_scores, matching_skills, missing_skills, job_data
        )

        return MatchResult(
            job_id=job_data["job_id"],
            job_title=job_data["job_title"],
            company_name=job_data["company_name"],
            location=job_data["location"],
            job_type=job_data["job_type"],
            apply_link=job_data["apply_link"],
            overall_score=overall_score,
            score_breakdown=feature_scores,
            missing_skills=missing_skills,
            matching_skills=matching_skills,
            experience_match=feature_scores["experience"],
            salary_match=feature_scores["salary"],
            location_match=feature_scores["location"],
            job_type_match=feature_scores["job_type"],
            explanation=explanation,
        )

    def _calculate_weighted_score(self, feature_scores: Dict[str, float]) -> float:
        """
        Enhanced weighted score calculation with:
        - Dynamic weight adjustment based on feature importance
        - Score normalization
        - Minimum viable feature requirements
        """

        # Get base weights from constants
        base_weights = self.constants.FEATURE_WEIGHTS

        # Calculate effective weights considering feature availability
        effective_weights = {}
        for feature, weight in base_weights.items():
            if feature in feature_scores and feature_scores[feature] is not None:
                effective_weights[feature] = weight

        # Special case: If skills are matched, increase its weight
        if "skills" in effective_weights and feature_scores.get("skills", 0) > 0.7:
            effective_weights["skills"] = min(
                effective_weights["skills"] * 1.2, 0.5
            )  # Cap at 50% weight

        # Calculate weighted sum with validation
        weighted_sum = 0.0
        total_weight = 0.0

        for feature, weight in effective_weights.items():
            score = feature_scores[feature]

            # Validate score range
            if not 0 <= score <= 1:
                logger.warning(
                    f"Invalid score for {feature}: {score}. Clamping to [0,1]"
                )
                score = max(0, min(1, score))

            weighted_sum += score * weight
            total_weight += weight

        # Normalize and handle edge cases
        if total_weight == 0:
            return 0.0

        raw_score = weighted_sum / total_weight

        # Apply non-linear scaling to emphasize higher scores
        scaled_score = raw_score**0.9  # Makes scores >0.7 more distinct

        # Ensure we return a native Python float
        return float(np.clip(scaled_score, 0, 1))

    def _generate_explanation(
        self,
        overall_score: float,
        feature_scores: Dict[str, float],
        matching_skills: List[str],
        missing_skills: List[str],
        job_data: Dict[str, Any],
    ) -> str:
        """Generate more structured explanations"""
        explanations = []

        # Overall match quality
        if overall_score >= 0.8:
            quality = "Excellent match"
        elif overall_score >= 0.6:
            quality = "Good match"
        elif overall_score >= 0.4:
            quality = "Moderate match"
        else:
            quality = "Weak match"
        explanations.append(
            f"{quality} for {job_data['job_title']} at {job_data['company_name']}."
        )

        # Skills breakdown
        if feature_scores["skills"] >= 0.7:
            explanations.append(
                f"Strong skill match: {len(matching_skills)}/{len(matching_skills)+len(missing_skills)} key skills."
            )
        elif feature_scores["skills"] >= 0.4:
            explanations.append(
                f"Partial skill match: {len(matching_skills)}/{len(matching_skills)+len(missing_skills)} key skills."
            )
        else:
            explanations.append("Limited skill overlap with requirements.")

        if missing_skills:
            explanations.append(f"Top missing skills: {', '.join(missing_skills[:3])}.")

        # Experience
        if feature_scores["experience"] >= 0.9:
            explanations.append("Your experience meets or exceeds requirements.")
        elif feature_scores["experience"] >= 0.6:
            explanations.append("Your experience is slightly below requirements.")
        else:
            explanations.append("Your experience is significantly below requirements.")

        # Location/Type
        if feature_scores["location"] >= 0.9:
            explanations.append(
                f"Location matches your preference for {job_data['location']}."
            )
        elif feature_scores["location"] >= 0.6:
            explanations.append(f"Location partially matches ({job_data['location']}).")

        if feature_scores["job_type"] >= 0.9:
            explanations.append(
                f"Job type ({job_data['job_type']}) matches your preference."
            )

        return " ".join(explanations) + "."
