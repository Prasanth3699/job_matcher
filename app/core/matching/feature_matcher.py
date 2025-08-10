# resume_matcher/core/matching/feature_matcher.py
import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..utils import safe_lower
from ...utils.logger import logger
from .models import MatchingConstants


class FeatureMatcher:
    """
    Handles feature-based matching between resume and job postings.
    Includes skills, experience, salary, location, etc.
    """

    def __init__(self):
        self.constants = MatchingConstants()

    def calculate_feature_similarity(
        self, resume_features: Dict[str, Any], job_features: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate similarity scores for different features.

        Args:
            resume_features: Features extracted from resume
            job_features: Features extracted from job

        Returns:
            Dictionary of feature scores
        """
        scores = {}

        # Skills matching (using pre-calculated matching)
        scores["skills"] = job_features.get("skill_match_score", 0)

        # Experience matching
        scores["experience"] = self._match_experience(
            resume_features.get("experience_level", 0),
            job_features.get("experience_level", 0),
        )

        # Salary matching
        scores["salary"] = self._match_salary(
            resume_features.get("salary_expectation", None),
            job_features.get("salary_level", None),
        )

        # Title matching
        scores["title"] = self._match_title(
            resume_features.get("target_title", ""),
            job_features.get("job_title", ""),
        )

        # Location matching
        scores["location"] = self._match_location(
            resume_features.get("location_preference", ""),
            job_features.get("location", ""),
        )

        # Job type matching
        scores["job_type"] = self._match_job_type(
            resume_features.get("job_type_preference", ""),
            job_features.get("job_type", ""),
        )

        # Title similarity
        scores["title"] = self._match_text_keywords(
            resume_features.get("target_title", ""),
            job_features.get("job_title_keywords", []),
        )

        # Company similarity (simple binary for now)
        scores["company"] = (
            1.0
            if (
                resume_features.get("preferred_companies")
                and job_features.get("company_name_normalized")
                in resume_features.get("preferred_companies", [])
            )
            else 0.0
        )

        return scores

    def _parse_experience(self, exp) -> float:
        """
        Parse various experience formats into a float number of years.
        Handles int, float, and string representations robustly.
        Returns 0.0 if parsing fails.
        """
        if exp is None:
            return 0.0
        # If already a float/int, return as float (nonnegative)
        if isinstance(exp, (int, float)):
            return max(0.0, float(exp))

        # Convert to string and lowercase for processing
        exp_str = str(exp).strip().lower()

        # Match patterns like "3 years", "2 yrs", "at least 5", "5+", "7-10 years"
        # 1. Range, e.g. "7-10 years"
        match = re.match(r"(\d+(?:\.\d+)?)\s*[-to]+\s*(\d+(?:\.\d+)?)", exp_str)
        if match:
            low = float(match.group(1))
            high = float(match.group(2))
            return (low + high) / 2.0

        # 2. Plus at the end (e.g. "3+", "5+ years")
        match = re.match(r"(\d+(?:\.\d+)?)\s*\+\s*(?:years?|yrs?)?", exp_str)
        if match:
            return float(match.group(1))

        # 3. Minimum/at least/similar phrases
        match = re.match(r"(?:minimum|min|at least)\s*(\d+(?:\.\d+)?)", exp_str)
        if match:
            return float(match.group(1))

        # 4. Simple number (e.g. "5", "5 years", "5yrs", "5.5 yr")
        match = re.match(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?|yr)?", exp_str)
        if match:
            return float(match.group(1))

        # 5. Special/fallback cases for 'no experience', 'entry', 'junior', etc.
        if any(
            word in exp_str for word in ["entry", "junior", "fresher", "no experience"]
        ):
            return 0.0

        # If all else fails
        return 0.0

    def _match_experience(self, resume_exp: str, job_exp: str) -> float:
        """More nuanced experience matching"""
        if not job_exp or "0 years" in safe_lower(job_exp):
            return 1.0

        try:
            resume_yrs = self._parse_experience(resume_exp)
            job_yrs = self._parse_experience(job_exp)

            if resume_yrs >= job_yrs:
                return 1.0
            elif resume_yrs >= job_yrs * 0.75:
                return 0.8
            elif resume_yrs >= job_yrs * 0.5:
                return 0.6
            else:
                return 0.3
        except:
            return 0.5  # Default if parsing fails

    def _match_title(self, resume_title: str, job_title: str) -> float:
        """Enhanced title matching"""
        if not resume_title or not job_title:
            return 0.5

        resume_words = set(safe_lower(resume_title).split())
        job_words = set(safe_lower(job_title).split())

        # Exact match
        if safe_lower(resume_title) == safe_lower(job_title):
            return 1.0

        # Partial match
        common_words = resume_words & job_words
        if len(common_words) >= 2:
            return 0.75

        # Weak match
        if len(common_words) >= 1:
            return 0.5

        return 0.0

    def _match_salary(
        self, resume_salary: Optional[int], job_salary: Optional[int]
    ) -> float:
        """Calculate salary match score"""
        if not job_salary:  # No salary info
            return 0.5

        if not resume_salary:  # No resume salary expectation
            return 0.5

        if resume_salary <= job_salary:
            return 1.0
        else:
            # Penalize based on how much higher the expectation is
            return max(0.0, 1.0 - (resume_salary - job_salary) / (job_salary + 1))

    def _match_location(self, resume_pref: str, job_location: str) -> float:
        """Calculate location match score"""
        if not resume_pref or resume_pref.lower() == "any":
            return 0.8  # Slight preference for matching locations

        if not job_location:
            return 0.5

        resume_pref = resume_pref.lower()
        job_location = job_location.lower()

        # Handle remote cases
        is_remote_pref = "remote" in resume_pref
        is_remote_job = "remote" in job_location

        if is_remote_pref and is_remote_job:
            return 1.0
        elif is_remote_job:
            return 0.8  # Partial credit for remote jobs
        elif is_remote_pref:
            return 0.3  # Small penalty if user wants remote but job isn't

        # Exact match
        if resume_pref == job_location:
            return 1.0

        # Partial match (same city/region)
        if any(word in job_location for word in resume_pref.split()):
            return 0.7

        return 0.0

    def _match_job_type(self, resume_pref: str, job_type: str) -> float:
        """Calculate job type match score"""
        if not resume_pref or resume_pref.lower() == "any":
            return 0.8  # Slight preference for matching types

        if not job_type:
            return 0.5

        resume_pref = resume_pref.lower()
        job_type = job_type.lower()

        if resume_pref == job_type:
            return 1.0
        elif "remote" in job_type and "remote" in resume_pref:
            return 1.0
        elif "hybrid" in job_type and "hybrid" in resume_pref:
            return 1.0
        else:
            return 0.0

    def _match_text_keywords(self, text: str, keywords: List[str]) -> float:
        """Calculate text similarity based on keyword overlap"""
        if not keywords:
            return 0.5

        if not text:
            return 0.0

        text_words = set(safe_lower(text).split())
        keyword_set = set(keywords)
        intersection = text_words.intersection(keyword_set)

        return len(intersection) / len(keyword_set)

    def match_single(self, resume_features: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Single job matching interface for ensemble compatibility.
        
        Args:
            resume_features: Features extracted from resume
            job: Job posting data
            
        Returns:
            Dict with score, confidence, explanation, and model_type
        """
        try:
            # Calculate feature similarities
            feature_scores = self.calculate_feature_similarity(resume_features, job)
            
            # Calculate overall score as weighted average
            weights = {
                'skills': 0.4,
                'experience': 0.25,
                'title': 0.2,
                'location': 0.1,
                'salary': 0.05
            }
            
            overall_score = 0.0
            total_weight = 0.0
            
            for feature, score in feature_scores.items():
                if feature in weights and score is not None:
                    overall_score += score * weights[feature]
                    total_weight += weights[feature]
            
            # Normalize if we have any valid scores
            if total_weight > 0:
                overall_score = overall_score / total_weight
            else:
                overall_score = 0.0
            
            # Ensure score is in [0, 1] range
            overall_score = max(0.0, min(1.0, overall_score))
            
            # Calculate confidence based on data completeness
            available_features = sum(1 for score in feature_scores.values() if score is not None)
            confidence = min(1.0, available_features / len(weights) + 0.2)
            
            return {
                'score': float(overall_score),
                'confidence': float(confidence),
                'explanation': f'Feature-based score: {overall_score:.3f} (based on {available_features} features)',
                'model_type': 'feature',
                'feature_breakdown': feature_scores
            }
            
        except Exception as e:
            logger.error(f"Feature matching error: {str(e)}")
            return {
                'score': 0.0,
                'confidence': 0.0,
                'explanation': 'Error in feature matching',
                'model_type': 'feature'
            }

    def match_batch(self, resume_features: Dict[str, Any], jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch job matching interface for ensemble compatibility.
        
        Args:
            resume_features: Features extracted from resume
            jobs: List of job posting data
            
        Returns:
            List of match results for each job
        """
        return [self.match_single(resume_features, job) for job in jobs]
