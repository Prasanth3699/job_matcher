# resume_matcher/core/matching/service.py
from typing import List, Dict, Any
from pathlib import Path
import json
import numpy as np

from .semantic_matcher import SemanticMatcher
from .hybrid_scorer import HybridScorer
from ...utils.logger import logger


class MatchingService:
    """
    High-level service for matching resumes to jobs.
    Handles data preparation, matching execution, and result formatting.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.scorer = HybridScorer(model_name)

    def match_resume_to_jobs(
        self,
        resume_data: Dict[str, Any],
        jobs_data: List[Dict[str, Any]],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Match a resume to multiple job postings.

        Args:
            resume_data: Processed resume data
            jobs_data: List of processed job data
            top_n: Number of top matches to return

        Returns:
            List of match results with scores and explanations
        """

        try:
            results = self.scorer.match_resume_to_jobs(resume_data, jobs_data, top_n)
            return [self._format_result(result) for result in results]
        except Exception as e:
            logger.error(f"Matching failed: {str(e)}")
            raise

    def match_resume_to_job_file(
        self, resume_data: Dict[str, Any], jobs_file: Path, top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Match a resume to jobs from a JSON file.

        Args:
            resume_data: Processed resume data
            jobs_file: Path to JSON file containing job data
            top_n: Number of top matches to return

        Returns:
            List of match results with scores and explanations
        """
        try:
            with open(jobs_file, "r", encoding="utf-8") as f:
                jobs_data = json.load(f)

            if not isinstance(jobs_data, list):
                jobs_data = [jobs_data]

            return self.match_resume_to_jobs(resume_data, jobs_data, top_n)
        except Exception as e:
            logger.error(f"File-based matching failed: {str(e)}")
            raise

    def _format_result(self, result) -> Dict[str, Any]:
        """Convert MatchResult to serializable dictionary with type conversion"""

        def convert_numpy_types(value):
            if isinstance(value, np.generic):
                return value.item()  # Convert numpy types to native Python types
            elif isinstance(value, (list, tuple)):
                return [convert_numpy_types(v) for v in value]
            elif isinstance(value, dict):
                return {k: convert_numpy_types(v) for k, v in value.items()}
            return value

        formatted = {
            "job_id": result.job_id,
            "overall_score": convert_numpy_types(result.overall_score),
            "score_breakdown": convert_numpy_types(result.score_breakdown),
            "missing_skills": [
                skill
                for skill in (result.missing_skills or [])
                if skill and len(skill.split()) <= 3
            ],
            "matching_skills": [
                skill
                for skill in (result.matching_skills or [])
                if skill  # Filter empty strings
            ],
            "experience_match": convert_numpy_types(result.experience_match),
            "salary_match": convert_numpy_types(result.salary_match),
            "location_match": convert_numpy_types(result.location_match),
            "job_type_match": convert_numpy_types(result.job_type_match),
            "explanation": result.explanation,
            "job_details": {
                "job_title": result.job_title,
                "company_name": result.company_name,
                "location": result.location,
                "job_type": result.job_type,
                "apply_link": result.apply_link,
            },
        }
        return formatted
