# resume_matcher/core/job_processing/analyzer.py
import logging
from typing import List, Dict, Any, Optional
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from .models import JobPosting

logger = logging.getLogger(__name__)


class JobAnalyzer:
    """
    Performs advanced analysis on job postings including:
    - Feature engineering for matching
    - Embedding generation
    - Similarity calculations
    """

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_md")  # Medium model for better vectors
        except OSError:
            logger.warning("spaCy model 'en_core_web_md' not found, downloading...")
            from spacy.cli import download

            download("en_core_web_md")
            self.nlp = spacy.load("en_core_web_md")

        self.tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
        self.tfidf_fitted = False

    def analyze_jobs(self, jobs: List[JobPosting]) -> List[JobPosting]:
        """
        Analyze and enhance job postings with additional features.

        Args:
            jobs: List of JobPosting objects

        Returns:
            List of enhanced JobPosting objects
        """
        if not jobs:
            return []

        # Fit TF-IDF if not already fitted
        if not self.tfidf_fitted:
            all_texts = [job.description for job in jobs]
            self.tfidf.fit(all_texts)
            self.tfidf_fitted = True

        # Analyze each job
        for job in jobs:
            self._enhance_job_features(job)

        return jobs

    def _enhance_job_features(self, job: JobPosting) -> None:
        """Add calculated features to job posting"""
        # Add description embedding
        job.normalized_features["description_embedding"] = (
            self._get_description_embedding(job.description)
        )

        # Add TF-IDF vector
        job.normalized_features["tfidf_vector"] = self._get_tfidf_vector(
            job.description
        )

        # Add skill vectors
        job.normalized_features["skill_vectors"] = self._get_skill_vectors(job.skills)

        # Add experience level
        job.normalized_features["experience_level"] = self._get_experience_level(
            job.experience
        )

        # Add salary level if available
        if job.salary and "min" in job.salary:
            job.normalized_features["salary_level"] = self._get_salary_level(job.salary)

    def _get_description_embedding(self, description: str) -> List[float]:
        """Get spaCy document vector for description"""
        doc = self.nlp(description)
        return doc.vector.tolist()

    def _get_tfidf_vector(self, text: str) -> List[float]:
        """Get TF-IDF vector for text"""
        vector = self.tfidf.transform([text])
        return vector.toarray()[0].tolist()

    def _get_skill_vectors(self, skills: List[str]) -> Dict[str, List[float]]:
        """Get embedding vectors for each skill"""
        skill_vectors = {}
        for skill in skills:
            doc = self.nlp(skill)
            skill_vectors[skill] = doc.vector.tolist()
        return skill_vectors

    def _get_experience_level(self, experience: Optional[Dict[str, Any]]) -> int:
        """Convert experience to numerical level"""
        if not experience:
            return 0

        if experience.get("type") == "entry":
            return 0
        elif experience.get("type") == "senior":
            return 2
        elif "min" in experience:
            return min(experience["min"] // 3, 4)  # Scale to 0-4
        return 1  # Default to mid-level

    def _get_salary_level(self, salary: Dict[str, Any]) -> int:
        """Convert salary to numerical level"""
        min_salary = salary["min"]

        # Normalize to USD if needed
        if salary.get("currency") == "INR":
            min_salary = min_salary / 80  # Approximate conversion

        if salary.get("period") == "month":
            min_salary *= 12
        elif salary.get("period") == "hour":
            min_salary *= 2080  # Approximate hours in a work year

        # Scale to 0-5
        if min_salary < 30000:
            return 0
        elif min_salary < 60000:
            return 1
        elif min_salary < 90000:
            return 2
        elif min_salary < 120000:
            return 3
        elif min_salary < 150000:
            return 4
        return 5
