# /core/matching/semantic_matcher.py
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ...utils.logger import logger
from .models import MatchingConstants
from ..utils import safe_lower


class SemanticMatcher:
    """
    Enhanced semantic matcher with multiple models for different purposes:
    - Fast: all-MiniLM-L6-v2 for quick comparisons
    - Accurate: all-mpnet-base-v2 for detailed matching
    - Domain-specific: msmarco-distilbert-base-v4 for job description understanding
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        self.constants = MatchingConstants()
        self.models = self._initialize_models()
        self.current_model = self.models["accurate"]

    def _initialize_models(self) -> Dict[str, SentenceTransformer]:
        """Initialize all models with proper error handling"""
        models = {}
        try:
            logger.info("Loading semantic models...")

            # Fast model (384 dimensions)
            models["fast"] = SentenceTransformer("all-MiniLM-L6-v2")

            # Accurate model (768 dimensions)
            models["accurate"] = SentenceTransformer("all-mpnet-base-v2")

            # Domain-specific model (for job descriptions)
            models["domain"] = SentenceTransformer("msmarco-distilbert-base-v4")

            logger.info("All models loaded successfully")
            return models

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            # Fallback to just the accurate model if others fail
            models["accurate"] = SentenceTransformer("all-mpnet-base-v2")
            return models

    def get_embeddings(
        self, texts: List[str], model_type: str = "accurate"
    ) -> np.ndarray:
        """
        Get embeddings with specified model type
        """
        if not texts:
            return np.array([])

        model = self.models.get(model_type, self.models["accurate"])
        return model.encode(texts, convert_to_numpy=True)

    def calculate_similarity(
        self, text1: str, text2: str, model_type: str = "accurate"
    ) -> float:
        """
        Calculate semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0

        embeddings = self.get_embeddings([text1, text2], model_type)
        similarity = cosine_similarity(
            embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1)
        )[0][0]

        return max(0.0, min(1.0, (similarity + 1) / 2))

    def enhanced_skill_match(
        self, resume_skills: List[str], job_skills: List[str], context: str = ""
    ) -> Tuple[float, List[str], List[str]]:
        """
        Advanced skill matching using:
        - Fast model for initial filtering
        - Accurate model for final scoring
        - Domain model if context is provided
        """
        # Initial cleaning and normalization
        job_skills = [self._normalize_skill(s) for s in job_skills if s]
        resume_skills = [self._normalize_skill(s) for s in resume_skills if s]

        if not job_skills:
            return 1.0, [], []

        if not resume_skills:
            return 0.0, [], job_skills

        # Get context embeddings if provided
        context_embedding = None
        if context:
            context_embedding = self.get_embeddings([context], "domain")[0]

        # Multi-stage matching
        results = []
        for job_skill in job_skills:
            best_match, best_score = self._find_best_skill_match(
                job_skill, resume_skills, context_embedding
            )
            results.append((job_skill, best_match, best_score))

        # Calculate final score
        matched = [
            (js, rm, sc)
            for js, rm, sc in results
            if sc >= self.constants.SKILL_MATCH_THRESHOLD
        ]
        missing = [
            js for js, _, sc in results if sc < self.constants.SKILL_MATCH_THRESHOLD
        ]

        match_score = (
            sum(sc for _, _, sc in matched) / len(job_skills) if job_skills else 0.0
        )
        matched_skills = [js for js, _, _ in matched]

        return float(match_score), matched_skills, missing

    def _find_best_skill_match(
        self,
        job_skill: str,
        resume_skills: List[str],
        context_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[str], float]:
        """Find best match for a single skill with optional context"""
        # First pass with fast model
        fast_embeddings = self.get_embeddings([job_skill] + resume_skills, "fast")
        job_vec = fast_embeddings[0]
        resume_vecs = fast_embeddings[1:]

        fast_scores = cosine_similarity([job_vec], resume_vecs)[0]
        best_fast_idx = np.argmax(fast_scores)

        # Only proceed with detailed matching if fast score is above threshold
        if fast_scores[best_fast_idx] < self.constants.SKILL_MATCH_THRESHOLD - 0.1:
            return None, fast_scores[best_fast_idx]

        # Detailed matching with accurate model
        accurate_embeddings = self.get_embeddings(
            [job_skill, resume_skills[best_fast_idx]], "accurate"
        )
        accurate_score = cosine_similarity(
            [accurate_embeddings[0]], [accurate_embeddings[1]]
        )[0][0]

        # Apply context boost if available
        if context_embedding is not None:
            context_boost = self._get_context_boost(job_skill, context_embedding)
            accurate_score = min(1.0, accurate_score + context_boost * 0.2)

        return resume_skills[best_fast_idx], accurate_score

    def _get_context_boost(self, skill: str, context_embedding: np.ndarray) -> float:
        """Calculate how relevant a skill is to the job context"""
        skill_embedding = self.get_embeddings([skill], "domain")[0]
        return cosine_similarity([skill_embedding], [context_embedding])[0][0]

    def match_skills(
        self, resume_skills: List[str], job_skills: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Enhanced skill matching with improved normalization and semantic similarity

        Args:
            resume_skills: Raw skills extracted from resume
            job_skills: Required skills from job posting

        Returns:
            Tuple of (match_score, matched_skills, missing_skills):
            - match_score: Normalized score (0-1)
            - matched_skills: List of actually matched skills from resume
            - missing_skills: List of truly missing required skills
        """

        # Normalize and clean skills
        def clean_skills(skills: List[str]) -> List[str]:
            cleaned = []
            for skill in skills:
                # Apply normalization
                normalized = self._normalize_skill(skill)

                # Skip empty or invalid skills
                if not normalized or len(normalized.split()) > 3:
                    continue

                # Deduplicate
                if normalized not in cleaned:
                    cleaned.append(normalized)
            return cleaned

        job_skills = clean_skills(job_skills)
        resume_skills = clean_skills(resume_skills)

        # Edge cases
        if not job_skills:
            return 1.0, [], []  # No skills required

        if not resume_skills:
            return 0.0, [], job_skills  # No skills provided

        try:
            # Get embeddings in batches to handle large skill lists
            batch_size = 64
            job_embeddings = []
            for i in range(0, len(job_skills), batch_size):
                batch = job_skills[i : i + batch_size]
                job_embeddings.append(self.get_embeddings(batch))
            job_embeddings = np.vstack(job_embeddings)

            resume_embeddings = []
            for i in range(0, len(resume_skills), batch_size):
                batch = resume_skills[i : i + batch_size]
                resume_embeddings.append(self.get_embeddings(batch))
            resume_embeddings = np.vstack(resume_embeddings)

            # Calculate similarity with threshold
            similarity_matrix = cosine_similarity(job_embeddings, resume_embeddings)
            similarity_matrix = np.round(
                similarity_matrix, 3
            )  # Reduce floating-point precision

            # Find matches
            matched_pairs = []
            missing_skills = []

            for i, job_skill in enumerate(job_skills):
                best_match_idx = np.argmax(similarity_matrix[i])
                best_score = similarity_matrix[i][best_match_idx]

                if best_score >= self.constants.SKILL_MATCH_THRESHOLD:
                    matched_resume_skill = resume_skills[best_match_idx]
                    matched_pairs.append(
                        (job_skill, matched_resume_skill, float(best_score))
                    )
                else:
                    missing_skills.append(job_skill)

            # Calculate weighted score based on similarity scores
            if matched_pairs:
                match_score = sum(score for _, _, score in matched_pairs) / len(
                    job_skills
                )
            else:
                match_score = 0.0

            # Convert to simple skill names for output
            matched_skills = [job_skill for job_skill, _, _ in matched_pairs]

            # Final filtering of missing skills
            missing_skills = [
                skill
                for skill in missing_skills
                if len(skill.split()) <= 3
                and not any(skill in matched for matched in matched_skills)
            ]

            # Convert numpy floats to native Python floats
            match_score = float(match_score)

            return match_score, matched_skills, missing_skills

        except Exception as e:
            logger.error(f"Skill matching failed: {str(e)}")
            return 0.0, [], job_skills

    def _normalize_skill(self, skill: str) -> str:
        """Normalize skill names using the mapping"""
        skill = safe_lower(skill)

        # Check if this is a variant of a known skill
        for canonical, variants in self.constants.SKILL_NORMALIZATIONS.items():
            if skill == canonical or skill in variants:
                return canonical

        return skill

    def _clean_skill(self, skill: str) -> str:
        """Normalize skill formatting"""
        skill = safe_lower(skill).strip(".,!?(){}[]")
        # Remove common prefixes
        for prefix in ["knowledge of", "experience with", "proficient in"]:
            if skill.startswith(prefix):
                skill = skill[len(prefix) :].strip()
        return skill
