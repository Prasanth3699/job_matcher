from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from functools import lru_cache
import os


class ModelConfig(BaseModel):
    name: str
    dimensions: int
    batch_size: int = 32
    max_length: int = 512
    use_gpu: bool = True


class SkillExtractionConfig(BaseModel):
    min_confidence: float = 0.8
    max_skills_per_job: int = 50
    skill_models: List[str] = [
        "jjzha/jobbert_skill_extraction",
        "microsoft/DialoGPT-medium",
        "sentence-transformers/all-mpnet-base-v2",
    ]
    custom_skill_patterns: Dict[str, List[str]] = Field(default_factory=dict)


class MatchingConfig(BaseModel):
    # Async Processing
    celery_broker: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    celery_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    max_workers: int = 4
    task_timeout: int = 300

    # Models Configuration
    primary_model: ModelConfig = ModelConfig(
        name="sentence-transformers/all-mpnet-base-v2", dimensions=768, batch_size=16
    )
    skill_model: ModelConfig = ModelConfig(
        name="jjzha/jobbert_skill_extraction", dimensions=768, batch_size=8
    )
    job_classification_model: ModelConfig = ModelConfig(
        name="microsoft/DialoGPT-medium", dimensions=1024, batch_size=4
    )

    # Matching Weights (learnable/configurable)
    feature_weights: Dict[str, float] = {
        "semantic_similarity": 0.35,
        "skill_match": 0.30,
        "experience_match": 0.15,
        "location_match": 0.08,
        "salary_match": 0.07,
        "job_type_match": 0.05,
    }

    # Thresholds
    min_match_score: float = 0.3
    skill_similarity_threshold: float = 0.75
    max_missing_skills_penalty: float = 0.4

    # Performance Settings
    cache_ttl: int = 3600  # 1 hour
    embedding_cache_size: int = 10000
    batch_processing_size: int = 50

    # Skill Extraction
    skill_extraction: SkillExtractionConfig = SkillExtractionConfig()


@lru_cache()
def get_matching_config() -> MatchingConfig:
    return MatchingConfig()
