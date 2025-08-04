"""
Machine Learning constants and model parameters for the resume job matcher service.
"""

from typing import Dict, List, Any


class MLConstants:
    """Core machine learning constants."""
    
    # Model types
    PRIMARY_MODEL_TYPE = "semantic"
    SECONDARY_MODEL_TYPE = "feature_based"
    ENSEMBLE_MODEL_TYPE = "hybrid"
    
    # Embedding dimensions
    MPNET_DIMENSIONS = 768
    MINILM_DIMENSIONS = 384
    BERT_DIMENSIONS = 768
    DEFAULT_DIMENSIONS = 768
    
    # Processing parameters
    DEFAULT_BATCH_SIZE = 32
    OPTIMAL_BATCH_SIZE = 16
    MAX_BATCH_SIZE = 128
    MIN_BATCH_SIZE = 1
    
    # Text processing limits
    MAX_SEQUENCE_LENGTH = 512
    MIN_SEQUENCE_LENGTH = 10
    DEFAULT_MAX_LENGTH = 512
    
    # GPU/CPU configuration
    USE_GPU_DEFAULT = True
    DEVICE_AUTO = "auto"
    DEVICE_CPU = "cpu"
    DEVICE_CUDA = "cuda"


class ModelParameters:
    """Pre-trained model configurations and parameters."""
    
    # Primary semantic matching models
    PRIMARY_MODELS = {
        "all-mpnet-base-v2": {
            "name": "sentence-transformers/all-mpnet-base-v2",
            "dimensions": 768,
            "max_length": 384,
            "batch_size": 16,
            "description": "High quality general-purpose model"
        },
        "all-MiniLM-L6-v2": {
            "name": "sentence-transformers/all-MiniLM-L6-v2", 
            "dimensions": 384,
            "max_length": 256,
            "batch_size": 32,
            "description": "Fast and efficient model"
        },
        "all-distilroberta-v1": {
            "name": "sentence-transformers/all-distilroberta-v1",
            "dimensions": 768,
            "max_length": 512,
            "batch_size": 16,
            "description": "Balanced speed and quality"
        }
    }
    
    # Skill extraction models
    SKILL_EXTRACTION_MODELS = {
        "jobbert": {
            "name": "jjzha/jobbert_skill_extraction",
            "dimensions": 768,
            "max_length": 512,
            "batch_size": 8,
            "description": "Specialized for job skill extraction"
        },
        "dialogpt": {
            "name": "microsoft/DialoGPT-medium",
            "dimensions": 1024,
            "max_length": 512,
            "batch_size": 4,
            "description": "Conversational AI for skill understanding"
        }
    }
    
    # Default model selections
    DEFAULT_PRIMARY_MODEL = "all-mpnet-base-v2"
    DEFAULT_SKILL_MODEL = "jobbert"
    DEFAULT_CLASSIFICATION_MODEL = "dialogpt"
    
    # Model loading parameters
    MODEL_CACHE_DIR = "./model_cache"
    TRUST_REMOTE_CODE = False
    USE_AUTH_TOKEN = False
    REVISION = "main"


class MatchingWeights:
    """Feature weights for hybrid matching algorithm."""
    
    # Primary feature weights (Phase 1 - Current Implementation)
    CURRENT_WEIGHTS = {
        "semantic_similarity": 0.35,
        "skill_match": 0.30,
        "experience_match": 0.15,
        "location_match": 0.08,
        "salary_match": 0.07,
        "job_type_match": 0.05
    }
    
    # Enhanced feature weights (Phase 2 - Future Implementation)
    ENHANCED_WEIGHTS = {
        "semantic_similarity": 0.25,
        "skill_match": 0.25,
        "experience_match": 0.15,
        "education_match": 0.10,
        "location_match": 0.08,
        "salary_match": 0.07,
        "job_type_match": 0.05,
        "company_culture_match": 0.03,
        "career_progression_match": 0.02
    }
    
    # Ensemble model weights (Phase 2)
    ENSEMBLE_WEIGHTS = {
        "primary_semantic": 0.40,
        "secondary_semantic": 0.25,
        "feature_based": 0.20,
        "rule_based": 0.10,
        "user_feedback": 0.05
    }
    
    # Weight validation
    MIN_WEIGHT = 0.0
    MAX_WEIGHT = 1.0
    WEIGHT_SUM_TOLERANCE = 0.001  # Allow small floating point errors
    
    @classmethod
    def validate_weights(cls, weights: Dict[str, float]) -> bool:
        """Validate that weights sum to 1.0 and are within valid range."""
        total = sum(weights.values())
        return (
            abs(total - 1.0) <= cls.WEIGHT_SUM_TOLERANCE and
            all(cls.MIN_WEIGHT <= w <= cls.MAX_WEIGHT for w in weights.values())
        )


class ScoringThresholds:
    """Scoring thresholds and similarity parameters."""
    
    # Overall match score thresholds
    EXCELLENT_MATCH = 0.85
    VERY_GOOD_MATCH = 0.75
    GOOD_MATCH = 0.65
    FAIR_MATCH = 0.50
    POOR_MATCH = 0.35
    MIN_VIABLE_MATCH = 0.30
    
    # Feature-specific thresholds
    SKILL_SIMILARITY_THRESHOLD = 0.75
    SEMANTIC_SIMILARITY_THRESHOLD = 0.70
    EXPERIENCE_MATCH_THRESHOLD = 0.60
    LOCATION_MATCH_THRESHOLD = 0.80
    SALARY_MATCH_THRESHOLD = 0.70
    
    # Penalty thresholds
    MAX_MISSING_SKILLS_PENALTY = 0.40
    MAX_OVERQUALIFICATION_PENALTY = 0.20
    MAX_UNDERQUALIFICATION_PENALTY = 0.30
    
    # Confidence thresholds
    HIGH_CONFIDENCE = 0.90
    MEDIUM_CONFIDENCE = 0.70
    LOW_CONFIDENCE = 0.50
    MIN_CONFIDENCE = 0.30


class AlgorithmParameters:
    """Algorithm-specific parameters and configurations."""
    
    # TF-IDF parameters
    TFIDF_MAX_FEATURES = 10000
    TFIDF_MIN_DF = 2
    TFIDF_MAX_DF = 0.95
    TFIDF_NGRAM_RANGE = (1, 3)
    
    # Cosine similarity parameters
    COSINE_SIMILARITY_THRESHOLD = 0.5
    COSINE_SIMILARITY_PRECISION = 4
    
    # Skill matching parameters
    SKILL_EXACT_MATCH_SCORE = 1.0
    SKILL_PARTIAL_MATCH_SCORE = 0.7
    SKILL_FUZZY_MATCH_THRESHOLD = 0.8
    SKILL_SYNONYM_MATCH_SCORE = 0.9
    
    # Experience scoring parameters
    EXPERIENCE_EXACT_MATCH_SCORE = 1.0
    EXPERIENCE_TOLERANCE_YEARS = 2
    EXPERIENCE_OVERQUALIFICATION_THRESHOLD = 5
    EXPERIENCE_UNDERQUALIFICATION_THRESHOLD = 2
    
    # Text preprocessing parameters
    REMOVE_STOPWORDS = True
    LEMMATIZE_TEXT = True
    MIN_WORD_LENGTH = 2
    MAX_WORD_LENGTH = 50
    
    # Normalization parameters
    L2_NORMALIZE_EMBEDDINGS = True
    FEATURE_SCALING_METHOD = "standard"  # "standard", "minmax", "robust"
    
    # Ranking parameters
    LEARNING_TO_RANK_ENABLED = False  # Phase 2 feature
    NDCG_K = 10
    MAP_K = 10
    MRR_CUTOFF = 10


class CachingParameters:
    """ML model and computation caching parameters."""
    
    # Embedding cache settings
    EMBEDDING_CACHE_SIZE = 10000
    EMBEDDING_CACHE_TTL = 86400  # 24 hours
    EMBEDDING_HASH_ALGORITHM = "sha256"
    
    # Model cache settings
    MODEL_CACHE_SIZE = 5
    MODEL_CACHE_TTL = 604800  # 7 days
    MODEL_MEMORY_THRESHOLD = 0.8  # 80% of available memory
    
    # Result cache settings
    RESULT_CACHE_SIZE = 1000
    RESULT_CACHE_TTL = 3600  # 1 hour
    PARTIAL_RESULT_CACHE_TTL = 300  # 5 minutes
    
    # Cache strategies
    CACHE_STRATEGY = "LRU"  # LRU, LFU, FIFO
    CACHE_COMPRESSION = True
    CACHE_SERIALIZATION = "pickle"  # pickle, json, msgpack


class ModelVersioning:
    """Model versioning and deployment parameters."""
    
    # Version management
    CURRENT_MODEL_VERSION = "v1.0.0"
    MODEL_REGISTRY_PATH = "./model_artifacts"
    MODEL_METADATA_FILE = "metadata.json"
    
    # Deployment settings
    A_B_TESTING_ENABLED = False  # Phase 2 feature
    CANARY_DEPLOYMENT_PERCENTAGE = 0.1
    ROLLBACK_THRESHOLD = 0.05  # 5% error rate triggers rollback
    
    # Model validation
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    CROSS_VALIDATION_FOLDS = 5
    
    # Performance monitoring
    PERFORMANCE_DEGRADATION_THRESHOLD = 0.1
    ACCURACY_ALERT_THRESHOLD = 0.8
    LATENCY_ALERT_THRESHOLD = 2.0  # seconds
    
    # Model update triggers
    RETRAIN_THRESHOLD_ACCURACY = 0.05
    RETRAIN_THRESHOLD_FEEDBACK = 100  # number of feedback samples
    RETRAIN_SCHEDULE_DAYS = 30


class EnsembleConfig:
    """Ensemble model configuration and weights."""
    
    # Model weights for ensemble combination
    MODEL_WEIGHTS = {
        'primary_semantic': 0.40,    # High-quality semantic embeddings
        'fast_semantic': 0.25,       # Speed-optimized embeddings
        'domain_specific': 0.20,     # Tech industry fine-tuned model
        'feature_based': 0.10,       # Traditional ML features
        'feedback_learned': 0.05     # User feedback neural model
    }
    
    # Cache configuration for models and results
    CACHE_CONFIG = {
        'embedding_ttl': 86400,      # 24 hours
        'result_ttl': 3600,          # 1 hour
        'model_ttl': 604800,         # 7 days
        'batch_cache_size': 1000,    # Max cached batches
        'embedding_cache_size': 10000 # Max cached embeddings
    }
    
    # Performance targets for Phase 2
    PERFORMANCE_TARGETS = {
        'accuracy_improvement': 0.15,     # 15% improvement target
        'latency_reduction': 0.30,        # 30% faster responses
        'cache_hit_rate': 0.85,          # 85% cache hit rate
        'throughput_increase': 1.0,       # 100% throughput increase
        'memory_efficiency': 0.30         # 30% memory reduction
    }
    
    # Model loading and optimization settings
    MODEL_CONFIG = {
        'lazy_loading': True,             # Load models on demand
        'model_quantization': True,       # Use quantized models for speed
        'gpu_acceleration': True,         # Use GPU when available
        'batch_inference': True,          # Process in batches
        'memory_pooling': True           # Pool model memory
    }


class FeedbackConfig:
    """User feedback and learning configuration."""
    
    # Feedback types and weights
    FEEDBACK_WEIGHTS = {
        'job_click': 0.1,
        'job_save': 0.3,
        'application_submit': 0.8,
        'interview_scheduled': 1.0,
        'offer_received': 1.5,
        'job_dismissed': -0.2
    }
    
    # Learning parameters
    LEARNING_PARAMS = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'update_frequency': 100,  # Update model every N feedback samples
        'min_feedback_threshold': 50,  # Minimum feedback before updates
        'feedback_decay': 0.95    # Decay older feedback influence
    }
    
    # Feedback validation
    VALIDATION_RULES = {
        'max_feedback_per_user_per_day': 100,
        'min_session_duration': 10,  # seconds
        'spam_detection_threshold': 0.8,
        'feedback_expiry_days': 365
    }


class RankingConfig:
    """Learning-to-rank configuration."""
    
    # Model architecture
    MODEL_ARCHITECTURE = {
        'hidden_layers': [256, 128, 64],
        'dropout_rate': 0.2,
        'activation': 'relu',
        'output_activation': 'sigmoid',
        'regularization': 'l2',
        'regularization_strength': 0.001
    }
    
    # Training parameters
    TRAINING_PARAMS = {
        'epochs': 100,
        'early_stopping_patience': 10,
        'validation_split': 0.2,
        'test_split': 0.1,
        'cross_validation_folds': 5,
        'learning_rate_scheduler': 'reduce_on_plateau'
    }
    
    # Ranking objectives
    RANKING_OBJECTIVES = {
        'primary': 'relevance',      # Main ranking objective
        'secondary': 'diversity',    # Ensure diverse results
        'tertiary': 'freshness',     # Prefer recent job postings
        'weights': [0.7, 0.2, 0.1]  # Objective weights
    }


class PerformanceMetrics:
    """Performance monitoring and evaluation metrics."""
    
    # Accuracy metrics
    ACCURACY_TARGETS = {
        "overall_accuracy": 0.90,
        "precision": 0.85,
        "recall": 0.80,
        "f1_score": 0.82,
        "auc_roc": 0.88
    }
    
    # Ranking metrics
    RANKING_TARGETS = {
        "ndcg_at_5": 0.85,
        "ndcg_at_10": 0.80,
        "map_at_5": 0.75,
        "map_at_10": 0.70,
        "mrr": 0.80
    }
    
    # Performance metrics (updated for Phase 2)
    PERFORMANCE_TARGETS = {
        "avg_response_time_ms": 1000,    # Improved from 2000-5000ms
        "p95_response_time_ms": 2000,    # Improved from 5000ms+
        "p99_response_time_ms": 3000,    # Improved from 10000ms+
        "throughput_rps": 100,           # Doubled from 50
        "memory_usage_mb": 1500,         # Optimized usage
        "cache_hit_rate": 0.85           # Improved from 0.60
    }
    
    # User satisfaction metrics
    SATISFACTION_TARGETS = {
        "user_satisfaction_score": 0.90,  # Improved from 0.75
        "click_through_rate": 0.75,       # Improved from 0.60
        "conversion_rate": 0.20,          # Improved from 0.15
        "time_to_apply": 240              # Reduced from 300 seconds
    }
    
    # Business impact metrics
    BUSINESS_TARGETS = {
        "application_rate_increase": 0.25,     # 25% more applications
        "interview_rate_increase": 0.40,       # 40% more interviews
        "user_engagement_increase": 0.25,      # 25% more engagement
        "cost_per_match_reduction": 0.20       # 20% cost reduction
    }


class EnsembleModelConfig:
    """Configuration for ensemble model implementations."""
    
    # Model implementation types
    MODEL_IMPLEMENTATIONS = {
        'primary_semantic': {
            'class': 'SentenceTransformerModel',
            'model_name': 'all-mpnet-base-v2',
            'priority': 1,
            'load_on_startup': True
        },
        'fast_semantic': {
            'class': 'SentenceTransformerModel', 
            'model_name': 'all-MiniLM-L6-v2',
            'priority': 2,
            'load_on_startup': True
        },
        'domain_specific': {
            'class': 'CustomSemanticModel',
            'model_name': 'tech_domain_fine_tuned',
            'priority': 3,
            'load_on_startup': False
        },
        'feature_based': {
            'class': 'FeatureBasedMatcher',
            'model_name': 'traditional_ml_features',
            'priority': 4,
            'load_on_startup': True
        },
        'feedback_learned': {
            'class': 'NeuralFeedbackModel',
            'model_name': 'user_feedback_neural',
            'priority': 5,
            'load_on_startup': False
        }
    }
    
    # Batch processing configuration
    BATCH_PROCESSING = {
        'default_batch_size': 32,
        'max_batch_size': 128,
        'min_batch_size': 1,
        'parallel_batches': 4,
        'timeout_seconds': 30
    }
    
    # Model loading timeouts
    LOADING_TIMEOUTS = {
        'model_load_timeout': 60,     # 60 seconds to load a model
        'embedding_timeout': 10,      # 10 seconds for embedding generation
        'prediction_timeout': 30,     # 30 seconds for prediction
        'batch_timeout': 120         # 2 minutes for batch processing
    }