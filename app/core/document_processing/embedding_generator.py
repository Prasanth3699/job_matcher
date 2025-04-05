# resume_matcher/core/document_processing/embedding_generator.py
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from ...utils.logger import logger


class EmbeddingGenerator:
    """Generates embeddings for text using sentence transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized embedding generator with model: {model_name}")

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for given text"""
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided for embedding")
            return np.zeros(384)  # Default dimension

        try:
            embedding = self.model.encode(text)
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return np.zeros(384)
