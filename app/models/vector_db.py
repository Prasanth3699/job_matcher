# resume_matcher/data/vector_db.py
import logging
from typing import List, Dict, Optional
import numpy as np
import faiss
from pathlib import Path
import json
from ..utils.logger import logger


class VectorDatabase:
    """
    Manages vector embeddings for semantic search using FAISS.
    Provides persistence and efficient similarity search.
    """

    def __init__(self, storage_path: Path = Path("vector_db")):
        self.storage_path = storage_path
        self.storage_path.mkdir(exist_ok=True)
        self.index = None
        self.id_to_job = {}
        self.dimension = None  # Will be set when we know the model

    def initialize_index(self):
        """Initialize or load FAISS index with the correct dimension"""
        index_file = self.storage_path / "index.faiss"
        meta_file = self.storage_path / "metadata.json"

        if index_file.exists() and meta_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(meta_file, "r") as f:
                    self.id_to_job = json.load(f)
                # Ensure loaded index matches our expected dimension
                if self.dimension and self.index.d != self.dimension:
                    print(
                        f"Warning: Loaded index dimension {self.index.d} doesn't match expected {self.dimension}"
                    )
                    self._create_new_index()
                logger.info("Loaded existing vector index")
            except Exception as e:
                logger.error(f"Failed to load vector index: {str(e)}")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create a new FAISS index with the current dimension"""
        if not self.dimension:
            raise ValueError("Dimension must be set before creating index")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_to_job = {}
        logger.info(f"Created new vector index with dimension {self.dimension}")

    def add_job_embeddings(self, job_embeddings: List[Dict]):
        """
        Add job embeddings to the vector database.

        Args:
            job_embeddings: List of dicts with 'job_id' and 'embedding'
        """
        if not job_embeddings:
            return

        # Convert to numpy arrays
        ids = []
        vectors = []

        for job in job_embeddings:
            if "job_id" not in job or "embedding" not in job:
                continue

            embedding = np.array(job["embedding"], dtype="float32")
            if embedding.shape[0] != self.dimension:
                logger.warning(f"Invalid embedding dimension: {embedding.shape}")
                continue

            ids.append(job["job_id"])
            vectors.append(embedding)

        if not vectors:
            return

        vectors = np.stack(vectors)

        # Add to index
        if not self.index:
            self.initialize_index()

        start_id = len(self.id_to_job)
        self.index.add(vectors)

        # Update metadata
        for i, job_id in enumerate(ids):
            self.id_to_job[str(start_id + i)] = job_id

        self._save_index()

    def search_similar_jobs(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> List[Dict]:
        """
        Fixed version that properly handles embedding dimensions
        """
        if not self.index or len(self.id_to_job) == 0:
            return []

        # Ensure proper shape and type
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype="float32")

        # Reshape to (1, dimension)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Verify dimension
        if query_embedding.shape[1] != self.dimension:
            logger.error(
                f"Query embedding dimension {query_embedding.shape[1]} doesn't match index dimension {self.dimension}"
            )
            return []

        # Search the index
        distances, indices = self.index.search(query_embedding, k)

        # Convert to results
        results = []
        for i, score in zip(indices[0], distances[0]):
            if i < 0:  # FAISS returns -1 for invalid indices
                continue
            job_id = self.id_to_job.get(str(i))
            if job_id:
                results.append({"job_id": job_id, "score": float(score)})

        return results

    def _save_index(self):
        """Save the index and metadata to disk"""
        if not self.index:
            return

        try:
            faiss.write_index(self.index, str(self.storage_path / "index.faiss"))
            with open(self.storage_path / "metadata.json", "w") as f:
                json.dump(self.id_to_job, f)
        except Exception as e:
            logger.error(f"Failed to save vector index: {str(e)}")

    def get_job_count(self) -> int:
        """Get number of jobs in the vector database"""
        return len(self.id_to_job) if self.id_to_job else 0
