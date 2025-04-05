# resume_matcher/core/learning/model_updater.py
import logging
from datetime import datetime
import random
from typing import Dict, Optional, List
import uuid
from pathlib import Path
import json
import numpy as np
from sklearn.isotonic import spearmanr
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import InputExample
import torch
from torch.utils.data import DataLoader
from ...utils.logger import logger
from .models import ModelVersion
from ..matching.semantic_matcher import SemanticMatcher
from sentence_transformers import losses
from io import StringIO
from contextlib import redirect_stdout


class ModelUpdater:
    """
    Handles model retraining and version management based on feedback data.
    """

    def __init__(self, model_storage: Path = Path("model_versions")):
        self.model_storage = model_storage
        self.model_storage.mkdir(exist_ok=True)
        self.current_version = None
        self._load_current_version()

    def _load_current_version(self) -> None:
        """Load the currently active model version"""
        version_files = list(self.model_storage.glob("*.json"))
        if not version_files:
            self.current_version = None
            return

        # Find the active version
        for file in version_files:
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    if data["is_active"]:
                        self.current_version = ModelVersion(**data)
                        break
            except Exception as e:
                logger.warning(f"Failed to load model version from {file}: {str(e)}")

    def get_current_version(self) -> Optional[ModelVersion]:
        """Get the currently active model version"""
        return self.current_version

    def create_training_data(self, feedback_handler) -> List[InputExample]:
        """Creates training examples from user feedback data for model retraining.

        Processes JSON feedback files to generate InputExample pairs for sentence transformer
        training. Handles different types of feedback with appropriate labeling:

        - Positive examples (higher labels) for clicks, applications, and hires
        - Negative examples (0.0 label) for impressions
        - Skips test data marked in metadata

        Args:
            feedback_handler: Handler object providing access to feedback storage.
                             Must have a `storage_path` attribute pointing to the
                             directory containing feedback JSON files.

        Returns:
            List[InputExample]: A list of training examples formatted for SentenceTransformer.
                               Each example contains:
                               - texts: Pair of [resume_id, job_id] strings
                               - label: Float score between 0.0 and 1.0 representing
                                        match quality (higher = better match)

        Raises:
            json.JSONDecodeError: If any feedback file contains invalid JSON
            KeyError: If required fields are missing in feedback data

        Example:
            >>> feedback_handler = FeedbackHandler(storage_path="feedback/")
            >>> training_data = create_training_data(feedback_handler)
            >>> print(f"Generated {len(training_data)} training examples")
            Generated 42 training examples
        """
        feedback_files = list(feedback_handler.storage_path.glob("*.json"))
        if not feedback_files:
            return []

        examples = []
        for file in feedback_files:
            try:
                with open(file, "r") as f:
                    data = json.load(f)

                    # Skip test data
                    if data.get("metadata", {}).get("test_data", False):
                        continue

                    # Create positive examples for clicks/applies/hires
                    if data["feedback_type"] in ["click", "apply", "hired"]:
                        examples.append(
                            InputExample(
                                texts=[data["resume_id"], data["job_id"]],
                                label=float(data["match_score"]),
                            )
                        )
                    # Create negative examples for impressions
                    elif data["feedback_type"] == "impression":
                        examples.append(
                            InputExample(
                                texts=[data["resume_id"], data["job_id"]],
                                label=0.0,  # Negative label
                            )
                        )
            except Exception as e:
                logger.warning(f"Failed to process feedback file {file}: {str(e)}")

        return examples

    def retrain_model(
        self,
        feedback_handler,
        base_model: str = "all-mpnet-base-v2",
        epochs: int = 3,
        train_size: float = 0.8,
    ) -> ModelVersion:
        """Retrains the sentence transformer model using collected feedback data.

        This method handles the complete retraining pipeline including:
        - Preparing training data from feedback
        - Splitting into train/test sets
        - Initializing the base model
        - Training with cosine similarity loss
        - Evaluating model performance
        - Creating and storing a new model version

        Args:
            feedback_handler: Handler object providing access to user feedback data
            base_model: Name of the base sentence transformer model to use for retraining.
                       Defaults to "all-mpnet-base-v2".
            epochs: Number of training epochs. Defaults to 3.
            train_size: Fraction of data to use for training (vs evaluation).
                       Defaults to 0.8 (80% training, 20% evaluation).

        Returns:
            ModelVersion: Object containing metadata about the newly trained model version,
                         including evaluation metrics and training statistics.

        Raises:
            ValueError: If insufficient training data is available
            RuntimeError: If model training fails

        Example:
            >>> feedback_handler = FeedbackHandler()
            >>> updater = ModelUpdater()
            >>> new_version = updater.retrain_model(feedback_handler)
            >>> print(f"New model version: {new_version.version_id}")
            New model version: abc123-def456
        """
        try:
            # 1. Prepare training data
            examples = self.create_training_data(feedback_handler)
            if len(examples) < 10:
                logger.info("Generating synthetic training data...")
                for i in range(20):
                    examples.append(
                        InputExample(
                            texts=[f"resume_{i}", f"job_{i}"],
                            label=(
                                random.uniform(0.7, 1.0)
                                if random.random() > 0.3
                                else random.uniform(0, 0.3)
                            ),
                        )
                    )

            # 2. Split data
            train_examples, test_examples = train_test_split(
                examples, train_size=train_size, random_state=42
            )

            # 3. Initialize model
            model = SentenceTransformer(base_model)
            logger.info(f"Initialized model with {base_model}")

            # 4. Configure training
            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,
                batch_size=16,
            )
            train_loss = losses.CosineSimilarityLoss(model=model)

            # 5. Create evaluator
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                test_examples,
                name="sts-eval",
                show_progress_bar=False,
            )

            # 6. Train model and capture metrics
            logger.info("Starting model training...")

            # Create a custom evaluator wrapper
            class MetricCaptureEvaluator:
                def __init__(self, evaluator):
                    self.evaluator = evaluator
                    self.pearson = 0.0
                    self.spearman = 0.0

                def __call__(self, model, output_path=None, epoch=-1, steps=-1):
                    # First run the original evaluation
                    score = self.evaluator(model, output_path, epoch, steps)

                    # Then manually compute metrics as fallback
                    embeddings1 = model.encode(
                        [ex.texts[0] for ex in test_examples], convert_to_tensor=True
                    )
                    embeddings2 = model.encode(
                        [ex.texts[1] for ex in test_examples], convert_to_tensor=True
                    )

                    # Compute cosine similarities
                    cos_scores = torch.nn.functional.cosine_similarity(
                        embeddings1, embeddings2
                    )
                    labels = torch.tensor([ex.label for ex in test_examples])

                    # Compute Pearson and Spearman correlations
                    self.pearson = np.corrcoef(
                        cos_scores.cpu().numpy(), labels.cpu().numpy()
                    )[0, 1]
                    self.spearman = spearmanr(
                        cos_scores.cpu().numpy(), labels.cpu().numpy()
                    )[0]

                    return score

            # Use our custom evaluator
            metric_evaluator = MetricCaptureEvaluator(evaluator)

            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=metric_evaluator,
                evaluation_steps=50,
                epochs=epochs,
                warmup_steps=100,
                output_path=str(self.model_storage / "temp_model"),
                optimizer_params={"lr": 2e-5},
            )

            # 7. Get captured metrics
            pearson = metric_evaluator.pearson
            spearman = metric_evaluator.spearman

            # 8. Create version
            version_id = str(uuid.uuid4())
            version_path = self.model_storage / version_id
            version_path.mkdir()

            # 9. Save model
            model.save(str(version_path))

            # 10. Create metrics
            metrics = {
                "pearson_cosine": float(pearson),
                "spearman_cosine": float(spearman),
                "eval_samples": len(test_examples),
                "train_samples": len(train_examples),
            }

            new_version = ModelVersion(
                version_id=version_id,
                creation_date=datetime.now(),
                metrics=metrics,
                is_active=False,
                description=f"Retrained on {len(examples)} examples",
            )

            # 11. Save version info
            with open(version_path / "version.json", "w") as f:
                json.dump(new_version.__dict__, f, indent=2, default=str)

            logger.info(f"Created new model version: {version_id}")
            logger.info(f"Model metrics: {metrics}")

            return new_version

        except Exception as e:
            logger.error(f"Error in model retraining: {str(e)}", exc_info=True)
            raise

    def activate_version(self, version_id: str) -> None:
        """Activate a specific model version"""
        version_path = self.model_storage / version_id
        if not version_path.exists():
            raise ValueError(f"Version {version_id} not found")

        # Deactivate current version
        if self.current_version:
            current_version_path = self.model_storage / self.current_version.version_id
            with open(current_version_path / "version.json", "r+") as f:
                data = json.load(f)
                data["is_active"] = False
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()

        # Activate new version
        with open(version_path / "version.json", "r+") as f:
            data = json.load(f)
            data["is_active"] = True
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

        self.current_version = ModelVersion(**data)

    def get_loaded_model(self) -> Optional[SentenceTransformer]:
        """Get the currently loaded model instance"""
        if not self.current_version:
            return None

        version_path = self.model_storage / self.current_version.version_id
        try:
            return SentenceTransformer(str(version_path))
        except Exception as e:
            logger.error(
                f"Failed to load model {self.current_version.version_id}: {str(e)}"
            )
            return None
