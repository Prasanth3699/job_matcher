# resume_matcher/core/learning/feedback_handler.py
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uuid
from pathlib import Path
import json
from .models import UserFeedback, FeedbackType, ModelVersion
from ...utils.logger import logger
from ...utils.security import SecurityUtils


class FeedbackHandler:
    """
    Handles collection, storage, and processing of user feedback.
    In production, this would use a proper database.
    """

    def __init__(self, storage_path: Path = Path("feedback_data")):
        self.storage_path = storage_path
        self.storage_path.mkdir(exist_ok=True)

    def record_feedback(
        self,
        user_id: str,
        job_id: str,
        resume_id: str,
        feedback_type: FeedbackType,
        match_score: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> UserFeedback:
        """Record a new feedback event"""
        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            user_id=user_id,
            job_id=job_id,
            resume_id=resume_id,
            feedback_type=feedback_type,
            timestamp=datetime.now(),
            match_score=match_score,
            metadata=metadata,
        )

        self._store_feedback(feedback)
        return feedback

    def get_feedback_for_job(self, job_id: str) -> List[UserFeedback]:
        """Get all feedback for a specific job"""
        feedback_files = self.storage_path.glob("*.json")
        feedbacks = []

        for file in feedback_files:
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    if data["job_id"] == job_id:
                        feedbacks.append(UserFeedback(**data))
            except Exception as e:
                logger.warning(f"Failed to read feedback file {file}: {str(e)}")

        return feedbacks

    def get_feedback_for_model(self, model_version: str) -> List[UserFeedback]:
        """Get all feedback for a specific model version"""
        feedback_files = self.storage_path.glob("*.json")
        feedbacks = []

        for file in feedback_files:
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    if data.get("metadata", {}).get("model_version") == model_version:
                        feedbacks.append(UserFeedback(**data))
            except Exception as e:
                logger.warning(f"Failed to read feedback file {file}: {str(e)}")

        return feedbacks

    def _store_feedback(self, feedback: UserFeedback) -> None:
        """Store feedback in JSON file"""
        try:
            file_path = self.storage_path / f"{feedback.feedback_id}.json"
            with open(file_path, "w") as f:
                json.dump(feedback.dict(), f, default=str)
        except Exception as e:
            logger.error(f"Failed to store feedback: {str(e)}")
            raise

    def calculate_conversion_rates(self, days: int = 30) -> Dict[str, float]:
        """
        Calculate conversion rates from feedback data.

        Returns:
            Dictionary with conversion rates:
            - impression_to_click
            - click_to_apply
            - apply_to_hired
        """
        feedback_files = self.storage_path.glob("*.json")
        events = {"impression": 0, "click": 0, "apply": 0, "hired": 0}

        cutoff_date = datetime.now() - timedelta(days=days)

        for file in feedback_files:
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    timestamp = datetime.strptime(
                        data["timestamp"], "%Y-%m-%dT%H:%M:%S.%f"
                    )
                    if timestamp >= cutoff_date:
                        events[data["feedback_type"]] += 1
            except Exception as e:
                logger.warning(f"Failed to process feedback file {file}: {str(e)}")

        rates = {}
        if events["impression"] > 0:
            rates["impression_to_click"] = events["click"] / events["impression"]
        if events["click"] > 0:
            rates["click_to_apply"] = events["apply"] / events["click"]
        if events["apply"] > 0:
            rates["apply_to_hired"] = events["hired"] / events["apply"]

        return rates
