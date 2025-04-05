from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import uuid
from app.core.learning.feedback_handler import FeedbackHandler
from app.core.learning.model_updater import ModelUpdater
from app.core.learning.performance_monitor import PerformanceMonitor
from app.utils.logger import logger
from app.utils.security import SecurityUtils

router = APIRouter(
    prefix="/learning", tags=["learning"], responses={404: {"description": "Not found"}}
)

# Initialize services
feedback_handler = FeedbackHandler()
model_updater = ModelUpdater()
performance_monitor = PerformanceMonitor()


class FeedbackRequest(BaseModel):
    user_id: str
    job_id: str
    resume_id: str
    feedback_type: str
    match_score: Optional[float] = None
    metadata: Optional[dict] = None


class ModelVersionResponse(BaseModel):
    version_id: str
    creation_date: datetime
    metrics: dict
    is_active: bool
    description: Optional[str] = None


class PerformanceReportResponse(BaseModel):
    version_id: str
    time_period: str
    metrics: dict


@router.post("/feedback", response_model=dict)
async def record_feedback(feedback: FeedbackRequest):
    """Record user feedback for continuous learning"""
    try:
        recorded = feedback_handler.record_feedback(
            user_id=feedback.user_id,
            job_id=feedback.job_id,
            resume_id=feedback.resume_id,
            feedback_type=feedback.feedback_type,
            match_score=feedback.match_score,
            metadata=feedback.metadata,
        )
        return {"status": "success", "feedback_id": recorded.feedback_id}
    except Exception as e:
        logger.error(f"Failed to record feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feedback recording failed",
        )


@router.get("/conversion-rates", response_model=dict)
async def get_conversion_rates(days: int = 30):
    """Get conversion rates from feedback data"""
    try:
        rates = feedback_handler.calculate_conversion_rates(days)
        return {"time_period": f"Last {days} days", "rates": rates}
    except Exception as e:
        logger.error(f"Failed to calculate conversion rates: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Conversion rate calculation failed",
        )


@router.post("/retrain", response_model=ModelVersionResponse)
async def retrain_model(epochs: int = 3, train_size: float = 0.8):
    """Trigger model retraining based on feedback data"""
    try:
        new_version = model_updater.retrain_model(
            feedback_handler, epochs=epochs, train_size=train_size
        )

        # Record initial metrics
        performance_monitor.record_metrics(new_version.version_id, new_version.metrics)

        return new_version
    except Exception as e:
        logger.error(f"Model retraining failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model retraining failed: {str(e)}",
        )


@router.post("/activate-version/{version_id}", response_model=ModelVersionResponse)
async def activate_version(version_id: str):
    """Activate a specific model version"""
    try:
        model_updater.activate_version(version_id)
        current = model_updater.get_current_version()
        if not current:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Version activation failed",
            )
        return current
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Version activation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Version activation failed",
        )


@router.get("/performance/{version_id}", response_model=PerformanceReportResponse)
async def get_performance_report(version_id: str, days: int = 30):
    """Get performance report for a model version"""
    try:
        report = performance_monitor.generate_performance_report(version_id, days)
        if not report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No performance data found",
            )
        return report
    except Exception as e:
        logger.error(f"Performance report failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Performance report generation failed",
        )


@router.get("/check-for-drops", response_model=dict)
async def check_for_performance_drops(threshold: float = 0.1):
    """Check for significant performance drops"""
    try:
        current_version = model_updater.get_current_version()
        if not current_version:
            return {"alerts": []}

        alert = performance_monitor.detect_performance_drop(
            current_version.version_id, threshold
        )
        return {"alerts": [alert] if alert else []}
    except Exception as e:
        logger.error(f"Performance drop check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Performance drop check failed",
        )
