from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
from app.core.analytics.user_behaviour import UserBehaviorAnalyzer
from app.core.analytics.market_trends import MarketTrendsAnalyzer
from app.core.analytics.salary_benchmark import SalaryBenchmarker

from app.utils.logger import logger

router = APIRouter(
    prefix="/analytics",
    tags=["analytics"],
    responses={404: {"description": "Not found"}},
)

# Initialize analyzers
user_behavior = UserBehaviorAnalyzer()
market_trends = MarketTrendsAnalyzer()
salary_benchmark = SalaryBenchmarker()


class SalaryPredictionRequest(BaseModel):
    job_type: str
    experience: int
    location: str


@router.get("/user/engagement")
async def get_user_engagement(days: int = 30):
    """Get user engagement analytics"""
    try:
        return user_behavior.analyze_user_engagement(days)
    except Exception as e:
        logger.error(f"User engagement analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis failed")


@router.get("/user/preferences/{user_id}")
async def get_user_preferences(user_id: str):
    """Get user preference insights"""
    try:
        return user_behavior.analyze_user_preferences(user_id)
    except Exception as e:
        logger.error(f"User preference analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis failed")


@router.get("/market/skills")
async def get_skill_demand(days: int = 90):
    """Get skill demand analysis"""
    try:
        return market_trends.analyze_skill_demand(days)
    except Exception as e:
        logger.error(f"Skill demand analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis failed")


@router.get("/market/salaries")
async def get_salary_trends(job_title: Optional[str] = None):
    """Get salary trends analysis"""
    try:
        return market_trends.analyze_salary_trends(job_title)
    except Exception as e:
        logger.error(f"Salary trends analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis failed")


@router.get("/market/jobs")
async def get_job_availability(window_days: int = 7):
    """Get job availability trends"""
    try:
        return market_trends.analyze_job_availability(window_days)
    except Exception as e:
        logger.error(f"Job availability analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis failed")


@router.post("/salary/predict")
async def predict_salary(request: SalaryPredictionRequest):
    """Predict salary range for job parameters"""
    try:
        return salary_benchmark.predict_salary(
            request.job_type, request.experience, request.location
        )
    except Exception as e:
        logger.error(f"Salary prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.get("/salary/distribution")
async def get_salary_distribution(job_title: Optional[str] = None):
    """Get salary distribution for job title"""
    try:
        return salary_benchmark.get_salary_distribution(job_title)
    except Exception as e:
        logger.error(f"Salary distribution analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis failed")


@router.get("/salary/compare")
async def compare_salary(
    current: float, job_title: str, location: Optional[str] = None
):
    """Compare salary to market benchmarks"""
    try:
        return salary_benchmark.compare_to_market(current, job_title, location)
    except Exception as e:
        logger.error(f"Salary comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Comparison failed")
