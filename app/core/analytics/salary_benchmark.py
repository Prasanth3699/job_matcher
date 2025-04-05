# resume_matcher/core/analytics/salary_benchmark.py
import logging
from typing import List, Dict, Optional
import pandas as pd
from sqlalchemy.orm import Session
from ...data.models import JobPosting
from ...data.database import db
from ...utils.logger import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np


class SalaryBenchmarker:
    """
    Provides salary benchmarking based on job characteristics.
    Predicts salary ranges for given job parameters.
    """

    def __init__(self):
        self.session_factory = db.session_scope
        self.model = None
        self.features = ["job_type", "experience_min", "experience_max", "location"]
        self.target = "salary_mid"
        self.encoder = None

    def train_model(self):
        """Train salary prediction model"""
        with self.session_factory() as session:
            jobs = (
                session.query(JobPosting)
                .filter(
                    JobPosting.salary_min.isnot(None),
                    JobPosting.salary_max.isnot(None),
                    JobPosting.job_type.isnot(None),
                    JobPosting.location.isnot(None),
                )
                .all()
            )

            if len(jobs) < 100:
                logger.warning("Insufficient data for salary model training")
                return False

            # Prepare data
            df = pd.DataFrame(
                [
                    {
                        "job_type": j.job_type,
                        "experience_min": j.experience_min or 0,
                        "experience_max": j.experience_max or 0,
                        "location": self._normalize_location(j.location),
                        "salary_mid": (j.salary_min + j.salary_max) / 2,
                    }
                    for j in jobs
                ]
            )

            # Encode categorical features
            df_encoded = pd.get_dummies(df[self.features])
            self.encoder = {col: idx for idx, col in enumerate(df_encoded.columns)}

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                df_encoded, df[self.target], test_size=0.2, random_state=42
            )

            # Train model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

            # Evaluate
            score = self.model.score(X_test, y_test)
            logger.info(f"Salary model trained with R^2 score: {score:.2f}")

            return True

    def predict_salary(self, job_type: str, experience: int, location: str) -> Dict:
        """
        Predict salary range for given job parameters.

        Args:
            job_type: Type of job (full time, part time, etc.)
            experience: Years of experience
            location: Job location

        Returns:
            Dictionary with predicted salary metrics
        """
        if not self.model:
            if not self.train_model():
                return {}

        # Prepare input features
        input_data = {
            "job_type": job_type,
            "experience_min": experience,
            "experience_max": experience,
            "location": self._normalize_location(location),
        }

        # One-hot encode
        encoded = np.zeros(len(self.encoder))
        for feature, value in input_data.items():
            col = f"{feature}_{value}"
            if col in self.encoder:
                encoded[self.encoder[col]] = 1

        # Predict
        prediction = self.model.predict([encoded])[0]

        # Calculate range (assuming ±20% of midpoint)
        lower = prediction * 0.8
        upper = prediction * 1.2

        return {
            "predicted_midpoint": prediction,
            "predicted_range": [lower, upper],
            "parameters": input_data,
        }

    def get_salary_distribution(self, job_title: str = None) -> Dict:
        """
        Get salary distribution for a job title or overall.

        Args:
            job_title: Specific job title to analyze

        Returns:
            Dictionary with salary distribution metrics
        """
        with self.session_factory() as session:
            query = session.query(JobPosting).filter(
                JobPosting.salary_min.isnot(None), JobPosting.salary_max.isnot(None)
            )

            if job_title:
                query = query.filter(JobPosting.job_title.ilike(f"%{job_title}%"))

            jobs = query.all()

            if not jobs:
                return {}

            # Calculate statistics
            salaries = [(j.salary_min + j.salary_max) / 2 for j in jobs]

            return {
                "count": len(salaries),
                "mean": np.mean(salaries),
                "median": np.median(salaries),
                "std": np.std(salaries),
                "percentiles": {
                    "10th": np.percentile(salaries, 10),
                    "25th": np.percentile(salaries, 25),
                    "75th": np.percentile(salaries, 75),
                    "90th": np.percentile(salaries, 90),
                },
            }

    def compare_to_market(
        self, current_salary: float, job_title: str, location: str
    ) -> Dict:
        """
        Compare current salary to market benchmarks.

        Args:
            current_salary: Current salary to compare
            job_title: Job title for comparison
            location: Location for comparison

        Returns:
            Dictionary with comparison metrics
        """
        distribution = self.get_salary_distribution(job_title)
        if not distribution:
            return {}

        comparison = {
            "current_salary": current_salary,
            "market_median": distribution["median"],
            "percentile": self._calculate_percentile(current_salary, distribution),
            "comparison": self._get_comparison_text(
                current_salary, distribution["median"]
            ),
        }

        # Add location adjustment if available
        if location:
            location_adj = self._get_location_adjustment(location)
            if location_adj:
                comparison["location_adjusted"] = distribution["median"] * location_adj
                comparison["location_adjustment_factor"] = location_adj

        return comparison

    def _normalize_location(self, location: str) -> str:
        """Normalize location to major categories"""
        if not location:
            return "remote"

        location = location.lower()
        if "remote" in location:
            return "remote"
        elif any(city in location for city in ["new york", "nyc", "manhattan"]):
            return "new_york"
        elif any(city in location for city in ["san francisco", "sf", "bay area"]):
            return "san_francisco"
        elif any(city in location for city in ["london", "uk", "united kingdom"]):
            return "london"
        else:
            return "other"

    def _calculate_percentile(self, salary: float, distribution: Dict) -> float:
        """Calculate salary percentile"""
        if salary < distribution["percentiles"]["10th"]:
            return 10
        elif salary < distribution["percentiles"]["25th"]:
            return 25
        elif salary < distribution["median"]:
            return 50
        elif salary < distribution["percentiles"]["75th"]:
            return 75
        elif salary < distribution["percentiles"]["90th"]:
            return 90
        else:
            return 95

    def _get_comparison_text(self, current: float, median: float) -> str:
        """Generate comparison description"""
        ratio = current / median
        if ratio < 0.7:
            return "Significantly below market"
        elif ratio < 0.9:
            return "Below market"
        elif ratio < 1.1:
            return "Market rate"
        elif ratio < 1.3:
            return "Above market"
        else:
            return "Significantly above market"

    def _get_location_adjustment(self, location: str) -> Optional[float]:
        """Get location-based salary adjustment factor"""
        adjustments = {
            "new_york": 1.2,
            "san_francisco": 1.25,
            "london": 0.9,
            "remote": 0.85,
            "other": 1.0,
        }
        return adjustments.get(self._normalize_location(location))
