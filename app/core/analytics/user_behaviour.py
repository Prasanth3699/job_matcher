# resume_matcher/core/analytics/user_behavior.py
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from ...data.models import Feedback
from ...data.database import db
from ...utils.logger import logger


class UserBehaviorAnalyzer:
    """
    Analyzes user interactions and behavior patterns.
    Provides insights into user engagement and preferences.
    """

    def __init__(self):
        self.session_factory = db.session_scope

    def analyze_user_engagement(self, days: int = 30) -> Dict:
        """
        Analyze user engagement metrics over time.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with engagement metrics
        """
        with self.session_factory() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Get raw feedback data
            feedback = (
                session.query(Feedback).filter(Feedback.created_at >= cutoff_date).all()
            )

            if not feedback:
                return {}

            # Convert to DataFrame
            df = pd.DataFrame(
                [
                    {
                        "user_id": f.user_id,
                        "timestamp": f.created_at,
                        "action": f.feedback_type,
                        "job_id": f.job_id,
                        "match_score": f.match_score,
                    }
                    for f in feedback
                ]
            )

            # Calculate metrics
            daily_active_users = df.groupby(df["timestamp"].dt.date)[
                "user_id"
            ].nunique()

            actions_per_user = df.groupby("user_id")["action"].count()

            conversion_funnel = {
                "impression": len(df[df["action"] == "impression"]),
                "click": len(df[df["action"] == "click"]),
                "apply": len(df[df["action"] == "apply"]),
                "hired": len(df[df["action"] == "hired"]),
            }

            return {
                "time_period": f"Last {days} days",
                "total_users": df["user_id"].nunique(),
                "daily_active_users_avg": daily_active_users.mean(),
                "daily_active_users_median": daily_active_users.median(),
                "actions_per_user_avg": actions_per_user.mean(),
                "actions_per_user_median": actions_per_user.median(),
                "conversion_funnel": conversion_funnel,
                "conversion_rates": self._calculate_conversion_rates(conversion_funnel),
            }

    def analyze_user_preferences(self, user_id: str = None) -> Dict:
        """
        Analyze user preferences based on their interactions.

        Args:
            user_id: Specific user to analyze (None for all users)

        Returns:
            Dictionary with preference insights
        """
        with self.session_factory() as session:
            query = session.query(Feedback)
            if user_id:
                query = query.filter(Feedback.user_id == user_id)

            feedback = query.all()

            if not feedback:
                return {}

            # Convert to DataFrame
            df = pd.DataFrame(
                [
                    {
                        "user_id": f.user_id,
                        "action": f.feedback_type,
                        "job_id": f.job_id,
                        "job_type": f.job.job_type if f.job else None,
                        "location": f.job.location if f.job else None,
                        "salary": (
                            (f.job.salary_min + f.job.salary_max) / 2
                            if f.job and f.job.salary_min and f.job.salary_max
                            else None
                        ),
                        "match_score": f.match_score,
                    }
                    for f in feedback
                    if f.job
                ]
            )

            if df.empty:
                return {}

            # Calculate preferences
            preferred_job_types = (
                df[df["action"].isin(["click", "apply", "hired"])]
                .groupby("job_type")
                .size()
                .nlargest(3)
                .to_dict()
            )

            preferred_locations = (
                df[df["action"].isin(["click", "apply", "hired"])]
                .groupby("location")
                .size()
                .nlargest(3)
                .to_dict()
            )

            avg_salary = df[df["action"].isin(["click", "apply", "hired"])][
                "salary"
            ].mean()

            success_rate_by_score = (
                df.groupby(pd.cut(df["match_score"], bins=[0, 0.3, 0.6, 0.8, 1.0]))[
                    "action"
                ]
                .apply(lambda x: (x.isin(["apply", "hired"])).mean())
                .to_dict()
            )

            return {
                "preferred_job_types": preferred_job_types,
                "preferred_locations": preferred_locations,
                "average_salary_preference": avg_salary,
                "success_rate_by_match_score": success_rate_by_score,
            }

    def _calculate_conversion_rates(self, funnel: Dict) -> Dict:
        """Calculate conversion rates between funnel steps"""
        rates = {}
        if funnel["impression"] > 0:
            rates["impression_to_click"] = funnel["click"] / funnel["impression"]
        if funnel["click"] > 0:
            rates["click_to_apply"] = funnel["apply"] / funnel["click"]
        if funnel["apply"] > 0:
            rates["apply_to_hired"] = funnel["hired"] / funnel["apply"]
        return rates

    def generate_user_segments(self) -> Dict:
        """
        Identify and categorize user segments based on behavior.

        Returns:
            Dictionary with user segments and characteristics
        """
        with self.session_factory() as session:
            feedback = session.query(Feedback).all()

            if not feedback:
                return {}

            df = pd.DataFrame(
                [
                    {
                        "user_id": f.user_id,
                        "action": f.feedback_type,
                        "timestamp": f.created_at,
                        "match_score": f.match_score,
                    }
                    for f in feedback
                ]
            )

            # Calculate user metrics
            user_metrics = df.groupby("user_id").agg(
                {
                    "action": ["count", lambda x: (x == "apply").sum()],
                    "timestamp": ["min", "max"],
                    "match_score": "mean",
                }
            )
            user_metrics.columns = [
                "actions",
                "applications",
                "first_seen",
                "last_seen",
                "avg_match_score",
            ]

            # Calculate days active
            user_metrics["days_active"] = (
                user_metrics["last_seen"] - user_metrics["first_seen"]
            ).dt.days + 1

            # Define segments
            conditions = [
                (user_metrics["applications"] >= 3)
                & (user_metrics["avg_match_score"] > 0.7),
                (user_metrics["actions"] >= 5) & (user_metrics["applications"] == 0),
                (user_metrics["days_active"] > 7) & (user_metrics["actions"] < 3),
                (user_metrics["avg_match_score"] < 0.4),
            ]
            choices = [
                "active_seekers",
                "window_shoppers",
                "passive_users",
                "poor_fit_users",
            ]

            user_metrics["segment"] = np.select(
                conditions, choices, default="regular_users"
            )

            return user_metrics.groupby("segment").size().to_dict()
