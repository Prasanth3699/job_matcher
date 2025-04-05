# resume_matcher/core/analytics/market_trends.py
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy.orm import Session
from ...data.models import JobPosting
from ...data.database import db
from ...utils.logger import logger
from collections import defaultdict


class MarketTrendsAnalyzer:
    """
    Analyzes job market trends and industry patterns.
    Provides insights into skill demand, salary trends, and job availability.
    """

    def __init__(self):
        self.session_factory = db.session_scope

    def analyze_skill_demand(self, days: int = 90) -> Dict:
        """
        Analyze most in-demand skills in the job market.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with skill demand metrics
        """
        with self.session_factory() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            jobs = (
                session.query(JobPosting)
                .filter(
                    JobPosting.posting_date >= cutoff_date,
                    JobPosting.processed_data.isnot(None),
                )
                .all()
            )

            if not jobs:
                return {}

            # Count skill occurrences
            skill_counts = defaultdict(int)
            for job in jobs:
                if "skills" in job.processed_data:
                    for skill in job.processed_data["skills"]:
                        skill_counts[skill.lower()] += 1

            # Get top skills
            top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[
                :20
            ]

            return {
                "time_period": f"Last {days} days",
                "total_jobs_analyzed": len(jobs),
                "top_skills": dict(top_skills),
                "skill_categories": self._categorize_skills(top_skills),
            }

    def analyze_salary_trends(self, job_title: str = None) -> Dict:
        """
        Analyze salary trends over time for specific job titles.

        Args:
            job_title: Specific job title to analyze (None for all jobs)

        Returns:
            Dictionary with salary trends
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

            # Convert to DataFrame
            df = pd.DataFrame(
                [
                    {
                        "job_title": j.job_title,
                        "company": j.company_name,
                        "salary_min": j.salary_min,
                        "salary_max": j.salary_max,
                        "salary_mid": (j.salary_min + j.salary_max) / 2,
                        "posting_date": j.posting_date,
                        "location": j.location,
                        "job_type": j.job_type,
                    }
                    for j in jobs
                ]
            )

            # Calculate trends
            trends = {}
            if not job_title:
                # Overall trends by month
                monthly = df.groupby(df["posting_date"].dt.to_period("M"))[
                    "salary_mid"
                ].agg(["mean", "median", "count"])
                trends["overall_monthly"] = monthly.to_dict()

                # By job type
                by_type = df.groupby("job_type")["salary_mid"].agg(
                    ["mean", "median", "count"]
                )
                trends["by_job_type"] = by_type.to_dict()
            else:
                # For specific job title
                monthly = df.groupby(df["posting_date"].dt.to_period("M"))[
                    "salary_mid"
                ].agg(["mean", "median", "count"])
                trends["monthly"] = monthly.to_dict()

                # By location
                by_location = df.groupby("location")["salary_mid"].agg(
                    ["mean", "median", "count"]
                )
                trends["by_location"] = by_location.to_dict()

            return trends

    def analyze_job_availability(self, window_days: int = 7) -> Dict:
        """
        Analyze job availability and growth trends.

        Args:
            window_days: Rolling window size in days

        Returns:
            Dictionary with job availability metrics
        """
        with self.session_factory() as session:
            jobs = session.query(JobPosting).order_by(JobPosting.posting_date).all()

            if not jobs:
                return {}

            # Convert to DataFrame
            df = pd.DataFrame(
                [
                    {
                        "job_id": j.id,
                        "job_title": j.job_title,
                        "company": j.company_name,
                        "posting_date": j.posting_date,
                        "job_type": j.job_type,
                        "location": j.location,
                    }
                    for j in jobs
                ]
            )

            # Calculate daily counts
            daily_counts = (
                df.groupby(df["posting_date"].dt.date).size().reset_index(name="count")
            )

            # Calculate rolling averages
            daily_counts["rolling_avg"] = (
                daily_counts["count"].rolling(window=window_days).mean()
            )

            # Calculate growth rate
            daily_counts["growth_rate"] = daily_counts["rolling_avg"].pct_change() * 100

            # By job type
            by_type = (
                df.groupby(["job_type", df["posting_date"].dt.date]).size().unstack(0)
            )

            return {
                "daily_counts": daily_counts.set_index("posting_date").to_dict()[
                    "count"
                ],
                "rolling_averages": daily_counts.set_index("posting_date").to_dict()[
                    "rolling_avg"
                ],
                "growth_rates": daily_counts.set_index("posting_date").to_dict()[
                    "growth_rate"
                ],
                "by_job_type": by_type.to_dict(),
            }

    def _categorize_skills(self, skills: List[tuple]) -> Dict:
        """Categorize skills into groups"""
        categories = {
            "programming": [],
            "cloud": [],
            "data": [],
            "devops": [],
            "soft_skills": [],
        }

        for skill, count in skills:
            lower_skill = skill.lower()
            if any(
                term in lower_skill
                for term in ["python", "java", "javascript", "c++", "go"]
            ):
                categories["programming"].append((skill, count))
            elif any(term in lower_skill for term in ["aws", "azure", "gcp", "cloud"]):
                categories["cloud"].append((skill, count))
            elif any(
                term in lower_skill
                for term in ["data", "sql", "analys", "machine learning"]
            ):
                categories["data"].append((skill, count))
            elif any(
                term in lower_skill
                for term in ["docker", "kubernetes", "ci/cd", "terraform"]
            ):
                categories["devops"].append((skill, count))
            elif any(
                term in lower_skill
                for term in ["communication", "team", "leadership", "problem solving"]
            ):
                categories["soft_skills"].append((skill, count))

        # Sort each category by count
        for category in categories:
            categories[category].sort(key=lambda x: x[1], reverse=True)

        return categories
