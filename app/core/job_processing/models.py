# resume_matcher/core/job_processing/models.py
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime
import re


@dataclass
class JobPosting:
    """Structured representation of a job posting"""

    job_id: str
    job_title: str
    company_name: str
    job_type: str
    salary: Optional[Dict[str, Any]]
    experience: Optional[Dict[str, Any]]
    location: str
    description: str
    apply_link: str
    posting_date: datetime
    skills: List[str]
    requirements: List[str]
    responsibilities: List[str]
    benefits: List[str]
    qualifications: List[str]
    normalized_features: Dict[str, Any]


class JobConstants:
    """Constants for job processing"""

    JOB_TYPES = {
        "full time": ["full-time", "full time", "ft", "permanent"],
        "part time": ["part-time", "part time", "pt", "contract"],
        "internship": ["internship", "intern", "trainee"],
        "remote": ["remote", "virtual", "work from home", "wfh"],
        "hybrid": ["hybrid", "flexible"],
    }

    SALARY_PATTERNS = [
        (
            r"\$(\d{1,3}(?:,\d{3})*)(?:\s*-\s*\$(\d{1,3}(?:,\d{3})*))?",
            "USD",
        ),  # $50,000 - $70,000
        (r"(\d+)\s*to\s*(\d+)\s*([A-Z]{3})", "CURRENCY"),  # 50,000 to 70,000 USD
        (r"(\d+)\s*-\s*(\d+)\s*([a-z]+)", "RANGE"),  # 50-70k
        (r"(\d+)\s*([a-z]{3})\s*per\s*(year|month|hour)", "PER_TIME"),  # 50k per year
        (r"(\d+)\s*lpa", "LPA"),  # 15 LPA (common in India)
    ]

    EXPERIENCE_PATTERNS = [
        (r"(\d+)\s*-\s*(\d+)\s*years?", "RANGE"),
        (r"(\d+)\s*\+\s*years?", "MIN"),
        (r"fresher|entry level|0 years?", "ENTRY"),
        (r"senior|lead|manager|director|head", "SENIOR"),
    ]

    SKILL_KEYWORDS = {
        "programming": [
            "python",
            "java",
            "javascript",
            "c++",
            "c#",
            "go",
            "ruby",
            "swift",
            "kotlin",
        ],
        "web": ["html", "css", "react", "angular", "vue", "django", "flask", "node.js"],
        "database": ["sql", "mysql", "postgresql", "mongodb", "redis", "oracle"],
        "devops": ["docker", "kubernetes", "aws", "azure", "gcp", "ci/cd", "terraform"],
        "data": ["pandas", "numpy", "tensorflow", "pytorch", "spark", "hadoop"],
        "testing": ["selenium", "junit", "pytest", "testng", "qa", "quality assurance"],
    }
