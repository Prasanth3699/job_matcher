# resume_matcher/core/matching/models.py
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class MatchResult:
    """Result of matching a resume against a job"""

    job_id: str
    job_title: str
    company_name: str
    location: str
    job_type: str
    apply_link: str
    overall_score: float
    score_breakdown: Dict[str, float]
    missing_skills: List[str]
    matching_skills: List[str]
    experience_match: float
    salary_match: float
    location_match: float
    job_type_match: float
    explanation: str


class MatchingConstants:
    """Constants for matching algorithm"""

    FEATURE_WEIGHTS = {
        "skills": 0.45,
        "experience": 0.20,
        "salary": 0.15,
        "location": 0.10,
        "job_type": 0.10,
        "title": 0.05,
        "company": 0.05,
    }

    SKILL_MATCH_THRESHOLD = 0.75  # Cosine similarity threshold for skill matching
    MIN_SKILLS_FOR_MATCH = 3  # Minimum skills needed for reliable matching

    SEMANTIC_MODELS = {
        "all-MiniLM-L6-v2": {
            "dimensions": 384,
            "description": "General purpose semantic similarity",
        },
        "all-mpnet-base-v2": {
            "dimensions": 768,
            "description": "Higher quality semantic similarity",
        },
    }

    # Add skill normalization mappings
    SKILL_NORMALIZATIONS = {
        # Programming Languages
        "c++": ["cpp", "c plus plus"],
        "c#": ["c-sharp", "csharp"],
        "javascript": ["js", "ecmascript"],
        "typescript": ["ts"],
        "python": ["py"],
        "java": ["j2ee", "j2se"],
        "ruby": ["ruby on rails"],  # Context dependent
        "php": [],
        "go": ["golang"],
        "swift": ["swiftui"],
        "kotlin": [],
        "scala": [],
        "rust": [],
        # Web Technologies
        "html": ["html5"],
        "css": ["css3"],
        "react": ["reactjs", "react.js"],
        "angular": ["angularjs", "angular.js"],
        "vue": ["vuejs", "vue.js"],
        "django": [],
        "flask": [],
        "laravel": [],
        "spring": ["spring boot"],
        "node.js": ["nodejs", "node"],
        "express.js": ["expressjs", "express"],
        # Databases
        "mysql": [],
        "postgresql": ["postgres"],
        "mongodb": ["mongo"],
        "sqlite": [],
        "oracle": ["oracle db"],
        "sql server": ["mssql", "ms sql"],
        "redis": [],
        "dynamodb": ["aws dynamodb"],
        # Cloud/DevOps
        "aws": ["amazon web services"],
        "azure": ["microsoft azure"],
        "gcp": ["google cloud platform"],
        "docker": ["docker container"],
        "kubernetes": ["k8s"],
        "terraform": [],
        "ansible": [],
        "jenkins": [],
        "git": ["git version control"],
        "github": [],
        "gitlab": [],
        # Data Science/AI
        "pandas": [],
        "numpy": ["numerical python"],
        "tensorflow": ["tf"],
        "pytorch": ["torch"],
        "scikit-learn": ["sklearn"],
        "opencv": ["open cv"],
        "spark": ["apache spark"],
        "hadoop": ["apache hadoop"],
        # Mobile
        "android": ["android development"],
        "ios": ["ios development"],
        "flutter": [],
        "react native": ["react-native"],
        # Testing
        "selenium": [],
        "junit": [],
        "pytest": ["python test"],
        "jest": [],
        "mocha": [],
        # Methodologies
        "agile": ["agile methodology"],
        "scrum": ["scrum methodology"],
        "devops": [],
        "ci/cd": ["continuous integration", "continuous deployment"],
        # Other Tech
        "linux": ["linux administration"],
        "bash": ["bash scripting"],
        "powershell": [],
        "rest api": ["restful api"],
        "graphql": [],
        "oauth": ["oauth2"],
        "jwt": ["json web token"],
        # Soft Skills (optional)
        "problem solving": ["problem-solving"],
        "teamwork": ["team work"],
        "communication": ["communication skills"],
        "leadership": ["leadership skills"],
        "time management": ["time-management"],
        "adaptability": ["adaptable"],
        "critical thinking": ["critical-thinking"],
        "creativity": ["creative thinking"],
        "attention to detail": ["attention-to-detail"],
        "customer service": ["customer-service"],
        "project management": ["project-management"],
        "negotiation": ["negotiation skills"],
        "presentation": ["presentation skills"],
        "conflict resolution": ["conflict-resolution"],
        "emotional intelligence": ["emotional-intelligence"],
        "interpersonal skills": ["interpersonal-skills"],
        "work ethic": ["strong work ethic"],
        "self-motivation": ["self-motivated"],
        "collaboration": ["collaborative"],
        "decision making": ["decision-making"],
        "networking": ["networking skills"],
        "mentoring": ["mentoring skills"],
        "sales": ["sales skills"],
        "marketing": ["marketing skills"],
        "business analysis": ["business analyst"],
        "data analysis": ["data analyst"],
        "research": ["research skills"],
        "public speaking": ["public-speaking"],
        "customer relationship management": ["crm"],
        "user experience": ["ux design"],
        "user interface": ["ui design"],
        "graphic design": ["graphic designer"],
        "video editing": ["video editor"],
        "content writing": ["content writer"],
        "copywriting": ["copywriter"],
        "social media": ["social media marketing"],
        "digital marketing": ["digital marketer"],
        "search engine optimization": ["seo"],
        "search engine marketing": ["sem"],
        # Add more as needed...
    }
