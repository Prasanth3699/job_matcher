"""
Matching domain module containing entities, services, and repositories
for resume-to-job matching business logic.
"""

from .entities import Match, MatchResult, MatchStatus
from .value_objects import Skills, Experience, Score, MatchConfidence
from .services import MatchingDomainService
from .repositories import MatchingRepository

__all__ = [
    "Match",
    "MatchResult", 
    "MatchStatus",
    "Skills",
    "Experience",
    "Score",
    "MatchConfidence",
    "MatchingDomainService",
    "MatchingRepository",
]