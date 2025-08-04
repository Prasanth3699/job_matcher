"""
Matching domain value objects representing immutable business concepts.
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Optional
from enum import Enum


class SkillLevel(Enum):
    """Skill proficiency levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ExperienceLevel(Enum):
    """Professional experience levels"""
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"


@dataclass(frozen=True)
class Skill:
    """Individual skill with proficiency level."""
    
    name: str
    level: SkillLevel = SkillLevel.INTERMEDIATE
    years_experience: Optional[float] = None
    category: Optional[str] = None
    
    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("Skill name cannot be empty")
        
        # Normalize skill name
        object.__setattr__(self, 'name', self.name.strip().lower())
    
    def is_advanced(self) -> bool:
        """Check if skill is at advanced level or higher."""
        return self.level in [SkillLevel.ADVANCED, SkillLevel.EXPERT]
    
    def matches(self, other: 'Skill', exact_level: bool = False) -> bool:
        """Check if this skill matches another skill."""
        if self.name != other.name:
            return False
        
        if exact_level:
            return self.level == other.level
        
        # Allow matching if this skill level is at least the required level
        level_order = [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, 
                      SkillLevel.ADVANCED, SkillLevel.EXPERT]
        return level_order.index(self.level) >= level_order.index(other.level)


@dataclass(frozen=True)
class Skills:
    """Collection of skills with analysis capabilities."""
    
    skills: List[Skill] = field(default_factory=list)
    
    def __post_init__(self):
        # Remove duplicates while preserving order
        unique_skills = []
        seen_names = set()
        
        for skill in self.skills:
            if skill.name not in seen_names:
                unique_skills.append(skill)
                seen_names.add(skill.name)
        
        object.__setattr__(self, 'skills', unique_skills)
    
    def get_skill_names(self) -> Set[str]:
        """Get set of all skill names."""
        return {skill.name for skill in self.skills}
    
    def get_skills_by_category(self, category: str) -> List[Skill]:
        """Get skills filtered by category."""
        return [skill for skill in self.skills if skill.category == category]
    
    def get_advanced_skills(self) -> List[Skill]:
        """Get skills at advanced level or higher."""
        return [skill for skill in self.skills if skill.is_advanced()]
    
    def find_skill(self, name: str) -> Optional[Skill]:
        """Find skill by name."""
        normalized_name = name.strip().lower()
        return next((skill for skill in self.skills if skill.name == normalized_name), None)
    
    def has_skill(self, skill_name: str, min_level: SkillLevel = SkillLevel.BEGINNER) -> bool:
        """Check if collection contains skill at minimum level."""
        skill = self.find_skill(skill_name)
        if not skill:
            return False
        
        level_order = [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, 
                      SkillLevel.ADVANCED, SkillLevel.EXPERT]
        return level_order.index(skill.level) >= level_order.index(min_level)
    
    def intersection(self, other: 'Skills') -> 'Skills':
        """Get skills that exist in both collections."""
        matching_skills = []
        for skill in self.skills:
            other_skill = other.find_skill(skill.name)
            if other_skill and skill.matches(other_skill):
                matching_skills.append(skill)
        
        return Skills(matching_skills)
    
    def difference(self, other: 'Skills') -> 'Skills':
        """Get skills that exist in this collection but not in other."""
        missing_skills = []
        for skill in self.skills:
            if not other.has_skill(skill.name):
                missing_skills.append(skill)
        
        return Skills(missing_skills)
    
    def __len__(self) -> int:
        return len(self.skills)
    
    def __bool__(self) -> bool:
        return bool(self.skills)


@dataclass(frozen=True)
class Experience:
    """Professional experience information."""
    
    total_years: float
    level: ExperienceLevel
    industry_years: Optional[float] = None
    leadership_years: Optional[float] = None
    relevant_roles: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.total_years < 0:
            raise ValueError("Total years cannot be negative")
        
        if self.industry_years and self.industry_years > self.total_years:
            raise ValueError("Industry years cannot exceed total years")
        
        if self.leadership_years and self.leadership_years > self.total_years:
            raise ValueError("Leadership years cannot exceed total years")
    
    def is_senior_level(self) -> bool:
        """Check if experience is at senior level or higher."""
        return self.level in [ExperienceLevel.SENIOR, ExperienceLevel.LEAD, ExperienceLevel.PRINCIPAL]
    
    def meets_requirement(self, required_years: float, required_level: ExperienceLevel) -> bool:
        """Check if experience meets specific requirements."""
        level_order = [ExperienceLevel.ENTRY, ExperienceLevel.JUNIOR, ExperienceLevel.MID,
                      ExperienceLevel.SENIOR, ExperienceLevel.LEAD, ExperienceLevel.PRINCIPAL]
        
        meets_years = self.total_years >= required_years
        meets_level = level_order.index(self.level) >= level_order.index(required_level)
        
        return meets_years and meets_level
    
    def get_industry_ratio(self) -> float:
        """Get ratio of industry experience to total experience."""
        if self.total_years == 0:
            return 0.0
        return (self.industry_years or 0) / self.total_years
    
    def get_leadership_ratio(self) -> float:
        """Get ratio of leadership experience to total experience."""
        if self.total_years == 0:
            return 0.0
        return (self.leadership_years or 0) / self.total_years


@dataclass(frozen=True)
class Score:
    """Matching score with validation."""
    
    value: float
    
    def __post_init__(self):
        if not 0 <= self.value <= 1:
            raise ValueError("Score must be between 0 and 1")
    
    def is_high(self, threshold: float = 0.7) -> bool:
        """Check if score is above threshold."""
        return self.value >= threshold
    
    def is_excellent(self, threshold: float = 0.9) -> bool:
        """Check if score is excellent."""
        return self.value >= threshold
    
    def percentage(self) -> float:
        """Get score as percentage."""
        return self.value * 100
    
    def grade(self) -> str:
        """Get letter grade representation."""
        if self.value >= 0.9:
            return "A"
        elif self.value >= 0.8:
            return "B"
        elif self.value >= 0.7:
            return "C"
        elif self.value >= 0.6:
            return "D"
        else:
            return "F"
    
    def __float__(self) -> float:
        return self.value
    
    def __add__(self, other: 'Score') -> 'Score':
        return Score(min(1.0, self.value + other.value))
    
    def __mul__(self, multiplier: float) -> 'Score':
        return Score(min(1.0, self.value * multiplier))


@dataclass(frozen=True)
class MatchConfidence:
    """Confidence level for match predictions."""
    
    value: float
    source: str = "ensemble"
    
    def __post_init__(self):
        if not 0 <= self.value <= 1:
            raise ValueError("Confidence must be between 0 and 1")
    
    def is_high(self, threshold: float = 0.8) -> bool:
        """Check if confidence is high."""
        return self.value >= threshold
    
    def is_reliable(self, threshold: float = 0.7) -> bool:
        """Check if confidence is reliable for decision making."""
        return self.value >= threshold
    
    def level(self) -> str:
        """Get confidence level description."""
        if self.value >= 0.9:
            return "Very High"
        elif self.value >= 0.8:
            return "High"
        elif self.value >= 0.7:
            return "Medium"
        elif self.value >= 0.5:
            return "Low"
        else:
            return "Very Low"


@dataclass(frozen=True)
class LocationPreference:
    """Location preference and matching criteria."""
    
    preferred_location: str
    remote_acceptable: bool = False
    max_commute_distance: Optional[float] = None
    relocation_acceptable: bool = False
    
    def __post_init__(self):
        if not self.preferred_location or not self.preferred_location.strip():
            raise ValueError("Preferred location cannot be empty")
        
        object.__setattr__(self, 'preferred_location', self.preferred_location.strip().lower())
        
        if self.max_commute_distance and self.max_commute_distance < 0:
            raise ValueError("Max commute distance cannot be negative")
    
    def matches_location(self, job_location: str) -> bool:
        """Check if job location matches preferences."""
        if not job_location:
            return False
        
        normalized_job_location = job_location.strip().lower()
        
        # Check for remote work
        if self.remote_acceptable and "remote" in normalized_job_location:
            return True
        
        # Check for exact location match
        if self.preferred_location in normalized_job_location:
            return True
        
        # Check for relocation if acceptable
        if self.relocation_acceptable:
            return True
        
        return False


@dataclass(frozen=True)
class SalaryExpectation:
    """Salary expectation and requirements."""
    
    min_salary: float
    max_salary: Optional[float] = None
    currency: str = "USD"
    negotiable: bool = True
    
    def __post_init__(self):
        if self.min_salary < 0:
            raise ValueError("Minimum salary cannot be negative")
        
        if self.max_salary and self.max_salary < self.min_salary:
            raise ValueError("Maximum salary cannot be less than minimum salary")
        
        object.__setattr__(self, 'currency', self.currency.upper())
    
    def matches_salary_range(self, job_min: Optional[float], job_max: Optional[float]) -> bool:
        """Check if job salary range matches expectations."""
        if not job_min and not job_max:
            return True  # No salary info available
        
        # If negotiable, be more flexible
        if self.negotiable:
            # Accept if job minimum is within 20% of our minimum
            flexibility_factor = 0.8 if self.negotiable else 1.0
            adjusted_min = self.min_salary * flexibility_factor
        else:
            adjusted_min = self.min_salary
        
        # Check if job's maximum salary meets our minimum
        if job_max and job_max >= adjusted_min:
            return True
        
        # Check if job's minimum salary meets our expectations
        if job_min and job_min >= adjusted_min:
            return True
        
        return False
    
    def salary_score(self, job_min: Optional[float], job_max: Optional[float]) -> float:
        """Calculate salary matching score."""
        if not job_min and not job_max:
            return 0.5  # Neutral score when no salary info
        
        job_salary = job_max or job_min or 0
        
        if job_salary >= self.min_salary:
            # Calculate how much above minimum
            if self.max_salary:
                if job_salary >= self.max_salary:
                    return 1.0
                else:
                    # Scale between min and max
                    range_size = self.max_salary - self.min_salary
                    position = job_salary - self.min_salary
                    return 0.7 + (0.3 * position / range_size)
            else:
                return 1.0
        else:
            # Below minimum, score based on how close
            ratio = job_salary / self.min_salary if self.min_salary > 0 else 0
            return min(0.6, ratio)  # Cap at 0.6 for below-minimum salaries