# resume_matcher/core/analysis/gap_analyzer.py
import logging
from typing import List, Dict, Optional
from datetime import datetime
from dateutil.relativedelta import relativedelta
import spacy
from ...utils.logger import logger


class GapAnalyzer:
    """
    Analyzes gaps in work history and missing skills in resumes.
    Provides actionable recommendations.
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def analyze_work_history(self, experiences: List[Dict]) -> Dict:
        """
        Analyze work history for employment gaps and overlaps.

        Args:
            experiences: List of work experiences from resume

        Returns:
            Dictionary with gap analysis results
        """
        if not experiences:
            return {"gaps": [], "overlaps": [], "total_gap_days": 0}

        # Sort experiences by start date
        sorted_experiences = sorted(
            [exp for exp in experiences if "duration" in exp],
            key=lambda x: self._parse_duration(x["duration"])[0] or datetime.min,
            reverse=True,
        )

        gaps = []
        overlaps = []
        total_gap_days = 0

        for i in range(len(sorted_experiences) - 1):
            current = sorted_experiences[i]
            next_exp = sorted_experiences[i + 1]

            current_start, current_end = self._parse_duration(current["duration"])
            next_start, next_end = self._parse_duration(next_exp["duration"])

            if not all([current_start, current_end, next_start, next_end]):
                continue

            # Check for gap
            if current_end < next_start:
                gap_days = (next_start - current_end).days
                gaps.append(
                    {
                        "between": f"{current.get('company', 'Previous role')} and {next_exp.get('company', 'Next role')}",
                        "start": current_end.strftime("%Y-%m"),
                        "end": next_start.strftime("%Y-%m"),
                        "duration_days": gap_days,
                        "duration_readable": self._days_to_readable(gap_days),
                    }
                )
                total_gap_days += gap_days

            # Check for overlap
            elif current_start < next_end:
                overlap_days = (
                    min(current_end, next_end) - max(current_start, next_start)
                ).days
                if overlap_days > 30:  # Only report significant overlaps
                    overlaps.append(
                        {
                            "between": f"{current.get('company')} and {next_exp.get('company')}",
                            "start": max(current_start, next_start).strftime("%Y-%m"),
                            "end": min(current_end, next_end).strftime("%Y-%m"),
                            "duration_days": overlap_days,
                            "duration_readable": self._days_to_readable(overlap_days),
                        }
                    )

        return {
            "gaps": gaps,
            "overlaps": overlaps,
            "total_gap_days": total_gap_days,
            "total_gap_readable": self._days_to_readable(total_gap_days),
        }

    def analyze_skill_gaps(
        self, resume_skills: List[str], job_skills: List[str]
    ) -> Dict:
        """
        Analyze missing skills compared to target job.

        Args:
            resume_skills: Skills from resume
            job_skills: Required skills from job description

        Returns:
            Dictionary with skill gap analysis
        """
        missing_skills = list(set(job_skills) - set(resume_skills))

        # Categorize missing skills
        categorized = {"technical": [], "soft": [], "tools": [], "certifications": []}

        for skill in missing_skills:
            lower_skill = skill.lower()
            if any(
                term in lower_skill
                for term in ["communication", "teamwork", "leadership"]
            ):
                categorized["soft"].append(skill)
            elif any(term in lower_skill for term in ["certified", "certification"]):
                categorized["certifications"].append(skill)
            elif any(term in lower_skill for term in ["excel", "word", "tool"]):
                categorized["tools"].append(skill)
            else:
                categorized["technical"].append(skill)

        return {
            "missing_skills": missing_skills,
            "categorized_skills": categorized,
            "missing_count": len(missing_skills),
            "match_percentage": (
                1 - len(missing_skills) / len(job_skills) if job_skills else 0
            ),
        }

    def generate_recommendations(
        self, gap_analysis: Dict, skill_analysis: Dict
    ) -> Dict:
        """
        Generate actionable recommendations based on analysis.

        Args:
            gap_analysis: Result from analyze_work_history
            skill_analysis: Result from analyze_skill_gaps

        Returns:
            Dictionary with personalized recommendations
        """
        recommendations = []

        # Work history recommendations
        if gap_analysis["gaps"]:
            rec = {
                "category": "Work History",
                "action": "Address employment gaps",
                "details": f"Your resume shows {len(gap_analysis['gaps'])} employment gaps totaling {gap_analysis['total_gap_readable']}.",
                "suggestions": [
                    "Consider adding explanations for significant gaps (e.g., education, travel, freelance work)",
                    "Highlight any relevant activities during gaps (courses, volunteering, personal projects)",
                ],
            }
            recommendations.append(rec)

        if gap_analysis["overlaps"]:
            rec = {
                "category": "Work History",
                "action": "Clarify overlapping positions",
                "details": f"Your resume shows {len(gap_analysis['overlaps'])} overlapping employment periods.",
                "suggestions": [
                    "Clearly indicate if roles were part-time or concurrent",
                    "Specify any contractor/freelance arrangements",
                ],
            }
            recommendations.append(rec)

        # Skill gap recommendations
        if skill_analysis["missing_count"] > 0:
            rec = {
                "category": "Skills Development",
                "action": "Acquire missing skills",
                "details": f"You're missing {skill_analysis['missing_count']} skills from the target job requirements.",
                "suggestions": [],
            }

            if skill_analysis["categorized_skills"]["technical"]:
                rec["suggestions"].append(
                    f"Focus on technical skills: {', '.join(skill_analysis['categorized_skills']['technical'][:3])}"
                )
            if skill_analysis["categorized_skills"]["certifications"]:
                rec["suggestions"].append(
                    f"Obtain certifications: {', '.join(skill_analysis['categorized_skills']['certifications'])}"
                )
            if skill_analysis["categorized_skills"]["tools"]:
                rec["suggestions"].append(
                    f"Learn tools: {', '.join(skill_analysis['categorized_skills']['tools'])}"
                )

            recommendations.append(rec)

        return {
            "recommendations": recommendations,
            "priority": (
                "High"
                if (gap_analysis["gaps"] or skill_analysis["missing_count"] > 3)
                else "Medium"
            ),
        }

    def _parse_duration(self, duration_str: str) -> tuple:
        """Parse duration string into start/end dates"""
        try:
            if "present" in duration_str.lower():
                end_date = datetime.now()
            else:
                end_date = datetime.strptime(
                    duration_str.split("-")[1].strip(), "%b %Y"
                )

            start_date = datetime.strptime(duration_str.split("-")[0].strip(), "%b %Y")
            return (start_date, end_date)
        except Exception:
            return (None, None)

    def _days_to_readable(self, days: int) -> str:
        """Convert days to human-readable format"""
        years, days = divmod(days, 365)
        months, days = divmod(days, 30)

        parts = []
        if years:
            parts.append(f"{years} year{'s' if years > 1 else ''}")
        if months:
            parts.append(f"{months} month{'s' if months > 1 else ''}")
        if days and not years:
            parts.append(f"{days} day{'s' if days > 1 else ''}")

        return " ".join(parts) if parts else "None"
