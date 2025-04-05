# resume_matcher/core/job_processing/parser.py
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import re
from dateutil.parser import parse
from ..document_processing.sanitizer import ContentSanitizer
from .models import JobPosting, JobConstants

logger = logging.getLogger(__name__)


class JobParser:
    """
    Parses and validates raw job postings into structured JobPosting objects.
    Handles all fields including complex description parsing.
    """

    def __init__(self):
        self.sanitizer = ContentSanitizer()
        self.constants = JobConstants()

    def parse_job(self, job_data: Dict[str, Any]) -> JobPosting:
        """
        Parse a single job posting from dictionary format to JobPosting object.

        Args:
            job_data: Dictionary containing job posting data

        Returns:
            JobPosting object with structured data

        Raises:
            ValueError: If required fields are missing or invalid
        """
        try:
            # Validate required fields
            if not all(
                k in job_data
                for k in ["job_title", "company_name", "description", "apply_link"]
            ):
                raise ValueError("Missing required job fields")

            # Generate a unique job ID if not provided
            job_id = job_data.get("job_id", self._generate_job_id(job_data))

            # Parse and sanitize all fields
            job_title = self.sanitizer.sanitize_text(job_data["job_title"])
            company_name = self.sanitizer.sanitize_text(job_data["company_name"])
            description = self.sanitizer.sanitize_text(job_data["description"])

            # Parse job type with normalization
            job_type = self._parse_job_type(job_data.get("job_type", ""))

            # Parse salary
            salary = self._parse_salary(job_data.get("salary"))

            # Parse experience
            experience = self._parse_experience(job_data.get("experience"))

            # Parse location
            location = self._parse_location(job_data.get("location", ""))

            # Parse posting date
            posting_date = self._parse_date(job_data.get("posting_date"))

            # Parse description sections
            desc_analysis = self._analyze_description(description)

            return JobPosting(
                job_id=job_id,
                job_title=job_title,
                company_name=company_name,
                job_type=job_type,
                salary=salary,
                experience=experience,
                location=location,
                description=description,
                apply_link=job_data["apply_link"],
                posting_date=posting_date,
                skills=desc_analysis["skills"],
                requirements=desc_analysis["requirements"],
                responsibilities=desc_analysis["responsibilities"],
                benefits=desc_analysis["benefits"],
                qualifications=desc_analysis["qualifications"],
                normalized_features=self._create_normalized_features(
                    job_title, company_name, description, desc_analysis
                ),
            )

        except Exception as e:
            logger.error(f"Failed to parse job: {str(e)}")
            raise ValueError(f"Job parsing failed: {str(e)}") from e

    def _generate_job_id(self, job_data: Dict[str, Any]) -> str:
        """Generate a unique job ID from job data"""
        from hashlib import md5

        unique_str = f"{job_data['job_title']}_{job_data['company_name']}_{job_data.get('posting_date', '')}"
        return md5(unique_str.encode("utf-8")).hexdigest()

    def _parse_job_type(self, job_type_str: str) -> str:
        """Normalize job type string"""
        if not job_type_str:
            return "full time"

        lower_type = job_type_str.lower()
        for normalized_type, variants in self.constants.JOB_TYPES.items():
            if any(v in lower_type for v in variants):
                return normalized_type
        return "other"

    def _parse_salary(self, salary_str: Optional[str]) -> Optional[Dict[str, Any]]:
        """Parse and normalize salary information"""
        if not salary_str:
            return None

        salary_str = salary_str.lower()
        for pattern, pattern_type in self.constants.SALARY_PATTERNS:
            match = re.search(pattern, salary_str)
            if match:
                if pattern_type == "USD":
                    min_sal = int(match.group(1).replace(",", ""))
                    max_sal = (
                        int(match.group(2).replace(",", ""))
                        if match.group(2)
                        else min_sal
                    )
                    return {
                        "min": min_sal,
                        "max": max_sal,
                        "currency": "USD",
                        "period": "year",
                    }
                elif pattern_type == "LPA":
                    value = int(match.group(1))
                    return {
                        "min": value * 100000,
                        "max": value * 100000,
                        "currency": "INR",
                        "period": "year",
                    }

        return {"raw": salary_str}

    def _parse_experience(self, exp_str: Optional[str]) -> Optional[Dict[str, Any]]:
        """Parse and normalize experience requirements"""
        if not exp_str:
            return None

        exp_str = exp_str.lower()
        for pattern, pattern_type in self.constants.EXPERIENCE_PATTERNS:
            match = re.search(pattern, exp_str)
            if match:
                if pattern_type == "RANGE":
                    min_exp = int(match.group(1))
                    max_exp = int(match.group(2))
                    return {"min": min_exp, "max": max_exp, "type": "range"}
                elif pattern_type == "MIN":
                    min_exp = int(match.group(1))
                    return {"min": min_exp, "type": "min"}
                elif pattern_type == "ENTRY":
                    return {"min": 0, "max": 1, "type": "entry"}
                elif pattern_type == "SENIOR":
                    return {"min": 5, "type": "senior"}

        return {"raw": exp_str}

    def _parse_location(self, location_str: str) -> str:
        """Normalize location string"""
        if not location_str:
            return "remote"

        location_str = location_str.lower().strip()
        if any(
            remote_word in location_str
            for remote_word in self.constants.JOB_TYPES["remote"]
        ):
            return "remote"
        return location_str

    def _parse_date(self, date_str: Optional[str]) -> datetime:
        """Parse posting date with fallback to current date"""
        if not date_str:
            return datetime.now()
        try:
            return parse(date_str)
        except (ValueError, TypeError):
            return datetime.now()

    def _analyze_description(self, description: str) -> Dict[str, List[str]]:
        """
        Analyze job description to extract:
        - Skills
        - Requirements
        - Responsibilities
        - Benefits
        - Qualifications
        """
        # Split description into sections
        sections = self._split_description_sections(description)

        # Extract from each section
        result = {
            "skills": [],
            "requirements": [],
            "responsibilities": [],
            "benefits": [],
            "qualifications": [],
        }

        for section_name, section_text in sections.items():
            if (
                "requirement" in section_name.lower()
                or "qualification" in section_name.lower()
            ):
                result["requirements"].extend(self._extract_requirements(section_text))
                result["qualifications"].extend(
                    self._extract_qualifications(section_text)
                )
            elif "responsibilit" in section_name.lower():
                result["responsibilities"].extend(
                    self._extract_responsibilities(section_text)
                )
            elif "benefit" in section_name.lower() or "perk" in section_name.lower():
                result["benefits"].extend(self._extract_benefits(section_text))
            elif "skill" in section_name.lower() or "technolog" in section_name.lower():
                result["skills"].extend(self._extract_skills(section_text))
            else:
                # Fallback: try to extract from any section
                result["skills"].extend(self._extract_skills(section_text))
                result["requirements"].extend(self._extract_requirements(section_text))

        # Deduplicate and clean
        for key in result:
            result[key] = list(set(result[key]))

        return result

    def _split_description_sections(self, description: str) -> Dict[str, str]:
        """Split description into logical sections"""
        sections = {}
        current_section = "description"
        current_text = []

        # Split by common section headers
        lines = description.split("\n")
        for line in lines:
            if re.match(r"^[A-Z][A-Za-z\s]+:$", line.strip()):
                if current_text:
                    sections[current_section] = "\n".join(current_text)
                    current_text = []
                current_section = line.strip().rstrip(":")
            else:
                current_text.append(line)

        if current_text:
            sections[current_section] = "\n".join(current_text)

        return sections

    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from text"""
        skills = set()
        text_lower = text.lower()

        # Match against known skill keywords
        for category, skill_list in self.constants.SKILL_KEYWORDS.items():
            for skill in skill_list:
                if skill in text_lower:
                    skills.add(skill)

        # Extract from bullet points
        bullet_points = re.findall(r"[-•*]\s*(.*?)(?=\n|$)", text)
        for point in bullet_points:
            for skill in self.constants.SKILL_KEYWORDS.keys():
                if skill in point.lower():
                    skills.add(point.strip())

        return sorted(skills)

    def _extract_requirements(self, text: str) -> List[str]:
        """Extract requirements from text"""
        requirements = []

        # Extract bullet points or numbered lists
        items = re.findall(
            r"[-•*]\s*(.*?)(?=\n|$)|^\d+\.\s*(.*?)(?=\n|$)", text, re.MULTILINE
        )
        for item in items:
            requirement = item[0] or item[1]
            if requirement:
                requirements.append(requirement.strip())

        # Fallback: split by newlines if no bullets found
        if not requirements:
            requirements = [line.strip() for line in text.split("\n") if line.strip()]

        return requirements

    def _extract_responsibilities(self, text: str) -> List[str]:
        """Extract responsibilities from text"""
        return self._extract_requirements(text)  # Same pattern as requirements

    def _extract_benefits(self, text: str) -> List[str]:
        """Extract benefits from text"""
        benefits = []

        # Look for common benefit phrases
        benefit_phrases = [
            "health insurance",
            "dental insurance",
            "vision insurance",
            "401k",
            "retirement plan",
            "paid time off",
            "pto",
            "flexible schedule",
            "remote work",
            "bonus",
            "stock options",
        ]

        for phrase in benefit_phrases:
            if phrase in text.lower():
                benefits.append(phrase)

        # Add bullet points
        benefits.extend(self._extract_requirements(text))

        return benefits

    def _extract_qualifications(self, text: str) -> List[str]:
        """Extract qualifications from text"""
        qualifications = []

        # Look for degree requirements
        degree_keywords = ["bachelor", "master", "phd", "degree", "diploma"]
        for line in text.split("\n"):
            if any(keyword in line.lower() for keyword in degree_keywords):
                qualifications.append(line.strip())

        return qualifications

    def _create_normalized_features(
        self,
        job_title: str,
        company_name: str,
        description: str,
        desc_analysis: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Create normalized features for ML matching"""
        return {
            "job_title_keywords": self._extract_keywords(job_title),
            "company_name_normalized": company_name.lower(),
            "description_keywords": self._extract_keywords(description),
            "skills_normalized": [skill.lower() for skill in desc_analysis["skills"]],
            "requirements_count": len(desc_analysis["requirements"]),
            "responsibilities_count": len(desc_analysis["responsibilities"]),
            "has_degree_requirement": any(
                "bachelor" in qual.lower() or "degree" in qual.lower()
                for qual in desc_analysis["qualifications"]
            ),
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple implementation - would be enhanced with proper NLP
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return list(set(words))
