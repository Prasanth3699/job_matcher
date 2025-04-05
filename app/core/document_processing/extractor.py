# resume_matcher/core/document_processing/extractor.py
import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import spacy
from spacy.tokens import Doc
from spacy.lang.en.stop_words import STOP_WORDS as ENGLISH_STOP_WORDS
from ...utils.logger import logger


MIN_SKILL_LENGTH = 2  # Minimum characters for a skill
MAX_SKILL_LENGTH = 30  # Maximum characters for a skill
MAX_WORDS_IN_SKILL = 3  # Maximum words in a skill phrase


@dataclass
class ResumeData:
    """Structured representation of extracted resume data"""

    raw_text: str
    skills: List[str]
    experiences: List[Dict]
    education: List[Dict]
    personal_info: Dict[str, str]
    certifications: List[str]
    projects: List[Dict]
    languages: List[str]
    metadata: Dict[str, Any]


class ResumeExtractor:
    """
    Extracts structured information from resume text using NLP and pattern matching.
    """

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found, downloading...")
            from spacy.cli import download

            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Regex patterns for extraction
        self.email_pattern = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
        self.phone_pattern = re.compile(
            r"(\+?\d{1,3}[-\.\s]?)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}"
        )
        self.url_pattern = re.compile(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+")

    def extract(self, raw_text: str, metadata: Optional[Dict] = None) -> ResumeData:
        """
        Extract structured information from resume text.

        Args:
            raw_text: Raw text content of the resume
            metadata: Optional metadata from the parser

        Returns:
            ResumeData object with structured information
        """
        if not metadata:
            metadata = {}

        doc = self.nlp(raw_text)

        return ResumeData(
            raw_text=raw_text,
            skills=self._extract_skills(doc),
            experiences=self._extract_experiences(doc),
            education=self._extract_education(doc),
            personal_info=self._extract_personal_info(doc),
            certifications=self._extract_certifications(doc),
            projects=self._extract_projects(doc),
            languages=self._extract_languages(doc),
            metadata=metadata,
        )

    def _extract_skills(self, text: str) -> List[str]:
        """Improved skill extraction with better filtering"""
        skills = set()

        # First extract using noun phrases
        doc = self.nlp(text)
        for chunk in doc.noun_chunks:
            skill = chunk.text.lower().strip()
            if self._is_valid_skill(skill):
                skills.add(skill)

        # Then extract from entities
        for ent in doc.ents:
            if ent.label_ in ["SKILL", "TECH"]:
                skill = ent.text.lower().strip()
                if self._is_valid_skill(skill):
                    skills.add(skill)

        # Finally extract from custom patterns
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and self._is_valid_skill(token.text):
                skills.add(token.text.lower())

        return sorted(skills)

    def _is_valid_skill(self, text: str) -> bool:
        """Check if a text fragment is a valid skill"""
        text = text.strip()
        words = text.split()

        # Length checks
        if not (MIN_SKILL_LENGTH <= len(text) <= MAX_SKILL_LENGTH):
            return False

        # Word count check
        if len(words) > MAX_WORDS_IN_SKILL:
            return False

        # Stopword check
        if any(word in ENGLISH_STOP_WORDS for word in words):
            return False

        # Special character check
        if any(char in "(){}[]<>" for char in text):
            return False

        return True

    def _extract_experiences(self, doc: Doc) -> List[Dict]:
        """Extract work experience using pattern matching"""
        experiences = []
        current_exp = {}

        # This is a simplified version - would be enhanced with proper NLP
        for sent in doc.sents:
            text = sent.text.lower()

            # Detect company names (simplified)
            if " at " in text:
                parts = text.split(" at ")
                if len(parts) > 1:
                    current_exp["position"] = parts[0].strip()
                    current_exp["company"] = parts[1].split("(")[0].strip()

            # Detect duration patterns
            duration_match = re.search(
                r"(\w+\s?\d{4})\s?[-–—]\s?(\w+\s?\d{4}|present)", text
            )
            if duration_match:
                current_exp["duration"] = (
                    f"{duration_match.group(1)} - {duration_match.group(2)}"
                )

            # Detect responsibilities
            if "responsibilities:" in text or "duties:" in text:
                current_exp["responsibilities"] = sent.text

            # If we have a complete experience, add it
            if current_exp.get("company") and current_exp.get("position"):
                experiences.append(current_exp)
                current_exp = {}

        return experiences

    def _extract_education(self, doc: Doc) -> List[Dict]:
        """Extract education information"""
        education = []

        for sent in doc.sents:
            text = sent.text.lower()

            # Match degree patterns
            degree_keywords = [
                "bachelor",
                "master",
                "phd",
                "doctorate",
                "diploma",
                "degree",
            ]
            if any(keyword in text for keyword in degree_keywords):
                edu_entry = {
                    "degree": (
                        sent.text.split(" in ")[0] if " in " in sent.text else sent.text
                    ),
                    "institution": self._extract_institution(sent.text),
                    "year": self._extract_year(sent.text),
                }
                education.append(edu_entry)

        return education

    def _extract_personal_info(self, doc: Doc) -> Dict[str, str]:
        """Extract personal information (name, email, phone, etc.)"""
        info = {}

        # Extract email
        emails = self.email_pattern.findall(doc.text)
        if emails:
            info["email"] = emails[0]

        # Extract phone
        phones = self.phone_pattern.findall(doc.text)
        if phones:
            info["phone"] = phones[0]

        # Extract name (first sentence is often the name)
        if len(doc.text.strip()) > 0:
            first_part = doc.text.split("\n")[0].strip()
            if (
                not any(char.isdigit() for char in first_part)
                and len(first_part.split()) <= 4
            ):
                info["name"] = first_part

        return info

    def _extract_certifications(self, doc: Doc) -> List[str]:
        """Extract certifications"""
        certs = []
        cert_keywords = ["certified", "certification", "certificate"]

        for sent in doc.sents:
            if any(keyword in sent.text.lower() for keyword in cert_keywords):
                certs.append(sent.text)

        return certs

    def _extract_projects(self, doc: Doc) -> List[Dict]:
        """Extract projects information"""
        projects = []
        current_project = {}

        for sent in doc.sents:
            text = sent.text.lower()

            if "project:" in text or "project name:" in text:
                current_project["name"] = sent.text.split(":")[1].strip()
            elif "description:" in text and current_project.get("name"):
                current_project["description"] = sent.text.split(":")[1].strip()
                projects.append(current_project)
                current_project = {}

        return projects

    def _extract_languages(self, doc: Doc) -> List[str]:
        """Extract known languages"""
        languages = []
        lang_keywords = ["english", "spanish", "french", "german", "hindi", "mandarin"]

        for sent in doc.sents:
            if "languages:" in sent.text.lower():
                for lang in lang_keywords:
                    if lang in sent.text.lower():
                        languages.append(lang.capitalize())

        return languages

    def _extract_institution(self, text: str) -> str:
        """Helper to extract institution name"""
        if " at " in text:
            return text.split(" at ")[1].split(",")[0].strip()
        elif " from " in text:
            return text.split(" from ")[1].split(",")[0].strip()
        return ""

    def _extract_year(self, text: str) -> str:
        """Helper to extract year"""
        year_match = re.search(r"\b(19|20)\d{2}\b", text)
        return year_match.group(0) if year_match else ""
