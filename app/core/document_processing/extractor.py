# resume_extractor.py

import re
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import spacy
from spacy.tokens import Doc
from spacy.lang.en.stop_words import STOP_WORDS as ENGLISH_STOP_WORDS

from transformers import (
    pipeline,
)  # Hugging Face Transformers :contentReference[oaicite:5]{index=5}
from ...utils.logger import logger

# Skill length and phrase limits
MIN_SKILL_LENGTH = 2
MAX_SKILL_LENGTH = 30
MAX_WORDS_IN_SKILL = 3


@dataclass
class ResumeData:
    """Structured representation of extracted resume data"""

    raw_text: str
    skills: List[str]
    experiences: List[Dict[str, Any]]
    education: List[Dict[str, Any]]
    personal_info: Dict[str, str]
    certifications: List[str]
    projects: List[Dict[str, Any]]
    languages: List[str]
    metadata: Dict[str, Any]


class ResumeExtractor:
    """
    Extracts structured information from resume text using spaCy and
    Hugging Face Transformers for skill recognition.
    """

    def __init__(self):
        # Load spaCy English model, download if missing :contentReference[oaicite:6]{index=6}
        try:
            self.nlp = spacy.load(
                "en_core_web_sm"
            )  # :contentReference[oaicite:7]{index=7}
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found, downloading...")
            from spacy.cli import download

            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Initialize Hugging Face skill‐NER pipeline
        try:
            self.skill_ner = pipeline(
                task="ner",
                model="algiraldohe/lm-ner-linkedin-skills-recognition",
                grouped_entities=True,  # merge multi‑token skills :contentReference[oaicite:8]{index=8}
            )
        except Exception as e:
            logger.warning(f"Could not load HF skill‐NER model: {e}")
            self.skill_ner = None

        # Regex patterns
        self.email_pattern = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
        self.phone_pattern = re.compile(
            r"(\+?\d{1,3}[-\.\s]?)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}"
        )
        self.url_pattern = re.compile(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+")

    def extract(
        self, raw_text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ResumeData:
        """
        Extract structured information from resume text.
        """
        if metadata is None:
            metadata = {}
        doc = self.nlp(raw_text)
        return ResumeData(
            raw_text=raw_text,
            skills=self._extract_skills(raw_text),
            experiences=self._extract_experiences(doc),
            education=self._extract_education(doc),
            personal_info=self._extract_personal_info(doc),
            certifications=self._extract_certifications(doc),
            projects=self._extract_projects(doc),
            languages=self._extract_languages(doc),
            metadata=metadata,
        )

    def _is_valid_skill(self, text: str) -> bool:
        """Check if a text fragment is a valid skill phrase."""
        text = text.strip()
        words = text.split()

        if not (MIN_SKILL_LENGTH <= len(text) <= MAX_SKILL_LENGTH):
            return False
        if len(words) > MAX_WORDS_IN_SKILL:
            return False
        if any(w in ENGLISH_STOP_WORDS for w in words):
            return False
        if any(c in "(){}[]<>" for c in text):
            return False
        return True

    def _extract_skills(self, text: str) -> List[str]:
        """
        Extract skills using the HF skill‐NER pipeline first, then fall back
        to spaCy noun‐chunk/entity/token heuristics if needed.
        """
        skills = set()

        # 1) Transformer‐based extraction (highest precision/recall)
        if self.skill_ner:
            try:
                for ent in self.skill_ner(text):
                    word = ent["word"].strip().lower()
                    if self._is_valid_skill(word):
                        skills.add(word)
            except Exception as e:
                logger.warning(f"Skill‐NER pipeline failed: {e}")

        # 2) Fallback: spaCy heuristics if no skills found :contentReference[oaicite:9]{index=9}
        if not skills:
            doc = self.nlp(text)
            # noun chunks
            for chunk in doc.noun_chunks:
                word = chunk.text.lower().strip()
                if self._is_valid_skill(word):
                    skills.add(word)
            # named entities labeled SKILL or TECH
            for ent in doc.ents:
                if ent.label_ in {"SKILL", "TECH"}:
                    word = ent.text.lower().strip()
                    if self._is_valid_skill(word):
                        skills.add(word)
            # single tokens
            for token in doc:
                w = token.text.lower().strip()
                if token.pos_ in {"NOUN", "PROPN"} and self._is_valid_skill(w):
                    skills.add(w)
        logger.info(f"Extracted skills using fallback method: {skills}")
        return sorted(skills)

    def _extract_experiences(self, doc: Doc) -> List[Dict[str, Any]]:
        """Extract work experience using simple pattern matching."""
        experiences = []
        current = {}
        for sent in doc.sents:
            t = sent.text.lower()
            if " at " in t:
                parts = t.split(" at ", 1)
                current["position"] = parts[0].strip()
                current["company"] = parts[1].split("(")[0].strip()
            m = re.search(r"(\w+\s?\d{4})\s?[-–—]\s?(\w+\s?\d{4}|present)", t)
            if m:
                current["duration"] = f"{m.group(1)} - {m.group(2)}"
            if "responsibilities:" in t or "duties:" in t:
                current["responsibilities"] = sent.text
            if current.get("position") and current.get("company"):
                experiences.append(current)
                current = {}
        return experiences

    def _extract_education(self, doc: Doc) -> List[Dict[str, Any]]:
        """Extract education entries by degree keywords."""
        edu = []
        degrees = ["bachelor", "master", "phd", "doctorate", "diploma", "degree"]
        for sent in doc.sents:
            t = sent.text.lower()
            if any(d in t for d in degrees):
                entry = {
                    "degree": (
                        sent.text.split(" in ")[0] if " in " in sent.text else sent.text
                    ),
                    "institution": self._extract_institution(sent.text),
                    "year": self._extract_year(sent.text),
                }
                edu.append(entry)
        return edu

    def _extract_personal_info(self, doc: Doc) -> Dict[str, str]:
        """Extract name, email, phone, etc."""
        info: Dict[str, str] = {}
        emails = self.email_pattern.findall(doc.text)
        if emails:
            info["email"] = emails[0]
        phones = self.phone_pattern.findall(doc.text)
        if phones:
            info["phone"] = phones[0][0]
        first_line = doc.text.splitlines()[0].strip()
        if (
            first_line
            and not any(ch.isdigit() for ch in first_line)
            and len(first_line.split()) <= 4
        ):
            info["name"] = first_line
        return info

    def _extract_certifications(self, doc: Doc) -> List[str]:
        """Extract sentences containing certification keywords."""
        certs = []
        for sent in doc.sents:
            if any(
                k in sent.text.lower()
                for k in ["certified", "certification", "certificate"]
            ):
                certs.append(sent.text.strip())
        return certs

    def _extract_projects(self, doc: Doc) -> List[Dict[str, Any]]:
        """Extract project names and descriptions."""
        projs = []
        current: Dict[str, str] = {}
        for sent in doc.sents:
            t = sent.text.lower()
            if "project:" in t:
                current["name"] = sent.text.split(":", 1)[1].strip()
            elif "description:" in t and current.get("name"):
                current["description"] = sent.text.split(":", 1)[1].strip()
                projs.append(current)
                current = {}
        return projs

    def _extract_languages(self, doc: Doc) -> List[str]:
        """Extract known languages from a “Languages:” line."""
        langs = []
        keys = ["english", "spanish", "french", "german", "hindi", "mandarin"]
        for sent in doc.sents:
            txt = sent.text.lower()
            if "languages:" in txt:
                for k in keys:
                    if k in txt:
                        langs.append(k.capitalize())
        return langs

    def _extract_institution(self, text: str) -> str:
        """Helper to split out institution names."""
        if " at " in text:
            return text.split(" at ", 1)[1].split(",")[0].strip()
        if " from " in text:
            return text.split(" from ", 1)[1].split(",")[0].strip()
        return ""

    def _extract_year(self, text: str) -> str:
        """Helper to find a 4‑digit year."""
        m = re.search(r"\b(19|20)\d{2}\b", text)
        return m.group(0) if m else ""
