import logging
import traceback
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import re
from dateutil.parser import parse
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline,
)
import torch
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from ..document_processing.sanitizer import ContentSanitizer
from .models import JobPosting, JobConstants
from app.utils.logger import logger
from ..utils import safe_lower


class JobParser:
    """
    Enhanced parser that uses modern NLP models to extract structured information
    from job postings with higher accuracy.
    """

    def __init__(self, use_gpu=False):
        self.sanitizer = ContentSanitizer()
        self.constants = JobConstants()
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        # Initialize NLP components
        self._init_nlp_models()

    def _init_nlp_models(self):
        """Initialize all NLP models used for parsing"""
        try:
            # Load spaCy model for general NLP tasks
            self.nlp = spacy.load("en_core_web_lg")

            # Sentence transformer for semantic understanding
            self.sentence_model = SentenceTransformer(
                "paraphrase-MiniLM-L6-v2", device=self.device
            )

            # Named entity recognition for skill extraction
            self.ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
            self.ner_model = AutoModelForTokenClassification.from_pretrained(
                "dslim/bert-base-NER"
            )
            self.ner_pipeline = pipeline(
                "ner",
                model=self.ner_model,
                tokenizer=self.ner_tokenizer,
                device=0 if self.device == "cuda" else -1,
            )

            # Text classification for section identification
            self.text_classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if self.device == "cuda" else -1,
            )

            # TF-IDF for keyword extraction
            self.tfidf = TfidfVectorizer(
                max_features=1000, stop_words="english", ngram_range=(1, 2)
            )

            # Load skill classification model (specialized for technical skills)
            self.skill_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", device=self.device
            )

            logger.info("Successfully loaded all NLP models")
        except Exception as e:
            logger.error(
                "Failed to load NLP models",
                error=str(e),
                exc_info=True,
                models=[
                    {
                        "name": "spaCy",
                        "status": "failed",
                        "config": {"model": "en_core_web_lg"},
                    },
                    {
                        "name": "SentenceTransformer",
                        "status": "failed",
                        "config": {"model": "paraphrase-MiniLM-L6-v2"},
                    },
                    {
                        "name": "NER",
                        "status": "failed",
                        "config": {
                            "model": "dslim/bert-base-NER",
                            "device": self.device,
                        },
                    },
                    {
                        "name": "TextClassifier",
                        "status": "failed",
                        "config": {
                            "model": "distilbert-base-uncased-finetuned-sst-2-english",
                            "device": self.device,
                        },
                    },
                ],
                system_info={
                    "device": self.device,
                    "cuda_available": torch.cuda.is_available(),
                    "gpu_enabled": self.device == "cuda",
                },
            )
            # Gracefully degrade to basic functionality
            self.nlp = None
            self.sentence_model = None
            raise RuntimeError(f"NLP model initialization failed: {str(e)}")

    def parse_job(self, job_data: Dict[str, Any]) -> JobPosting:
        """
        Parse a single job posting from dictionary format to JobPosting object
        using enhanced NLP techniques.

        Args:
            job_data: Dictionary containing job posting data

        Returns:
            JobPosting object with structured data

        Raises:
            ValueError: If required fields are missing or invalid
        """
        try:

            if not isinstance(job_data, dict):
                raise ValueError("job_data must be a dictionary")

            # Validate required fields
            if not all(
                k in job_data
                for k in ["job_title", "company_name", "description", "apply_link"]
            ):
                raise ValueError("Missing required job fields")

            # Generate a unique job ID if not provided
            job_id = job_data.get("job_id", self._generate_job_id(job_data))

            # Parse and sanitize all fields
            job_title = self.sanitizer.sanitize_text(str(job_data.get("job_title", "")))
            company_name = self.sanitizer.sanitize_text(
                str(job_data.get("company_name", ""))
            )
            description = self.sanitizer.sanitize_text(
                str(job_data.get("description", ""))
            )
            apply_link = self.sanitizer.sanitize_text(
                str(job_data.get("apply_link", ""))
            )

            # Parse job type with NLP enhanced normalization
            job_type = self._parse_job_type_nlp(
                job_data.get("job_type", ""), description
            )

            # Parse salary with enhanced pattern matching
            # First check for salary in normalized_features
            normalized_features = job_data.get("normalized_features", {})
            salary_str = (
                normalized_features.get("salary") if normalized_features else None
            )

            # If not in normalized_features, check main job_data
            if not salary_str:
                salary_str = job_data.get("salary")

            if salary_str:
                salary = self._parse_salary_nlp(salary_str, job_data["description"])
            else:
                # Try to extract from description as fallback
                salary = self._parse_salary_nlp(None, job_data["description"])

            # Parse experience with NLP context
            experience = self._parse_experience_nlp(
                job_data.get("experience"), description
            )

            # Parse location with NLP entity recognition
            location = self._parse_location_nlp(
                job_data.get("location", ""), description
            )

            # Parse posting date
            posting_date = self._parse_date(job_data.get("posting_date"))

            # Parse description sections with advanced NLP
            desc_analysis = self._analyze_description_nlp(description, job_title)

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
                normalized_features=self._create_normalized_features_nlp(
                    job_title, company_name, description, desc_analysis
                ),
            )

        except Exception as e:
            logger.error(
                "Failed to parse job",
                error=str(e),
                exc_info=True,
                input_data={
                    "type": type(job_data).__name__,
                    "keys": (
                        list(job_data.keys()) if isinstance(job_data, dict) else None
                    ),
                    "length": (
                        len(str(job_data)) if hasattr(job_data, "__len__") else None
                    ),
                },
                validation={
                    "required_fields": [
                        "job_title",
                        "company_name",
                        "description",
                        "apply_link",
                    ],
                    "missing_fields": (
                        [
                            field
                            for field in [
                                "job_title",
                                "company_name",
                                "description",
                                "apply_link",
                            ]
                            if field not in job_data
                        ]
                        if isinstance(job_data, dict)
                        else None
                    ),
                },
            )
            raise ValueError(f"Job parsing failed: {str(e)}") from e

    def _generate_job_id(self, job_data: Dict[str, Any]) -> str:
        """Generate a unique job ID from job data"""
        from hashlib import md5

        unique_str = f"{job_data['job_title']}_{job_data['company_name']}_{job_data.get('posting_date', '')}"
        return md5(unique_str.encode("utf-8")).hexdigest()

    def _parse_job_type_nlp(self, job_type_str: str, description: str) -> str:
        """
        Enhanced job type parsing using NLP context from both job_type and description
        """

        if isinstance(job_type_str, (int, float)):
            job_type_str = str(job_type_str)
        elif job_type_str is None:
            job_type_str = ""

        if not job_type_str:
            # Try to extract job type from description
            job_type_keywords = {
                "full time": ["full time", "full-time", "permanent", "regular"],
                "part time": ["part time", "part-time", "hourly"],
                "contract": ["contract", "temporary", "interim", "fixed term"],
                "freelance": ["freelance", "self-employed", "independent contractor"],
                "internship": ["internship", "intern", "trainee", "apprentice"],
                "remote": ["remote", "work from home", "wfh", "virtual", "telecommute"],
            }

            desc_lower = safe_lower(description)
            for job_type, keywords in job_type_keywords.items():
                if any(keyword in desc_lower for keyword in keywords):
                    return job_type

            # Default to full time if nothing found
            return "full time"

        # Process provided job type string
        lower_type = safe_lower(job_type_str)
        for normalized_type, variants in self.constants.JOB_TYPES.items():
            if any(v in lower_type for v in variants):
                return normalized_type

        # Use NLP embedding to find closest match if no direct match
        if self.sentence_model:
            # Map section embeddings to categories
            job_types = list(self.constants.JOB_TYPES.keys())
            job_type_embeddings = self.sentence_model.encode(job_types)
            input_embedding = self.sentence_model.encode([lower_type])

            similarities = np.dot(job_type_embeddings, input_embedding.T).flatten()
            best_match_idx = np.argmax(similarities)

            if similarities[best_match_idx] > 0.7:  # Threshold for confidence
                return job_types[best_match_idx]

        return "other"

    def _parse_salary_nlp(
        self, salary_str: Optional[Union[str, int, float]], description: str
    ) -> Optional[Dict[str, Any]]:
        """
        Enhanced salary parsing with better pattern matching and validation
        """
        try:

            # Handle numeric input (already normalized)
            if isinstance(salary_str, (int, float)):
                result = {
                    "min": float(salary_str),
                    "max": float(salary_str),
                    "currency": "INR",
                    "period": "year",
                }
                return result

            def convert_to_float(value):
                if isinstance(value, (int, float)):
                    result = float(value)
                elif isinstance(value, str):
                    # Remove all non-numeric characters except decimal point
                    clean_value = re.sub(r"[^\d.]", "", value)
                    result = float(clean_value) if clean_value else 0.0
                else:
                    result = 0.0
                return result

            def validate_salary_range(min_val, max_val):
                try:
                    min_float = convert_to_float(min_val)
                    max_float = convert_to_float(max_val)
                    # Ensure min is not greater than max
                    if min_float and max_float and min_float > max_float:
                        min_float, max_float = max_float, min_float
                    return min_float, max_float
                except (ValueError, TypeError):
                    return None, None

            def create_salary_dict(
                min_val: float, max_val: float, multiplier: float = 1.0
            ) -> Dict[str, Any]:
                try:
                    min_salary = int(float(min_val) * multiplier)
                    max_salary = int(float(max_val) * multiplier)

                    # Ensure values are numeric before comparison
                    min_value = float(min_salary)
                    max_value = float(max_salary)

                    # Validate reasonable salary range (1000 to 100M INR)
                    if (
                        1000 <= min_value <= 100000000
                        and 1000 <= max_value <= 100000000
                    ):
                        return {
                            "min": min_salary,
                            "max": max_salary,
                            "currency": "INR",
                            "period": "year",
                        }
                    return None
                except (ValueError, TypeError):
                    return None

            # Handle direct dictionary input
            if isinstance(salary_str, dict):
                min_val = salary_str.get("min")
                max_val = salary_str.get("max")
                if min_val is not None:
                    min_float, max_float = validate_salary_range(
                        min_val, max_val or min_val
                    )
                    if min_float is not None:
                        return create_salary_dict(min_float, max_float or min_float)

            # Handle string input
            if isinstance(salary_str, (str, int, float)):
                salary_str_str = safe_lower(str(salary_str).strip())
                if salary_str_str != "none":
                    # Handle LPA format (e.g., "20-30 LPA", "5-10 LPA")
                    patterns = [
                        # LPA range pattern
                        (r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*LPA", 100000),
                        # Single LPA value
                        (r"(\d+(?:\.\d+)?)\s*LPA", 100000),
                        # Lakhs range pattern
                        (
                            r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*L(?:akh?s?)?",
                            100000,
                        ),
                        # INR range with K/M/Cr multiplier
                        (
                            r"(?:₹|Rs\.?|INR)?\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*([KMCr])",
                            {"K": 1000, "M": 1000000, "Cr": 10000000},
                        ),
                        # Basic range pattern
                        (
                            r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)",
                            100000,
                        ),  # Assuming LPA if no unit
                    ]

                    for pattern, multiplier in patterns:
                        matches = re.search(pattern, salary_str_str, re.IGNORECASE)
                        if matches:
                            if isinstance(multiplier, dict):
                                # Handle K/M/Cr multiplier
                                mult = multiplier.get(matches.group(3).upper(), 1)
                                min_val, max_val = validate_salary_range(
                                    matches.group(1), matches.group(2)
                                )
                                if min_val is not None:
                                    return create_salary_dict(min_val, max_val, mult)
                            else:
                                # Handle standard multiplier
                                if len(matches.groups()) == 1:
                                    # Single value
                                    value = convert_to_float(matches.group(1))
                                    return create_salary_dict(value, value, multiplier)
                                else:
                                    # Range values
                                    min_val, max_val = validate_salary_range(
                                        matches.group(1), matches.group(2)
                                    )
                                    if min_val is not None:
                                        return create_salary_dict(
                                            min_val, max_val, multiplier
                                        )

            # Try to extract from description as last resort
            if description:
                salary_patterns = [
                    r"(?:salary|compensation|package|ctc)(?:\s+range)?\s*:?\s*(?:₹|Rs\.?|INR)?\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(?:L|LPA|Lakhs?)",
                    r"(?:₹|Rs\.?|INR)\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(?:L|LPA|Lakhs?)",
                    r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(?:L|LPA|Lakhs?)",
                ]

                for pattern in salary_patterns:
                    matches = re.findall(pattern, description, re.IGNORECASE)
                    if matches:
                        min_val, max_val = validate_salary_range(
                            matches[0][0], matches[0][1]
                        )
                        if min_val is not None:
                            return create_salary_dict(min_val, max_val, 100000)

            return None

        except Exception as e:
            logger.error(
                "Failed to parse salary",
                error=str(e),
                exc_info=True,
                salary_input={
                    "value": repr(salary_str),
                    "type": type(salary_str).__name__,
                },
                description={
                    "length": len(description) if description else 0,
                    "type": type(description).__name__,
                },
                context={
                    "job_data_keys": list(locals().get("job_data", {}).keys()),
                    "normalized_features_keys": list(
                        locals().get("normalized_features", {}).keys()
                    ),
                },
            )
            return None

    def _normalize_salary_value(self, value: str) -> int:
        """
        Normalize salary values accounting for different formats
        """
        if not value:
            return 0

        value = safe_lower(str(value)).replace(",", "")
        multiplier = 1

        if "k" in value:
            multiplier = 1000
            value = value.replace("k", "")
        elif "m" in value:
            multiplier = 1000000
            value = value.replace("m", "")
        elif any(x in value for x in ["l", "lakh", "lakhs"]):
            multiplier = 100000
            value = re.sub(r"l|lakh|lakhs", "", value)

        return int(float(value) * multiplier)

    def _parse_numeric_value(self, value_str: str) -> int:
        """Parse a numeric value from string, handling K, M, L abbreviations"""
        if not value_str:
            return 0

        value_str = safe_lower(str(value_str).replace(",", ""))
        # Handle multipliers
        if "k" in value_str:
            return int(float(value_str.replace("k", "")) * 1000)
        elif "m" in value_str:
            return int(float(value_str.replace("m", "")) * 1000000)
        elif "l" in value_str or "lakh" in value_str:
            return int(float(re.sub(r"l|lakh", "", value_str)) * 100000)
        else:
            return int(float(value_str))

    def _normalize_salary_values(
        self, min_val: str, max_val: str, context: str
    ) -> Dict[str, Any]:
        """Normalize extracted salary values with currency and period detection"""
        min_num = self._parse_numeric_value(min_val)
        max_num = self._parse_numeric_value(max_val) if max_val else min_num

        # Determine currency from context
        currency = "USD"
        ctx = safe_lower(str(context))
        if any(sym in ctx for sym in ["₹", "rs", "inr", "rupee"]):
            currency = "INR"
        elif "€" in ctx:
            currency = "EUR"
        elif "£" in ctx:
            currency = "GBP"

        # Determine period from context
        period = "year"
        if "hour" in ctx or "/hr" in ctx:
            period = "hour"
        elif "month" in ctx or "/mo" in ctx:
            period = "month"

        return {"min": min_num, "max": max_num, "currency": currency, "period": period}

    def _parse_experience_nlp(
        self, exp_str: Optional[str], description: str
    ) -> Optional[Dict[str, Any]]:
        """
        Enhanced experience parsing using NLP and contextual extraction
        """
        result = None

        # Handle numeric experience values
        if isinstance(exp_str, (int, float)):
            return {"min": int(exp_str), "type": "exact"}
        elif isinstance(exp_str, dict):
            return exp_str  # Return as-is if already formatted

        # Convert to string if needed and apply safe_lower
        exp_str_lower = safe_lower(str(exp_str) if exp_str is not None else "")

        # Try to extract from provided experience string
        if exp_str_lower:
            # Check for years of experience patterns
            range_match = re.search(
                r"(\d+)(?:\s*-\s*|\s+to\s+)(\d+)\s*(?:years|yrs)", exp_str_lower
            )
            if range_match:
                return {
                    "min": int(range_match.group(1)),
                    "max": int(range_match.group(2)),
                    "type": "range",
                }

            min_match = re.search(
                r"(?:minimum|min|\+|at least)\s*(\d+)\s*(?:years|yrs)", exp_str_lower
            )
            if min_match:
                return {"min": int(min_match.group(1)), "type": "min"}

            exact_match = re.search(r"^(\d+)\s*(?:years|yrs)$", exp_str_lower)
            if exact_match:
                years = int(exact_match.group(1))
                return {"min": years, "max": years, "type": "exact"}

            # Check for experience level descriptors
            if any(
                term in exp_str_lower
                for term in ["entry", "junior", "fresher", "graduate", "no experience"]
            ):
                return {"min": 0, "max": 2, "type": "entry"}

            if any(term in exp_str_lower for term in ["mid", "intermediate"]):
                return {"min": 2, "max": 5, "type": "mid"}

            if any(term in exp_str_lower for term in ["senior", "experienced", "lead"]):
                return {"min": 5, "type": "senior"}
            range_match = re.search(
                r"(\d+)(?:\s*-\s*|\s+to\s+)(\d+)\s*(?:years|yrs)", exp_str_lower
            )
            if range_match:
                return {
                    "min": int(range_match.group(1)),
                    "max": int(range_match.group(2)),
                    "type": "range",
                }

            min_match = re.search(
                r"(?:minimum|min|\+|at least)\s*(\d+)\s*(?:years|yrs)", exp_str_lower
            )
            if min_match:
                return {"min": int(min_match.group(1)), "type": "min"}

            exact_match = re.search(r"^(\d+)\s*(?:years|yrs)$", exp_str_lower)
            if exact_match:
                years = int(exact_match.group(1))
                return {"min": years, "max": years, "type": "exact"}

            # Check for experience level descriptors
            if any(
                term in exp_str_lower
                for term in ["entry", "junior", "fresher", "graduate", "no experience"]
            ):
                return {"min": 0, "max": 2, "type": "entry"}

            if any(term in exp_str_lower for term in ["mid", "intermediate"]):
                return {"min": 2, "max": 5, "type": "mid"}

            if any(term in exp_str_lower for term in ["senior", "experienced", "lead"]):
                return {"min": 5, "type": "senior"}

        # If we couldn't extract from exp_str, try the description
        if not result:
            desc_lower = safe_lower(description)

            # Search for experience requirements in description
            exp_patterns = [
                r"(\d+)(?:\s*-\s*|\s+to\s+)(\d+)\s*(?:years|yrs)(?:\s+of\s+experience)?",
                r"(?:minimum|min|\+|at least)\s*(\d+)\s*(?:years|yrs)(?:\s+of\s+experience)?",
                r"(\d+)\s*(?:years|yrs)(?:\s+of\s+experience)",
            ]

            for pattern in exp_patterns:
                matches = re.findall(pattern, desc_lower)
                if matches:
                    if len(matches[0]) == 2:  # Range pattern
                        return {
                            "min": int(matches[0][0]),
                            "max": int(matches[0][1]),
                            "type": "range",
                        }
                    else:  # Single value pattern
                        years = int(matches[0])
                        return {"min": years, "type": "min"}

            # Check for experience level indicators
            if re.search(
                r"entry[- ]level|junior|fresher|graduate|no experience required",
                desc_lower,
            ):
                return {"min": 0, "max": 2, "type": "entry"}

            if re.search(r"mid[- ]level|intermediate", desc_lower):
                return {"min": 2, "max": 5, "type": "mid"}

            if re.search(r"senior|experienced|lead", desc_lower):
                return {"min": 5, "type": "senior"}

        # Default case when nothing is found
        if exp_str:
            return {"raw": exp_str}
        return None

    def _parse_location_nlp(self, location_str: str, description: str) -> str:
        """
        Parse location using NLP entity recognition from both location field
        and job description
        """
        # First try the provided location string
        if location_str:
            if isinstance(location_str, (int, float)):
                location_str = str(location_str or "")
            location_str = location_str.strip()

            # Check for remote indicators
            loc_str_lower = safe_lower(location_str)
            if any(
                remote_word in loc_str_lower
                for remote_word in self.constants.JOB_TYPES["remote"]
            ):
                return "remote"

            # Try to normalize the location
            return location_str

        # Try to extract from description if no location string provided
        # Check for remote work indicators
        if re.search(
            r"(remote|work from home|wfh|virtual|telecommute|anywhere)",
            safe_lower(description),  # Use safe_lower
        ):
            return "remote"

        # Use NER to extract location entities
        if self.nlp:
            doc = self.nlp(description)
            locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]

            if locations:
                # Return the most frequently mentioned location
                location_counts = {}
                for loc in locations:
                    location_counts[loc] = location_counts.get(loc, 0) + 1

                return max(location_counts.items(), key=lambda x: x[1])[0]

        # Default to remote if we can't determine location
        return "remote"

    def _parse_date(self, date_str: Optional[str]) -> datetime:
        """Parse posting date with fallback to current date"""
        if not date_str:
            return datetime.now()
        try:
            return parse(date_str)
        except (ValueError, TypeError):
            return datetime.now()

    def _analyze_description_nlp(
        self, description: str, job_title: str
    ) -> Dict[str, List[str]]:
        """
        Analyze job description using advanced NLP techniques to extract:
        - Skills
        - Requirements
        - Responsibilities
        - Benefits
        - Qualifications
        """
        result = {
            "skills": [],
            "requirements": [],
            "responsibilities": [],
            "benefits": [],
            "qualifications": [],
        }

        # First use standard section splitting
        sections = self._split_description_sections_nlp(description)

        # Process each section with specialized NLP techniques
        for section_name, section_text in sections.items():
            section_name_lower = safe_lower(section_name)  # Use safe_lower

            # Classify text sections
            if self.sentence_model:
                # Map section embeddings to categories
                section_embedding = self.sentence_model.encode(section_name_lower)
                category_names = [
                    "requirements",
                    "qualifications",
                    "responsibilities",
                    "benefits",
                    "skills",
                    "about",
                    "company",
                ]
                category_embeddings = self.sentence_model.encode(category_names)

                similarities = np.dot(category_embeddings, section_embedding)
                best_category_idx = np.argmax(similarities)
                best_category = category_names[best_category_idx]

                # Only use similarity if above threshold
                if similarities[best_category_idx] > 0.65:
                    section_name_lower = best_category

            # Extract content based on section type
            if (
                "requirement" in section_name_lower
                or "qualification" in section_name_lower
            ):
                result["requirements"].extend(
                    self._extract_requirements_nlp(section_text)
                )
                result["qualifications"].extend(
                    self._extract_qualifications_nlp(section_text)
                )
            elif "responsibilit" in section_name_lower or "role" in section_name_lower:
                result["responsibilities"].extend(
                    self._extract_responsibilities_nlp(section_text)
                )
            elif (
                "benefit" in section_name_lower
                or "perk" in section_name_lower
                or "offer" in section_name_lower
            ):
                result["benefits"].extend(self._extract_benefits_nlp(section_text))
            elif "skill" in section_name_lower or "technolog" in section_name_lower:
                result["skills"].extend(
                    self._extract_skills_nlp(section_text, job_title)
                )
            else:
                # Extract from general text
                result["skills"].extend(
                    self._extract_skills_nlp(section_text, job_title)
                )

                # Try to find requirements in unsectioned text
                if not result["requirements"]:
                    result["requirements"].extend(
                        self._extract_requirements_nlp(section_text)
                    )

        # Additional skills extraction from whole description
        if not result["skills"]:
            result["skills"] = self._extract_skills_nlp(description, job_title)

        # Ensure we have at least some qualifications
        if not result["qualifications"] and result["requirements"]:
            result["qualifications"] = self._filter_education_requirements(
                result["requirements"]
            )

        # Deduplicate and clean
        for key in result:
            result[key] = [
                safe_lower(str(item)).strip()
                for item in set(result[key])
                if str(item).strip()
            ]

        return result

    def _split_description_sections_nlp(self, description: str) -> Dict[str, str]:
        """
        Enhanced section splitting using NLP and pattern recognition
        """
        sections = {}

        # Replace HTML breaks with newlines for better parsing
        description = re.sub(r"<br\s*/?>", "\n", description)

        # Split by common section headers using more flexible patterns
        section_patterns = [
            # Headers with colon at end
            r"^([A-Z][A-Za-z\s&\/\-]+):[\s]*$",
            # Headers with colon in middle
            r"^([A-Z][A-Za-z\s&\/\-]+):(.*?)$",
            # Headers in all caps
            r"^([A-Z][A-Z\s&\/\-]+)[\s]*$",
            # Headers with numbering
            r"^(?:\d+\.|\•|\*)\s*([A-Z][A-Za-z\s&\/\-]+)(?::|-)[\s]*$",
            # Headers with formatting indicators like ** or __
            r"^(?:\*\*|__)([A-Z][A-Za-z\s&\/\-]+)(?:\*\*|__)[\s]*$",
        ]

        # First try to identify all section headers
        lines = description.split("\n")
        section_indices = []
        section_names = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Try each pattern to find headers
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    section_name = match.group(1).strip()
                    section_indices.append(i)
                    section_names.append(section_name)
                    break

        # If we found sections, extract content between them
        if section_indices:
            for i in range(len(section_indices)):
                start_idx = section_indices[i]
                end_idx = (
                    section_indices[i + 1]
                    if i + 1 < len(section_indices)
                    else len(lines)
                )

                section_content = "\n".join(lines[start_idx + 1 : end_idx]).strip()
                if section_content:
                    sections[section_names[i]] = section_content
        else:
            # Fallback: If no sections found, use the whole description
            sections["description"] = description

            # Try to split by bullet points if available
            if "\n•" in description or "\n-" in description or "\n*" in description:
                bullet_splits = re.split(r"\n(?:•|-|\*)\s*", description)
                if len(bullet_splits) > 1:
                    sections["Requirements"] = "\n".join(
                        [
                            f"• {item.strip()}"
                            for item in bullet_splits[1:]
                            if item.strip()
                        ]
                    )

        return sections

    def _init_skill_categories(self):
        """
        Initialize comprehensive skill categories
        """
        self.skill_categories = {
            "soft_skills": [
                "communication",
                "teamwork",
                "leadership",
                "problem-solving",
                "time-management",
                "adaptability",
                "creativity",
                "flexibility",
                "self-motivation",
                "attention-to-detail",
                "critical-thinking",
                "decision-making",
                "collaboration",
            ],
            "hard_skills": [
                "data-analysis",
                "data-visualization",
                "machine-learning",
                "deep-learning",
                "data-science",
                "statistics",
                "predictive-modeling",
                "regression",
                "classification",
                "clustering",
            ],
            "fundamental_skills": [
                "algorithms",
                "data structures",
                "object oriented programming",
                "design-patterns",
                "version-control",
                "testing",
                "debugging",
                "continuous-integration",
                "continuous-delivery",
                "version-control",
                "git",
                "github",
                "bitbucket",
            ],
            "programming_languages": [
                "python",
                "java",
                "javascript",
                "typescript",
                "c++",
                "c#",
                "ruby",
                "php",
                "swift",
                "kotlin",
                "go",
                "rust",
                "scala",
                "perl",
                "r",
                "matlab",
                "bash",
                "shell",
                "powershell",
                "vba",
                "objective-c",
                "dart",
                "lua",
                "groovy",
            ],
            "frameworks": [
                "django",
                "flask",
                "fastapi",
                "pyramid",
                "spring",
                "springboot",
                "express",
                "next.js",
                "nuxt.js",
                "rails",
                "laravel",
                "symfony",
                "cakephp",
                "zend",
                "asp.net",
                "asp.net core",
                "vue",
                "angular",
                "react",
                "svelte",
                "ember",
                "backbone",
                "meteor",
                "gatsby",
                "phoenix",
                "hapi",
                "koa",
                "nestjs",
                "quasar",
                "ionic",
                "cordova",
                "electron",
                "unity",
                "unreal engine",
                "flutter",
                "xamarin",
                "qt",
                "wxwidgets",
                "bootstrap",
                "tailwind",
                "material-ui",
                "chakra-ui",
                "primefaces",
                "struts",
                "jsf",
                "play framework",
                "micronaut",
                "dropwizard",
            ],
            "web_technologies": [
                "html",
                "css",
                "sass",
                "less",
                "bootstrap",
                "tailwind",
                "jquery",
                "react",
                "angular",
                "vue",
                "svelte",
                "next.js",
                "nuxt.js",
                "webpack",
                "babel",
                "typescript",
                "redux",
                "graphql",
                "rest api",
                "soap",
                "xml",
                "json",
            ],
            "databases": [
                "sql",
                "mysql",
                "postgresql",
                "mongodb",
                "oracle",
                "redis",
                "elasticsearch",
                "dynamodb",
                "cassandra",
                "sqlite",
                "mariadb",
                "neo4j",
                "couchdb",
                "firebase",
                "supabase",
            ],
            "cloud_platforms": [
                "aws",
                "azure",
                "gcp",
                "digitalocean",
                "heroku",
                "netlify",
                "vercel",
                "cloudflare",
                "alibaba cloud",
                "ibm cloud",
                "oracle cloud",
            ],
            "devops_tools": [
                "git",
                "docker",
                "kubernetes",
                "jenkins",
                "travis ci",
                "circle ci",
                "terraform",
                "ansible",
                "puppet",
                "chef",
                "prometheus",
                "grafana",
                "elk stack",
                "nginx",
                "apache",
            ],
            "testing": [
                "selenium",
                "junit",
                "testng",
                "cypress",
                "jest",
                "mocha",
                "chai",
                "puppeteer",
                "postman",
                "soapui",
                "jmeter",
                "gatling",
                "cucumber",
                "protractor",
                "karma",
                "pytest",
                "phpunit",
                "rspec",
                "testcafe",
            ],
            "project_management": [
                "jira",
                "trello",
                "asana",
                "basecamp",
                "monday.com",
                "clickup",
                "confluence",
                "notion",
                "microsoft project",
                "smartsheet",
                "agile",
                "scrum",
                "kanban",
                "waterfall",
                "prince2",
            ],
            "design_tools": [
                "figma",
                "sketch",
                "adobe xd",
                "photoshop",
                "illustrator",
                "indesign",
                "after effects",
                "premiere pro",
                "invision",
                "zeplin",
                "principle",
            ],
            "soft_skills": [
                "communication",
                "leadership",
                "teamwork",
                "problem solving",
                "critical thinking",
                "time management",
                "adaptability",
                "creativity",
                "collaboration",
                "presentation",
                "negotiation",
                "conflict resolution",
            ],
            "machine_learning": [
                "tensorflow",
                "pytorch",
                "scikit-learn",
                "keras",
                "opencv",
                "pandas",
                "numpy",
                "scipy",
                "matplotlib",
                "seaborn",
                "nltk",
                "spacy",
                "hugging face",
            ],
            "mobile_development": [
                "android",
                "ios",
                "react native",
                "flutter",
                "xamarin",
                "ionic",
                "swift",
                "kotlin",
                "objective-c",
                "mobile testing",
                "responsive design",
            ],
            "security": [
                "cybersecurity",
                "encryption",
                "oauth",
                "jwt",
                "authentication",
                "authorization",
                "penetration testing",
                "security testing",
                "owasp",
                "ssl/tls",
                "firewall",
                "vpn",
            ],
        }

    def _extract_skills_nlp(self, text: str, job_title: str) -> List[str]:
        """
        Enhanced skill extraction with better handling of all skill categories
        """
        extracted_skills = set()
        text_lower = safe_lower(text)  # Use safe_lower at the start
        # text_lower = text.lower() # Remove redundant original call

        # Initialize skill categories if not already done
        if not hasattr(self, "skill_categories"):
            self._init_skill_categories()

        # Preprocess text: replace hyphens and underscores with spaces
        text_normalized = text_lower.replace("-", " ").replace("_", " ")

        # Helper function to normalize skill names
        def normalize_skill(skill: Any) -> str:
            # Ensure skill is a string before processing
            skill_str = str(skill) if skill is not None else ""
            return skill_str.replace("-", " ").replace("_", " ")

        # Helper function to check if a skill is present in text
        def find_skill_in_text(skill: str, text: str) -> bool:
            skill_normalized = normalize_skill(skill)
            # Check for exact match with word boundaries
            if re.search(r"\b" + re.escape(skill_normalized) + r"\b", text):
                return True
            # Check for variations (e.g., "data-structure" vs "data structure")
            skill_parts = skill_normalized.split()
            if len(skill_parts) > 1:
                # Check for parts appearing close to each other
                parts_pattern = (
                    r"\b" + r"\W+\w*\W+".join(map(re.escape, skill_parts)) + r"\b"
                )
                if re.search(parts_pattern, text, re.I):
                    return True
            return False

        # Extract skills from specific sections
        skill_sections = re.findall(
            r"(?:required skills|key skills|technical skills|requirements|qualifications|what you'll need|what you need|what we're looking for)[:|-](.*?)(?:\n\n|\Z)",
            text,
            re.I | re.S,
        )

        # Process each skill section
        for section in skill_sections:
            # Extract bullet points and lines
            skills = re.findall(r"(?:•|-|\*|\d+\.)\s*(.*?)(?:\n|$)", section)
            for skill_text in skills:
                skill_text_str = str(skill_text)
                skill_text_lower = safe_lower(skill_text.strip())  # Use safe_lower
                # Check each category and their skills
                for category, category_skills in self.skill_categories.items():
                    for category_skill in category_skills:
                        if find_skill_in_text(
                            category_skill, skill_text_lower
                        ):  # Use skill_text_lower
                            extracted_skills.add(normalize_skill(category_skill))
                    for category_skill in category_skills:
                        if find_skill_in_text(category_skill, skill_text):
                            extracted_skills.add(normalize_skill(category_skill))

        # Look for skills throughout the entire text
        for category, skills in self.skill_categories.items():
            for skill in skills:
                if find_skill_in_text(skill, text_normalized):
                    extracted_skills.add(normalize_skill(skill))

        # Extract skills from experience requirements
        experience_patterns = [
            r"experience (?:in|with) (.*?)(?:\.|\n|$)",
            r"knowledge of (.*?)(?:\.|\n|$)",
            r"familiarity with (.*?)(?:\.|\n|$)",
            r"proficiency in (.*?)(?:\.|\n|$)",
            r"background in (.*?)(?:\.|\n|$)",
        ]

        for pattern in experience_patterns:
            matches = re.finditer(pattern, text_normalized, re.I)
            for match in matches:
                exp_text = safe_lower(match.group(1))  # Use safe_lower
                for category, skills in self.skill_categories.items():
                    for skill in skills:
                        if find_skill_in_text(skill, exp_text):
                            extracted_skills.add(normalize_skill(skill))

        # Add job-specific skills based on job title
        job_skills = self._get_job_specific_skills(job_title)
        for skill in job_skills:
            if find_skill_in_text(skill, text_normalized):
                extracted_skills.add(normalize_skill(skill))

        # Filter out common false positives and invalid entries
        false_positives = {
            "experience",
            "year",
            "years",
            "knowledge",
            "understanding",
            "ability",
            "skills",
            "proficiency",
            "expertise",
            "background",
            "plus",
            "minimum",
            "maximum",
            "required",
            "preferred",
            "strong",
            "good",
            "excellent",
            "basic",
            "advanced",
            "intermediate",
            "working",
            "hands on",
            "practical",
        }

        # Clean and normalize final skill set
        cleaned_skills = set()
        for skill in extracted_skills:
            if skill not in false_positives:
                # Additional validation for skill quality
                if (
                    len(skill.split()) <= 4 and len(skill) >= 2
                ):  # Reasonable skill length
                    cleaned_skills.add(skill)

        # Sort skills by category for better organization
        categorized_skills = []
        for category, category_skills in self.skill_categories.items():
            category_skills_normalized = {normalize_skill(s) for s in category_skills}
            matching_skills = sorted(
                cleaned_skills.intersection(category_skills_normalized)
            )
            categorized_skills.extend(matching_skills)

        # Add any remaining skills that weren't categorized
        remaining_skills = sorted(cleaned_skills - set(categorized_skills))
        categorized_skills.extend(remaining_skills)

        return categorized_skills

    def _get_job_specific_skills(self, job_title: str) -> List[str]:
        """
        Return likely skills based on the job title
        """
        job_title_lower = safe_lower(job_title)

        # Define skill mappings for common job titles
        job_skills_map = {
            "frontend": [
                "html",
                "css",
                "javascript",
                "typescript",
                "react",
                "angular",
                "vue",
                "webpack",
                "sass",
                "less",
                "responsive design",
                "web accessibility",
                "redux",
                "jest",
                "cypress",
            ],
            "backend": [
                "python",
                "java",
                "nodejs",
                "php",
                "ruby",
                "go",
                "sql",
                "nosql",
                "rest api",
                "graphql",
                "microservices",
                "docker",
                "kubernetes",
            ],
            "fullstack": [
                "javascript",
                "python",
                "java",
                "sql",
                "react",
                "angular",
                "nodejs",
                "rest api",
                "html",
                "css",
                "git",
                "docker",
            ],
            "data": [
                "python",
                "r",
                "sql",
                "pandas",
                "numpy",
                "scikit-learn",
                "tensorflow",
                "pytorch",
                "tableau",
                "power bi",
                "hadoop",
                "spark",
            ],
            "devops": [
                "aws",
                "azure",
                "gcp",
                "docker",
                "kubernetes",
                "jenkins",
                "terraform",
                "ansible",
                "ci/cd",
                "linux",
                "shell scripting",
                "monitoring",
            ],
        }

        relevant_skills = set()

        # Match job title with skill sets
        for job_type, skills in job_skills_map.items():
            if job_type in job_title_lower:
                relevant_skills.update(skills)

        return list(relevant_skills)

    def _extract_requirements_nlp(self, text: str) -> List[str]:
        """
        Extract requirements using NLP analysis
        """
        requirements = []

        # Split text into sentences
        if self.nlp:
            doc = self.nlp(str(text))

            for sent in doc.sents:
                # Look for requirement indicators
                if any(
                    word.lower_ in ["must", "should", "need", "require", "essential"]
                    for word in sent
                ):
                    requirements.append(
                        safe_lower(sent.text).strip()
                    )  # Apply safe_lower before strip

                # Look for bullet points
                elif (
                    safe_lower(sent.text).strip().startswith(("•", "-", "*"))
                ):  # Apply safe_lower before strip
                    requirements.append(
                        safe_lower(sent.text).strip()
                    )  # Apply safe_lower before strip

        # Fallback to regex if NLP fails
        if not requirements:
            # Apply safe_lower within list comprehension for fallback
            requirements = [
                safe_lower(req).strip()
                for req in re.findall(r"(?:•|-|\*)\s*(.*?)(?=\n|$)", text)
                if req.strip()
            ]

        # Ensure final list elements are safe_lower and stripped
        return [
            safe_lower(str(req)).strip() for req in requirements if str(req).strip()
        ]

    def _extract_responsibilities_nlp(self, text: str) -> List[str]:
        """
        Extract job responsibilities using NLP
        """
        responsibilities = []

        if self.nlp:
            doc = self.nlp(str(text))

            for sent in doc.sents:
                # Look for action verbs at start of sentence
                if len(sent) > 0:  # Check if sentence has tokens
                    first_token = sent[0]  # Get first token directly
                    if first_token.lower_ in [  # Use lower_ attribute
                        "develop",
                        "manage",
                        "create",
                        "design",
                        "implement",
                        "maintain",
                        "coordinate",
                        "lead",
                        "build",
                        "analyze",
                    ]:
                        responsibilities.append(
                            safe_lower(sent.text).strip()
                        )  # Apply safe_lower before strip

        # Extract bullet points
        # Apply safe_lower within list comprehension for bullet points
        bullet_points = [
            safe_lower(str(point)).strip()
            for point in re.findall(r"(?:•|-|\*)\s*(.*?)(?=\n|$)", text)
            if str(point).strip()
        ]
        responsibilities.extend(bullet_points)

        # Ensure final list elements are safe_lower and stripped
        return [safe_lower(resp).strip() for resp in responsibilities if resp.strip()]

    def _extract_benefits_nlp(self, text: str) -> List[str]:
        """
        Extract benefits using NLP and pattern matching
        """
        benefits = set()

        # Common benefit keywords
        benefit_keywords = [
            "health insurance",
            "dental",
            "vision",
            "401k",
            "retirement",
            "pto",
            "vacation",
            "remote work",
            "flexible",
            "bonus",
            "stock options",
            "parental leave",
            "education",
            "training",
            "gym",
            "wellness",
        ]

        # Extract using NLP
        if self.nlp:
            doc = self.nlp(str(text))

            for sent in doc.sents:
                sent_lower = safe_lower(sent.text)  # Use safe_lower
                # Check if sentence contains benefit keywords
                if any(keyword in sent_lower for keyword in benefit_keywords):
                    benefits.add(sent.text.strip())

        # Extract bullet points
        bullet_points = re.findall(r"(?:•|-|\*)\s*(.*?)(?=\n|$)", text)
        for point in bullet_points:
            point_lower = safe_lower(point)  # Use safe_lower
            if any(keyword in point_lower for keyword in benefit_keywords):
                benefits.add(point.strip())

        return list(benefits)

    def _extract_qualifications_nlp(self, text: str) -> List[str]:
        """
        Extract qualifications using NLP
        """
        qualifications = []

        # Education keywords
        edu_keywords = [
            "degree",
            "bachelor",
            "master",
            "phd",
            "diploma",
            "certification",
        ]

        if self.nlp:
            doc = self.nlp(str(text))

            for sent in doc.sents:
                sent_lower = safe_lower(sent.text)  # Use safe_lower
                # Look for education requirements
                if any(keyword in sent_lower for keyword in edu_keywords):
                    qualifications.append(sent.text.strip())

        # Extract from bullet points
        bullet_points = re.findall(r"(?:•|-|\*)\s*(.*?)(?=\n|$)", text)
        for point in bullet_points:
            point_lower = safe_lower(point)  # Use safe_lower
            if any(keyword in point_lower for keyword in edu_keywords):
                qualifications.append(point.strip())

        return [qual.strip() for qual in qualifications if qual.strip()]

    def _filter_education_requirements(self, requirements: List[str]) -> List[str]:
        """
        Filter requirements list for education-related items
        """
        edu_keywords = [
            "degree",
            "bachelor",
            "master",
            "phd",
            "diploma",
            "certification",
        ]
        return [
            req
            for req in requirements
            if any(keyword in safe_lower(req) for keyword in edu_keywords)
        ]

    def _create_normalized_features_nlp(
        self,
        job_title: str,
        company_name: str,
        description: str,
        desc_analysis: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """
        Create normalized features for ML matching using NLP
        """
        # Get embeddings if available
        title_embedding = None
        desc_embedding = None
        if self.sentence_model:
            title_embedding = self.sentence_model.encode([job_title])[0].tolist()
            desc_embedding = self.sentence_model.encode([description])[0].tolist()

        return {
            "job_title_keywords": self._extract_keywords_nlp(job_title),
            "company_name_normalized": safe_lower(company_name),
            "description_keywords": self._extract_keywords_nlp(description),
            # Ensure skills_normalized are consistently lowercased
            "skills_normalized": [
                safe_lower(skill) for skill in desc_analysis.get("skills", [])
            ],
            "requirements_count": len(desc_analysis.get("requirements", [])),
            "responsibilities_count": len(desc_analysis.get("responsibilities", [])),
            # Ensure qualification check uses safe_lower consistently
            "has_degree_requirement": any(
                "degree" in safe_lower(qual)
                for qual in desc_analysis.get("qualifications", [])
            ),
            "title_embedding": title_embedding,
            "description_embedding": desc_embedding,
            "seniority_level": self._detect_seniority_level(job_title, description),
            # Ensure technical skill check uses safe_lower consistently
            "technical_skills_count": len(
                [
                    skill
                    for skill in desc_analysis.get("skills", [])  # Use .get for safety
                    # Ensure comparison is robust: check against lowercased constants
                    if safe_lower(skill)
                    in {
                        safe_lower(k)
                        for k in self.constants.SKILL_KEYWORDS.get("technical", [])
                    }
                ]
            ),
        }

    def _extract_keywords_nlp(self, text: str) -> List[str]:
        """
        Extract keywords using NLP and TF-IDF
        """
        if self.nlp:
            doc = self.nlp(str(text))
            keywords = []

            # Extract important tokens
            for token in doc:
                if not token.is_stop and not token.is_punct and len(token.text) > 2:
                    if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
                        keywords.append(safe_lower(token.lemma_))  # Use safe_lower

            return list(set(keywords))

        # Fallback to simple word extraction
        return list(set(re.findall(r"\b[a-zA-Z]{3,}\b", safe_lower(str(text)))))

    def _detect_seniority_level(self, job_title: str, description: str) -> str:
        """
        Detect job seniority level from title and description
        """
        # Use safe_lower for robustness
        text = safe_lower(f"{job_title} {description}")

        if any(
            word in text
            for word in ["senior", "lead", "principal", "architect", "head"]
        ):
            return "senior"
        elif any(word in text for word in ["junior", "entry", "graduate", "trainee"]):
            return "junior"
        else:
            return "mid"
