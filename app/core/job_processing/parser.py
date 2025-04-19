import logging
from typing import Dict, List, Optional, Any
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

logger = logging.getLogger(__name__)


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
            logger.error(f"Failed to load NLP models: {str(e)}")
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

            # Parse job type with NLP enhanced normalization
            job_type = self._parse_job_type_nlp(
                job_data.get("job_type", ""), description
            )

            # Parse salary with enhanced pattern matching
            salary = self._parse_salary_nlp(job_data.get("salary"), description)

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
            logger.error(f"Failed to parse job: {str(e)}")
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

            desc_lower = description.lower()
            for job_type, keywords in job_type_keywords.items():
                if any(keyword in desc_lower for keyword in keywords):
                    return job_type

            # Default to full time if nothing found
            return "full time"

        # Process provided job type string
        lower_type = job_type_str.lower()
        for normalized_type, variants in self.constants.JOB_TYPES.items():
            if any(v in lower_type for v in variants):
                return normalized_type

        # Use NLP embedding to find closest match if no direct match
        if self.sentence_model:
            job_types = list(self.constants.JOB_TYPES.keys())
            job_type_embeddings = self.sentence_model.encode(job_types)
            input_embedding = self.sentence_model.encode([lower_type])

            similarities = np.dot(job_type_embeddings, input_embedding.T).flatten()
            best_match_idx = np.argmax(similarities)

            if similarities[best_match_idx] > 0.7:  # Threshold for confidence
                return job_types[best_match_idx]

        return "other"

    def _parse_salary_nlp(
        self, salary_str: Optional[str], description: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse salary using enhanced pattern matching and NLP extraction from both
        the salary field and job description
        """
        if not salary_str:
            # Try to extract salary from description
            salary_patterns = [
                r"(?:salary|compensation)[:\s]*(?:is|of|up to|range)?[:\s]*[$₹€£]?\s*(\d+[\d,.]*)\s*(?:k|thousand|l|lakh|lakhs|lpa|m|million)?(?:\s*-\s*[$₹€£]?\s*(\d+[\d,.]*)\s*(?:k|thousand|l|lakh|lakhs|lpa|m|million)?)?",
                r"[$₹€£]\s*(\d+[\d,.]*)\s*(?:k|thousand|l|lakh|lakhs|lpa|m|million)?(?:\s*-\s*[$₹€£]?\s*(\d+[\d,.]*)\s*(?:k|thousand|l|lakh|lakhs|lpa|m|million)?)?",
                r"(\d+[\d,.]*)\s*(?:k|thousand|l|lakh|lakhs|lpa|m|million)(?:\s*-\s*(\d+[\d,.]*)\s*(?:k|thousand|l|lakh|lakhs|lpa|m|million)?)?",
            ]

            for pattern in salary_patterns:
                matches = re.findall(pattern, description, re.IGNORECASE)
                if matches:
                    for match in matches:
                        min_val, max_val = match
                        if min_val:
                            return self._normalize_salary_values(
                                min_val, max_val, description
                            )

        if salary_str:
            # Parse provided salary string
            salary_str = salary_str.lower()

            # Check for LPA (Lakh Per Annum) pattern first
            lpa_match = re.search(r"(\d+)(?:\s*-\s*(\d+))?\s*lpa", salary_str)
            if lpa_match:
                min_sal = int(lpa_match.group(1))
                max_sal = int(lpa_match.group(2)) if lpa_match.group(2) else min_sal
                return {
                    "min": min_sal * 100000,
                    "max": max_sal * 100000,
                    "currency": "INR",
                    "period": "year",
                }

            # Check for standard currency patterns
            currency_match = re.search(
                r"(\d+[\d,.]*)(?:\s*-\s*(\d+[\d,.]*))?", salary_str
            )
            if currency_match:
                min_sal = self._parse_numeric_value(currency_match.group(1))
                max_sal = (
                    self._parse_numeric_value(currency_match.group(2))
                    if currency_match.group(2)
                    else min_sal
                )

                # Determine currency
                currency = "USD"
                if (
                    "₹" in salary_str
                    or "rs" in salary_str
                    or "inr" in salary_str
                    or "rupee" in salary_str
                ):
                    currency = "INR"
                elif "€" in salary_str or "eur" in salary_str:
                    currency = "EUR"
                elif "£" in salary_str or "gbp" in salary_str:
                    currency = "GBP"

                # Determine period
                period = "year"
                if "hour" in salary_str or "hr" in salary_str:
                    period = "hour"
                elif "day" in salary_str:
                    period = "day"
                elif "month" in salary_str or "pm" in salary_str:
                    period = "month"

                return {
                    "min": min_sal,
                    "max": max_sal,
                    "currency": currency,
                    "period": period,
                }

        return None

    def _parse_numeric_value(self, value_str: str) -> int:
        """Parse a numeric value from string, handling K, M, L abbreviations"""
        if not value_str:
            return 0

        value_str = value_str.replace(",", "")

        # Handle multipliers
        if "k" in value_str.lower():
            return int(float(value_str.lower().replace("k", "")) * 1000)
        elif "m" in value_str.lower():
            return int(float(value_str.lower().replace("m", "")) * 1000000)
        elif "l" in value_str.lower() or "lakh" in value_str.lower():
            return int(float(re.sub(r"l|lakh", "", value_str.lower())) * 100000)
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
        if any(sym in context for sym in ["₹", "rs", "inr", "rupee"]):
            currency = "INR"
        elif "€" in context:
            currency = "EUR"
        elif "£" in context:
            currency = "GBP"

        # Determine period from context
        period = "year"
        if "hour" in context.lower() or "/hr" in context.lower():
            period = "hour"
        elif "month" in context.lower() or "/mo" in context.lower():
            period = "month"

        return {"min": min_num, "max": max_num, "currency": currency, "period": period}

    def _parse_experience_nlp(
        self, exp_str: Optional[str], description: str
    ) -> Optional[Dict[str, Any]]:
        """
        Enhanced experience parsing using NLP and contextual extraction
        """
        result = None

        # Try to extract from provided experience string
        if exp_str:
            exp_str = exp_str.lower()

            # Check for years of experience patterns
            range_match = re.search(
                r"(\d+)(?:\s*-\s*|\s+to\s+)(\d+)\s*(?:years|yrs)", exp_str
            )
            if range_match:
                return {
                    "min": int(range_match.group(1)),
                    "max": int(range_match.group(2)),
                    "type": "range",
                }

            min_match = re.search(
                r"(?:minimum|min|\+|at least)\s*(\d+)\s*(?:years|yrs)", exp_str
            )
            if min_match:
                return {"min": int(min_match.group(1)), "type": "min"}

            exact_match = re.search(r"^(\d+)\s*(?:years|yrs)$", exp_str)
            if exact_match:
                years = int(exact_match.group(1))
                return {"min": years, "max": years, "type": "exact"}

            # Check for experience level descriptors
            if any(
                term in exp_str
                for term in ["entry", "junior", "fresher", "graduate", "no experience"]
            ):
                return {"min": 0, "max": 2, "type": "entry"}

            if any(term in exp_str for term in ["mid", "intermediate"]):
                return {"min": 2, "max": 5, "type": "mid"}

            if any(term in exp_str for term in ["senior", "experienced", "lead"]):
                return {"min": 5, "type": "senior"}

        # If we couldn't extract from exp_str, try the description
        if not result:
            desc_lower = description.lower()

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
            location_str = location_str.strip()

            # Check for remote indicators
            if any(
                remote_word in location_str.lower()
                for remote_word in self.constants.JOB_TYPES["remote"]
            ):
                return "remote"

            # Try to normalize the location
            return location_str

        # Try to extract from description if no location string provided
        # Check for remote work indicators
        if re.search(
            r"(remote|work from home|wfh|virtual|telecommute|anywhere)",
            description.lower(),
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
            section_name_lower = section_name.lower()

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
            result[key] = list(set(result[key]))

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

    def _extract_skills_nlp(self, text: str, job_title: str) -> List[str]:
        """
        Extract skills using NLP techniques including NER and embeddings
        """
        extracted_skills = set()
        text_lower = text.lower()

        # 1. Use regex to extract skills from bullet points
        bullet_points = re.findall(r"(?:•|-|\*)\s*(.*?)(?=\n|$)", text)

        # 2. Add known technical skills from our constants
        for category, skill_list in self.constants.SKILL_KEYWORDS.items():
            for skill in skill_list:
                if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
                    extracted_skills.add(skill)

        # 3. Use NLP to extract technical terms and skills
        if self.nlp:
            doc = self.nlp(text)

            # Extract noun phrases that could be skills
            for chunk in doc.noun_chunks:
                # Filter for likely skill phrases
                if 2 <= len(chunk.text.split()) <= 4 and not any(
                    word.is_stop for word in chunk
                ):
                    candidate = chunk.text.lower()

                    # Only add if it looks like a technical skill
                    if any(
                        tech_word in candidate
                        for tech_word in [
                            "programming",
                            "language",
                            "framework",
                            "database",
                            "software",
                            "system",
                            "development",
                            "design",
                            "analysis",
                            "management",
                            "java",
                            "python",
                            "c++",
                            "javascript",
                            "react",
                            "angular",
                            "vue",
                            "sql",
                            "nosql",
                            "cloud",
                            "aws",
                            "azure",
                            "ml",
                            "ai",
                            "data",
                        ]
                    ):
                        extracted_skills.add(candidate)

            # Extract named entities that could be frameworks, tools, etc.
            for ent in doc.ents:
                if ent.label_ in ["PRODUCT", "ORG", "WORK_OF_ART"]:
                    # Filter for likely technical products
                    if not any(
                        common in ent.text.lower()
                        for common in [
                            "microsoft",
                            "google",
                            "amazon",
                            "facebook",
                            "apple",
                            "inc",
                            "llc",
                            "ltd",
                        ]
                    ):
                        extracted_skills.add(ent.text.lower())

        # 4. Use the embedding model to extract skills based on similarity to known tech terms
        if self.sentence_model and bullet_points:
            # Create embeddings for common skill categories
            skill_categories = [
                "programming language",
                "framework",
                "software tool",
                "database",
                "methodology",
                "cloud platform",
                "design pattern",
            ]

            category_embeddings = self.sentence_model.encode(skill_categories)

            # Check each bullet point for possible skills
            for point in bullet_points:
                point_embedding = self.sentence_model.encode([point])
                similarities = np.dot(category_embeddings, point_embedding.T).flatten()

                if max(similarities) > 0.6:  # Threshold for skill relevance
                    # Extract key phrases from the bullet point
                    if self.nlp:
                        point_doc = self.nlp(point)
                        for token in point_doc:
                            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                                # Check if this could be a technical term
                                if len(token.text) > 2 and not token.text.lower() in [
                                    "year",
                                    "month",
                                    "day",
                                ]:
                                    extracted_skills.add(token.text.lower())

        # 5. Extract programming languages, frameworks, etc. directly
        tech_patterns = [
            r"\b(java|python|c\+\+|javascript|typescript|php|ruby|swift|kotlin|go|rust|scala|html|css|sql)\b",
            r"\b(react|angular|vue|node|express|django|flask|spring|laravel|rails|tensorflow|pytorch|keras)\b",
            r"\b(aws|azure|gcp|docker|kubernetes|jenkins|git|ci/cd|agile|scrum|jira|confluence)\b",
            r"\b(mysql|postgresql|mongodb|redis|elasticsearch|firebase|dynamodb|cassandra|oracle|sql server)\b",
        ]

        for pattern in tech_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                extracted_skills.add(match.group(1).lower())

        # 6. Use job title context to extract relevant skills
        job_specific_skills = self._get_job_specific_skills(job_title)
        for skill in job_specific_skills:
            if skill.lower() in text_lower:
                extracted_skills.add(skill.lower())

        # Return sorted skills list
        return sorted(list(extracted_skills))

    def _get_job_specific_skills(self, job_title: str) -> List[str]:
        """
        Return likely skills based on the job title
        """
        job_title_lower = job_title.lower()

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
            doc = self.nlp(text)

            for sent in doc.sents:
                # Look for requirement indicators
                if any(
                    word.lower_ in ["must", "should", "need", "require", "essential"]
                    for word in sent
                ):
                    requirements.append(sent.text.strip())

                # Look for bullet points
                elif sent.text.strip().startswith(("•", "-", "*")):
                    requirements.append(sent.text.strip())

        # Fallback to regex if NLP fails
        if not requirements:
            requirements = re.findall(r"(?:•|-|\*)\s*(.*?)(?=\n|$)", text)

        return [req.strip() for req in requirements if req.strip()]

    def _extract_responsibilities_nlp(self, text: str) -> List[str]:
        """
        Extract job responsibilities using NLP
        """
        responsibilities = []

        if self.nlp:
            doc = self.nlp(text)

            for sent in doc.sents:
                # Look for action verbs at start of sentence
                first_word = next(sent.iter_tokens()).lower_
                if first_word in [
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
                    responsibilities.append(sent.text.strip())

        # Extract bullet points
        bullet_points = re.findall(r"(?:•|-|\*)\s*(.*?)(?=\n|$)", text)
        responsibilities.extend(bullet_points)

        return [resp.strip() for resp in responsibilities if resp.strip()]

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
            doc = self.nlp(text)

            for sent in doc.sents:
                # Check if sentence contains benefit keywords
                if any(keyword in sent.text.lower() for keyword in benefit_keywords):
                    benefits.add(sent.text.strip())

        # Extract bullet points
        bullet_points = re.findall(r"(?:•|-|\*)\s*(.*?)(?=\n|$)", text)
        for point in bullet_points:
            if any(keyword in point.lower() for keyword in benefit_keywords):
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
            doc = self.nlp(text)

            for sent in doc.sents:
                # Look for education requirements
                if any(keyword in sent.text.lower() for keyword in edu_keywords):
                    qualifications.append(sent.text.strip())

        # Extract from bullet points
        bullet_points = re.findall(r"(?:•|-|\*)\s*(.*?)(?=\n|$)", text)
        for point in bullet_points:
            if any(keyword in point.lower() for keyword in edu_keywords):
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
            if any(keyword in req.lower() for keyword in edu_keywords)
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
            "company_name_normalized": company_name.lower(),
            "description_keywords": self._extract_keywords_nlp(description),
            "skills_normalized": [skill.lower() for skill in desc_analysis["skills"]],
            "requirements_count": len(desc_analysis["requirements"]),
            "responsibilities_count": len(desc_analysis["responsibilities"]),
            "has_degree_requirement": any(
                "degree" in qual.lower() for qual in desc_analysis["qualifications"]
            ),
            "title_embedding": title_embedding,
            "description_embedding": desc_embedding,
            "seniority_level": self._detect_seniority_level(job_title, description),
            "technical_skills_count": len(
                [
                    skill
                    for skill in desc_analysis["skills"]
                    if skill.lower()
                    in self.constants.SKILL_KEYWORDS.get("technical", [])
                ]
            ),
        }

    def _extract_keywords_nlp(self, text: str) -> List[str]:
        """
        Extract keywords using NLP and TF-IDF
        """
        if self.nlp:
            doc = self.nlp(text)
            keywords = []

            # Extract important tokens
            for token in doc:
                if not token.is_stop and not token.is_punct and len(token.text) > 2:
                    if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
                        keywords.append(token.lemma_.lower())

            return list(set(keywords))

        # Fallback to simple word extraction
        return list(set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())))

    def _detect_seniority_level(self, job_title: str, description: str) -> str:
        """
        Detect job seniority level from title and description
        """
        text = f"{job_title} {description}".lower()

        if any(
            word in text
            for word in ["senior", "lead", "principal", "architect", "head"]
        ):
            return "senior"
        elif any(word in text for word in ["junior", "entry", "graduate", "trainee"]):
            return "junior"
        else:
            return "mid"
