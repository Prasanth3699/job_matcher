"""
Validation rules and constraints for the resume job matcher service.
"""

import re
from typing import List, Dict, Pattern, Any


class ValidationRules:
    """Validation rules and patterns for various data types."""
    
    # Email validation
    EMAIL_PATTERN = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    # Phone number validation (international format)
    PHONE_PATTERN = re.compile(
        r'^\+?[1-9]\d{1,14}$'
    )
    
    # URL validation
    URL_PATTERN = re.compile(
        r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?$'
    )
    
    # Job ID validation (positive integers)
    JOB_ID_PATTERN = re.compile(r'^\d+$')
    
    # File name validation (alphanumeric, dots, dashes, underscores)
    FILENAME_PATTERN = re.compile(
        r'^[a-zA-Z0-9._-]+\.[a-zA-Z0-9]+$'
    )
    
    # Skill name validation (letters, spaces, some special chars)
    SKILL_NAME_PATTERN = re.compile(
        r'^[a-zA-Z0-9\s\+\#\.\-\/\(\)]{2,50}$'
    )
    
    # Salary validation (numbers with optional K, M, commas, currency symbols)
    SALARY_PATTERN = re.compile(
        r'^\$?[\d,]+(?:\.\d{2})?(?:[KkMm])?(?:\s*-\s*\$?[\d,]+(?:\.\d{2})?(?:[KkMm])?)?$'
    )
    
    # Location validation (letters, spaces, commas, periods)
    LOCATION_PATTERN = re.compile(
        r'^[a-zA-Z0-9\s,.-]{2,100}$'
    )
    
    # Company name validation
    COMPANY_NAME_PATTERN = re.compile(
        r'^[a-zA-Z0-9\s&.,\-\'\"()]{2,100}$'
    )
    
    # Job title validation
    JOB_TITLE_PATTERN = re.compile(
        r'^[a-zA-Z0-9\s,.\-/()&]{2,100}$'
    )


class FileConstraints:
    """File upload and processing constraints."""
    
    # Supported file types
    ALLOWED_RESUME_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt']
    ALLOWED_MIME_TYPES = [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'text/plain'
    ]
    
    # File size limits (in bytes)
    MIN_FILE_SIZE = 100          # 100 bytes
    MAX_FILE_SIZE = 10485760     # 10 MB
    OPTIMAL_FILE_SIZE = 2097152  # 2 MB
    
    # Content length limits
    MIN_TEXT_LENGTH = 100        # characters
    MAX_TEXT_LENGTH = 50000      # characters
    OPTIMAL_TEXT_LENGTH = 5000   # characters
    
    # Document structure limits
    MAX_PAGES = 10
    MIN_WORDS = 50
    MAX_WORDS = 10000
    
    # File name constraints
    MAX_FILENAME_LENGTH = 255
    MIN_FILENAME_LENGTH = 1
    
    @classmethod
    def is_valid_extension(cls, filename: str) -> bool:
        """Check if file extension is allowed."""
        if not filename:
            return False
        return any(filename.lower().endswith(ext) for ext in cls.ALLOWED_RESUME_EXTENSIONS)
    
    @classmethod
    def is_valid_size(cls, size: int) -> bool:
        """Check if file size is within limits."""
        return cls.MIN_FILE_SIZE <= size <= cls.MAX_FILE_SIZE
    
    @classmethod
    def is_valid_mime_type(cls, mime_type: str) -> bool:
        """Check if MIME type is allowed."""
        return mime_type in cls.ALLOWED_MIME_TYPES


class InputLimits:
    """Input validation limits for various fields."""
    
    # String field limits
    STRING_LIMITS = {
        'user_id': {'min': 1, 'max': 50},
        'match_job_id': {'min': 1, 'max': 50},
        'job_title': {'min': 2, 'max': 100},
        'company_name': {'min': 2, 'max': 100},
        'location': {'min': 2, 'max': 100},
        'skill_name': {'min': 2, 'max': 50},
        'description': {'min': 10, 'max': 5000},
        'requirements': {'min': 10, 'max': 2000},
        'responsibilities': {'min': 10, 'max': 2000},
        'benefits': {'min': 5, 'max': 1000},
        'qualifications': {'min': 10, 'max': 1000}
    }
    
    # Numeric field limits
    NUMERIC_LIMITS = {
        'job_id': {'min': 1, 'max': 2147483647},  # Max int32
        'user_id': {'min': 1, 'max': 2147483647},
        'experience_years': {'min': 0, 'max': 50},
        'salary_min': {'min': 0, 'max': 10000000},
        'salary_max': {'min': 0, 'max': 10000000},
        'match_score': {'min': 0.0, 'max': 1.0},
        'confidence_score': {'min': 0.0, 'max': 1.0}
    }
    
    # Array field limits
    ARRAY_LIMITS = {
        'job_ids': {'min_items': 1, 'max_items': 50},
        'skills': {'min_items': 1, 'max_items': 100},
        'requirements': {'min_items': 0, 'max_items': 50},
        'responsibilities': {'min_items': 0, 'max_items': 50},
        'benefits': {'min_items': 0, 'max_items': 30},
        'qualifications': {'min_items': 0, 'max_items': 30},
        'preferred_locations': {'min_items': 0, 'max_items': 10},
        'preferred_companies': {'min_items': 0, 'max_items': 20}
    }
    
    # Date/time limits
    DATE_LIMITS = {
        'min_year': 1950,
        'max_year': 2030,
        'max_future_days': 365,
        'max_past_years': 80
    }


class TextProcessingRules:
    """Rules for text processing and cleaning."""
    
    # Characters to remove or replace
    FORBIDDEN_CHARS = ['\x00', '\x01', '\x02', '\x03', '\x04', '\x05']
    REPLACE_CHARS = {
        '\u2019': "'",  # Right single quotation mark
        '\u2018': "'",  # Left single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u00a0': ' ',  # Non-breaking space
    }
    
    # Text normalization rules
    NORMALIZE_WHITESPACE = True
    REMOVE_HTML_TAGS = True
    DECODE_HTML_ENTITIES = True
    NORMALIZE_UNICODE = True
    
    # Content filtering
    MIN_WORD_LENGTH = 2
    MAX_WORD_LENGTH = 50
    REMOVE_NUMBERS_ONLY = True
    REMOVE_SPECIAL_CHARS_ONLY = True
    
    # Language detection
    SUPPORTED_LANGUAGES = ['en', 'es', 'fr', 'de', 'it', 'pt']
    DEFAULT_LANGUAGE = 'en'
    MIN_CONFIDENCE_LANGUAGE = 0.8


class SecurityRules:
    """Security validation rules and constraints."""
    
    # SQL injection prevention patterns
    SQL_INJECTION_PATTERNS = [
        re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)", re.IGNORECASE),
        re.compile(r"(\b(OR|AND)\s+\d+\s*=\s*\d+)", re.IGNORECASE),
        re.compile(r"(\';|\";\s*--)", re.IGNORECASE),
        re.compile(r"(\bxp_\w+\b)", re.IGNORECASE)
    ]
    
    # XSS prevention patterns
    XSS_PATTERNS = [
        re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),
        re.compile(r"<iframe[^>]*>.*?</iframe>", re.IGNORECASE | re.DOTALL)
    ]
    
    # Path traversal prevention
    PATH_TRAVERSAL_PATTERNS = [
        re.compile(r"\.\.[\\/]"),
        re.compile(r"[\\/]\.\."),
        re.compile(r"%2e%2e"),
        re.compile(r"0x2e0x2e")
    ]
    
    # Content validation rules
    MAX_UPLOAD_ATTEMPTS = 5
    ALLOWED_DOMAINS = []  # Define allowed domains for URLs
    BLOCKED_DOMAINS = ['malware.com', 'phishing.net']  # Example blocked domains
    
    # Rate limiting rules
    RATE_LIMITS = {
        'requests_per_minute': 60,
        'requests_per_hour': 1000,
        'uploads_per_day': 100,
        'matches_per_day': 50
    }


class BusinessValidationRules:
    """Business logic validation rules."""
    
    # Job matching rules
    MATCHING_RULES = {
        'min_resume_skills': 3,
        'max_job_skills': 100,
        'min_match_confidence': 0.3,
        'max_processing_time_minutes': 10
    }
    
    # User preference rules
    PREFERENCE_RULES = {
        'max_preferred_locations': 10,
        'max_preferred_companies': 20,
        'max_salary_range_ratio': 3.0,  # max/min salary ratio
        'valid_job_types': [
            'full-time', 'part-time', 'contract', 'temporary', 
            'internship', 'freelance', 'remote'
        ]
    }
    
    # System resource rules
    RESOURCE_RULES = {
        'max_concurrent_matches': 100,
        'max_queue_size': 1000,
        'max_memory_usage_mb': 2048,
        'max_processing_timeout_seconds': 600
    }
    
    # Data quality rules
    QUALITY_RULES = {
        'min_job_description_words': 20,
        'min_resume_sections': 2,  # Contact + at least one other section
        'required_resume_sections': ['contact'],
        'optional_resume_sections': [
            'summary', 'experience', 'education', 'skills', 
            'certifications', 'projects'
        ]
    }


class ValidationHelpers:
    """Helper methods for validation operations."""
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Validate email address format."""
        if not email or len(email) > 254:
            return False
        return bool(ValidationRules.EMAIL_PATTERN.match(email))
    
    @staticmethod
    def is_valid_phone(phone: str) -> bool:
        """Validate phone number format."""
        if not phone:
            return False
        cleaned = re.sub(r'[\s\-\(\)]', '', phone)
        return bool(ValidationRules.PHONE_PATTERN.match(cleaned))
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Validate URL format."""
        if not url or len(url) > 2048:
            return False
        return bool(ValidationRules.URL_PATTERN.match(url))
    
    @staticmethod
    def is_safe_content(text: str) -> bool:
        """Check if text content is safe from security threats."""
        if not text:
            return True
        
        # Check for SQL injection patterns
        for pattern in SecurityRules.SQL_INJECTION_PATTERNS:
            if pattern.search(text):
                return False
        
        # Check for XSS patterns
        for pattern in SecurityRules.XSS_PATTERNS:
            if pattern.search(text):
                return False
        
        # Check for path traversal patterns
        for pattern in SecurityRules.PATH_TRAVERSAL_PATTERNS:
            if pattern.search(text):
                return False
        
        return True
    
    @staticmethod
    def validate_string_length(value: str, field_name: str) -> bool:
        """Validate string length against defined limits."""
        if field_name not in InputLimits.STRING_LIMITS:
            return True
        
        limits = InputLimits.STRING_LIMITS[field_name]
        length = len(value) if value else 0
        return limits['min'] <= length <= limits['max']
    
    @staticmethod
    def validate_numeric_range(value: float, field_name: str) -> bool:
        """Validate numeric value against defined limits."""
        if field_name not in InputLimits.NUMERIC_LIMITS:
            return True
        
        limits = InputLimits.NUMERIC_LIMITS[field_name]
        return limits['min'] <= value <= limits['max']
    
    @staticmethod
    def validate_array_length(items: List[Any], field_name: str) -> bool:
        """Validate array length against defined limits."""
        if field_name not in InputLimits.ARRAY_LIMITS:
            return True
        
        limits = InputLimits.ARRAY_LIMITS[field_name]
        length = len(items) if items else 0
        return limits['min_items'] <= length <= limits['max_items']
    
    @staticmethod
    def clean_text_content(text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove forbidden characters
        for char in TextProcessingRules.FORBIDDEN_CHARS:
            text = text.replace(char, '')
        
        # Replace special characters
        for old_char, new_char in TextProcessingRules.REPLACE_CHARS.items():
            text = text.replace(old_char, new_char)
        
        # Normalize whitespace
        if TextProcessingRules.NORMALIZE_WHITESPACE:
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text