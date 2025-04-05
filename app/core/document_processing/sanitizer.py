# resume_matcher/core/document_processing/sanitizer.py
import re
import html
from typing import Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ContentSanitizer:
    """
    Handles security aspects of document processing including:
    - Malicious content detection
    - SQL injection prevention
    - XSS prevention
    - Binary content detection
    """

    def __init__(self, max_file_size: int = 10 * 1024 * 1024):  # 10MB default
        self.max_file_size = max_file_size
        self.malicious_patterns = [
            re.compile(r"<\s*script[^>]*>.*?<\s*/\s*script\s*>", re.IGNORECASE),
            re.compile(r"<\s*iframe[^>]*>.*?<\s*/\s*iframe\s*>", re.IGNORECASE),
            re.compile(r"(?i)(SELECT\s.*FROM|INSERT\sINTO|UPDATE\s.*SET|DELETE\sFROM)"),
            re.compile(r"(?i)(DROP\sTABLE|ALTER\sTABLE|CREATE\sTABLE)"),
            re.compile(r"(?i)(javascript:|vbscript:|data:)"),
            re.compile(r"(?i)(<\?php|<\?=)"),
            re.compile(r"(?i)(\\x[0-9a-f]{2})"),
        ]

    def validate_file(self, file_path: Path) -> bool:
        """Check if file is safe to process"""
        try:
            if not file_path.exists():
                raise ValueError(f"File not found: {file_path}")

            if file_path.stat().st_size > self.max_file_size:
                raise ValueError(f"File too large: {file_path.stat().st_size} bytes")

            if not self._is_allowed_extension(file_path):
                raise ValueError(f"Unsupported file extension: {file_path.suffix}")

            return True
        except Exception as e:
            logger.error(f"File validation failed: {str(e)}")
            raise

    def _is_allowed_extension(self, file_path: Path) -> bool:
        allowed_extensions = {".pdf", ".doc", ".docx", ".txt"}
        return file_path.suffix.lower() in allowed_extensions

    def sanitize_text(self, text: str) -> str:
        """Sanitize extracted text content"""
        if not text:
            return ""

        # Remove potentially malicious content
        sanitized = text
        for pattern in self.malicious_patterns:
            sanitized = pattern.sub("", sanitized)

        # HTML escape to prevent XSS
        sanitized = html.escape(sanitized)

        # Remove non-printable characters except newlines and tabs
        sanitized = re.sub(r"[^\x20-\x7E\r\n\t]", "", sanitized)

        return sanitized.strip()
