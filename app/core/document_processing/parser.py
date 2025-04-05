# resume_matcher/core/document_processing/parser.py
import fitz  # PyMuPDF
import io
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from .sanitizer import ContentSanitizer

logger = logging.getLogger(__name__)


class ResumeParser:
    """
    Handles parsing of resume documents in various formats (PDF, DOC, DOCX, TXT)
    with security considerations.
    """

    def __init__(self):
        self.sanitizer = ContentSanitizer()

    def parse(self, file_path: Path) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Parse a resume file and return (raw_text, metadata)

        Args:
            file_path: Path to the resume file

        Returns:
            Tuple of (raw_text, metadata) or (None, {}) if parsing fails

        Raises:
            ValueError: If file is invalid or cannot be parsed
        """
        try:
            self.sanitizer.validate_file(file_path)

            if file_path.suffix.lower() == ".pdf":
                return self._parse_pdf(file_path)
            elif file_path.suffix.lower() in {".doc", ".docx"}:
                return self._parse_word(file_path)
            elif file_path.suffix.lower() == ".txt":
                return self._parse_text(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

        except Exception as e:
            logger.error(f"Failed to parse resume {file_path}: {str(e)}")
            raise ValueError(f"Failed to parse resume: {str(e)}") from e

    def _parse_pdf(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse PDF file using PyMuPDF"""
        metadata = {}
        text_parts = []

        try:
            with fitz.open(file_path) as doc:
                metadata = {
                    "format": "PDF",
                    "pages": len(doc),
                    "author": doc.metadata.get("author", ""),
                    "title": doc.metadata.get("title", ""),
                    "creation_date": doc.metadata.get("creationDate", ""),
                    "modification_date": doc.metadata.get("modDate", ""),
                }

                for page in doc:
                    text = page.get_text()
                    sanitized = self.sanitizer.sanitize_text(text)
                    text_parts.append(sanitized)

            return "\n\n".join(text_parts), metadata

        except Exception as e:
            logger.error(f"PDF parsing error for {file_path}: {str(e)}")
            raise ValueError(f"PDF parsing failed: {str(e)}") from e

    def _parse_word(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse Word documents using python-docx"""
        try:
            import docx
            from docx.opc.exceptions import PackageNotFoundError

            metadata = {"format": "Word"}
            text_parts = []

            try:
                doc = docx.Document(file_path)
                metadata["core_properties"] = {
                    "author": doc.core_properties.author,
                    "created": doc.core_properties.created,
                    "modified": doc.core_properties.modified,
                    "title": doc.core_properties.title,
                }

                for para in doc.paragraphs:
                    sanitized = self.sanitizer.sanitize_text(para.text)
                    text_parts.append(sanitized)

                return "\n\n".join(text_parts), metadata

            except PackageNotFoundError:
                # Fallback to textract if python-docx fails
                return self._fallback_parse(file_path)

        except ImportError:
            logger.warning("python-docx not available, falling back to textract")
            return self._fallback_parse(file_path)

    def _parse_text(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse plain text files"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                sanitized = self.sanitizer.sanitize_text(text)
                return sanitized, {"format": "Text"}
        except Exception as e:
            logger.error(f"Text file parsing error for {file_path}: {str(e)}")
            raise ValueError(f"Text file parsing failed: {str(e)}") from e

    def _fallback_parse(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Fallback parsing using textract"""
        try:
            import textract

            text = textract.process(str(file_path)).decode("utf-8")
            sanitized = self.sanitizer.sanitize_text(text)
            return sanitized, {"format": "Fallback"}
        except Exception as e:
            logger.error(f"Fallback parsing failed for {file_path}: {str(e)}")
            raise ValueError(f"Fallback parsing failed: {str(e)}") from e
