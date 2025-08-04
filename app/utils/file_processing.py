"""
Unified file processing utilities for the resume job matcher service.

This module provides centralized file handling functionality to eliminate
code duplication and ensure consistent file operations across the application.
"""

import os
import re
import tempfile
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union, BinaryIO
from contextlib import contextmanager
import logging
import shutil

from app.core.constants import (
    FileConstraints,
    ValidationRules,
    ErrorCodes,
    ErrorMessages,
    BusinessRules,
)
from app.utils.validation import FileValidator, ValidationResult, ValidationError

logger = logging.getLogger(__name__)


class FileProcessingError(Exception):
    """Custom exception for file processing errors."""

    def __init__(
        self,
        message: str,
        error_code: str = ErrorCodes.PROCESSING_RESUME_PARSE_FAILED,
        original_error: Optional[Exception] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.original_error = original_error
        super().__init__(self.message)


class SecureFileHandler:
    """
    Secure file handling utility with validation and cleanup.

    Provides safe file operations with automatic cleanup and validation
    to prevent security issues and resource leaks.
    """

    @staticmethod
    def validate_and_save_upload(
        file_content: bytes,
        filename: str,
        content_type: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> Tuple[Path, ValidationResult]:
        """
        Validate and save uploaded file to secure temporary location.

        Args:
            file_content: File content as bytes
            filename: Original filename
            content_type: MIME type of the file
            max_size: Maximum allowed file size (overrides default)

        Returns:
            Tuple of (temp_file_path, validation_result)

        Raises:
            FileProcessingError: If file processing fails
        """
        # Validate file upload
        file_size = len(file_content)
        max_allowed_size = max_size or FileConstraints.MAX_FILE_SIZE

        validation_result = FileValidator.validate_file_upload(
            filename=filename,
            file_size=file_size,
            content_type=content_type,
            file_content=file_content,
        )

        if not validation_result.is_valid:
            return None, validation_result

        try:
            # Create secure temporary file
            temp_path = SecureFileHandler._create_secure_temp_file(
                file_content, filename
            )

            logger.info(f"File saved to secure temporary location: {temp_path}")
            return temp_path, validation_result

        except Exception as e:
            logger.error(f"Failed to save uploaded file: {e}")
            raise FileProcessingError(
                f"Failed to save uploaded file: {str(e)}",
                error_code=ErrorCodes.PROCESSING_RESUME_PARSE_FAILED,
                original_error=e,
            )

    @staticmethod
    def _create_secure_temp_file(file_content: bytes, original_filename: str) -> Path:
        """
        Create a secure temporary file with proper permissions.

        Args:
            file_content: File content to write
            original_filename: Original filename for extension detection

        Returns:
            Path to the created temporary file
        """
        # Generate secure filename
        file_extension = Path(original_filename).suffix.lower()
        secure_filename = (
            f"resume_{hashlib.md5(file_content).hexdigest()}{file_extension}"
        )

        # Create temporary file with secure permissions
        temp_dir = tempfile.gettempdir()
        temp_path = Path(temp_dir) / secure_filename

        # Write file with secure permissions (read/write for owner only)
        with open(temp_path, "wb") as temp_file:
            temp_file.write(file_content)

        # Set secure file permissions (owner read/write only)
        os.chmod(temp_path, 0o600)

        return temp_path

    @staticmethod
    def cleanup_temp_file(file_path: Union[str, Path]) -> bool:
        """
        Safely delete temporary file.

        Args:
            file_path: Path to file to delete

        Returns:
            True if file was deleted successfully, False otherwise
        """
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                os.unlink(path)
                logger.debug(f"Cleaned up temporary file: {path}")
                return True
            else:
                logger.warning(f"Temporary file not found for cleanup: {path}")
                return False
        except Exception as e:
            logger.error(f"Failed to cleanup temporary file {file_path}: {e}")
            return False

    @staticmethod
    @contextmanager
    def temporary_file_context(
        file_content: bytes, filename: str, content_type: Optional[str] = None
    ):
        """
        Context manager for temporary file handling with automatic cleanup.

        Args:
            file_content: File content as bytes
            filename: Original filename
            content_type: MIME type of the file

        Yields:
            Path to temporary file

        Example:
            with SecureFileHandler.temporary_file_context(content, "resume.pdf") as temp_path:
                # Process file
                parsed_data = parser.parse(temp_path)
            # File is automatically cleaned up
        """
        temp_path = None
        try:
            temp_path, validation_result = SecureFileHandler.validate_and_save_upload(
                file_content, filename, content_type
            )

            if not validation_result.is_valid:
                raise FileProcessingError(
                    f"File validation failed: {validation_result.get_error_messages()}",
                    error_code=ErrorCodes.VALIDATION_INVALID_FILE_TYPE,
                )

            yield temp_path

        finally:
            if temp_path:
                SecureFileHandler.cleanup_temp_file(temp_path)


class FileMetadataExtractor:
    """
    Utility for extracting file metadata and properties.
    """

    @staticmethod
    def extract_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata
        """
        path = Path(file_path)

        if not path.exists():
            raise FileProcessingError(
                f"File does not exist: {path}",
                error_code=ErrorCodes.RESOURCE_RESUME_NOT_FOUND,
            )

        try:
            stat_info = path.stat()

            metadata = {
                "filename": path.name,
                "file_size": stat_info.st_size,
                "file_extension": path.suffix.lower(),
                "created_time": stat_info.st_ctime,
                "modified_time": stat_info.st_mtime,
                "is_readable": os.access(path, os.R_OK),
                "is_writable": os.access(path, os.W_OK),
            }

            # Detect MIME type
            mime_type, encoding = mimetypes.guess_type(str(path))
            metadata["mime_type"] = mime_type
            metadata["encoding"] = encoding

            # Calculate file hash for integrity checking
            metadata["file_hash"] = FileMetadataExtractor._calculate_file_hash(path)

            # Validate against constraints
            metadata["is_valid_size"] = FileConstraints.is_valid_size(
                metadata["file_size"]
            )
            metadata["is_valid_extension"] = FileConstraints.is_valid_extension(
                metadata["filename"]
            )
            metadata["is_valid_mime_type"] = (
                FileConstraints.is_valid_mime_type(metadata["mime_type"])
                if metadata["mime_type"]
                else False
            )

            return metadata

        except Exception as e:
            logger.error(f"Failed to extract metadata from {path}: {e}")
            raise FileProcessingError(
                f"Failed to extract file metadata: {str(e)}",
                error_code=ErrorCodes.PROCESSING_RESUME_PARSE_FAILED,
                original_error=e,
            )

    @staticmethod
    def _calculate_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
        """
        Calculate hash of file content for integrity verification.

        Args:
            file_path: Path to the file
            algorithm: Hash algorithm to use

        Returns:
            Hexadecimal hash string
        """
        hash_obj = hashlib.new(algorithm)

        with open(file_path, "rb") as file:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: file.read(4096), b""):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    @staticmethod
    def validate_file_integrity(
        file_path: Union[str, Path], expected_hash: str, algorithm: str = "sha256"
    ) -> bool:
        """
        Validate file integrity using hash comparison.

        Args:
            file_path: Path to the file
            expected_hash: Expected hash value
            algorithm: Hash algorithm used

        Returns:
            True if file integrity is valid, False otherwise
        """
        try:
            actual_hash = FileMetadataExtractor._calculate_file_hash(
                Path(file_path), algorithm
            )
            return actual_hash.lower() == expected_hash.lower()
        except Exception as e:
            logger.error(f"Failed to validate file integrity: {e}")
            return False


class FileContentAnalyzer:
    """
    Utility for analyzing file content and extracting basic information.
    """

    @staticmethod
    def analyze_text_content(content: str) -> Dict[str, Any]:
        """
        Analyze text content for basic statistics and properties.

        Args:
            content: Text content to analyze

        Returns:
            Dictionary containing content analysis results
        """
        if not content:
            return {
                "character_count": 0,
                "word_count": 0,
                "line_count": 0,
                "paragraph_count": 0,
                "is_sufficient_content": False,
                "estimated_reading_time": 0,
            }

        # Basic statistics
        character_count = len(content)
        word_count = len(content.split())
        line_count = len(content.splitlines())
        paragraph_count = len([p for p in content.split("\n\n") if p.strip()])

        # Content quality assessment
        is_sufficient_content = (
            character_count >= FileConstraints.MIN_TEXT_LENGTH
            and word_count >= FileConstraints.MIN_WORDS
        )

        # Estimated reading time (assume 200 words per minute)
        estimated_reading_time = max(1, word_count // 200)

        # Language detection (basic heuristic)
        language_hints = FileContentAnalyzer._detect_language_hints(content)

        return {
            "character_count": character_count,
            "word_count": word_count,
            "line_count": line_count,
            "paragraph_count": paragraph_count,
            "is_sufficient_content": is_sufficient_content,
            "estimated_reading_time": estimated_reading_time,
            "language_hints": language_hints,
        }

    @staticmethod
    def _detect_language_hints(content: str) -> Dict[str, Any]:
        """
        Basic language detection using simple heuristics.

        Args:
            content: Text content to analyze

        Returns:
            Dictionary with language detection hints
        """
        # Count common English words
        common_english_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "experience",
            "work",
            "job",
            "company",
            "skills",
            "education",
            "university",
            "college",
            "degree",
        }

        words = set(word.lower().strip(".,!?;") for word in content.split())
        english_word_count = len(words.intersection(common_english_words))
        total_words = len(words)

        english_ratio = english_word_count / max(total_words, 1)

        return {
            "likely_english": english_ratio > 0.1,
            "english_word_ratio": english_ratio,
            "total_unique_words": total_words,
            "common_english_words_found": english_word_count,
        }

    @staticmethod
    def extract_contact_info_hints(content: str) -> Dict[str, Any]:
        """
        Extract basic contact information hints from text content.

        Args:
            content: Text content to analyze

        Returns:
            Dictionary with contact information hints
        """
        # Basic patterns for contact information
        email_pattern = ValidationRules.EMAIL_PATTERN
        phone_pattern = re.compile(r"[\+]?[1-9][\d\s\-\(\)]{7,15}")
        url_pattern = ValidationRules.URL_PATTERN

        emails = email_pattern.findall(content)
        phones = phone_pattern.findall(content)
        urls = url_pattern.findall(content)

        return {
            "has_email": len(emails) > 0,
            "email_count": len(emails),
            "has_phone": len(phones) > 0,
            "phone_count": len(phones),
            "has_urls": len(urls) > 0,
            "url_count": len(urls),
            "appears_to_be_resume": (
                len(emails) > 0
                and any(
                    keyword in content.lower()
                    for keyword in [
                        "experience",
                        "education",
                        "skills",
                        "work",
                        "employment",
                    ]
                )
            ),
        }


class BulkFileProcessor:
    """
    Utility for processing multiple files efficiently.
    """

    @staticmethod
    def process_file_batch(
        files: List[Tuple[bytes, str, Optional[str]]], max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Process multiple files with error isolation.

        Args:
            files: List of (content, filename, content_type) tuples
            max_concurrent: Maximum concurrent file processing

        Returns:
            List of processing results with error information
        """
        results = []

        for i, (content, filename, content_type) in enumerate(files):
            try:
                # Validate and process file
                temp_path, validation_result = (
                    SecureFileHandler.validate_and_save_upload(
                        content, filename, content_type
                    )
                )

                if validation_result.is_valid and temp_path:
                    # Extract metadata
                    metadata = FileMetadataExtractor.extract_metadata(temp_path)

                    # Analyze content if it's a text-based file
                    content_analysis = None
                    if metadata["mime_type"] and "text" in metadata["mime_type"]:
                        try:
                            text_content = content.decode("utf-8", errors="ignore")
                            content_analysis = FileContentAnalyzer.analyze_text_content(
                                text_content
                            )
                        except Exception:
                            pass

                    results.append(
                        {
                            "index": i,
                            "success": True,
                            "filename": filename,
                            "temp_path": str(temp_path),
                            "metadata": metadata,
                            "content_analysis": content_analysis,
                            "validation_errors": [],
                        }
                    )

                else:
                    results.append(
                        {
                            "index": i,
                            "success": False,
                            "filename": filename,
                            "temp_path": None,
                            "metadata": None,
                            "content_analysis": None,
                            "validation_errors": [
                                error.to_dict() for error in validation_result.errors
                            ],
                        }
                    )

            except Exception as e:
                logger.error(f"Failed to process file {filename}: {e}")
                results.append(
                    {
                        "index": i,
                        "success": False,
                        "filename": filename,
                        "temp_path": None,
                        "metadata": None,
                        "content_analysis": None,
                        "validation_errors": [{"error": str(e)}],
                    }
                )

        return results

    @staticmethod
    def cleanup_batch_results(results: List[Dict[str, Any]]) -> int:
        """
        Cleanup temporary files from batch processing results.

        Args:
            results: Results from process_file_batch

        Returns:
            Number of files successfully cleaned up
        """
        cleaned_count = 0

        for result in results:
            temp_path = result.get("temp_path")
            if temp_path and SecureFileHandler.cleanup_temp_file(temp_path):
                cleaned_count += 1

        return cleaned_count


# Convenience functions for common file operations
def save_upload_securely(
    file_content: bytes, filename: str, content_type: Optional[str] = None
) -> Tuple[Path, ValidationResult]:
    """Convenience function for secure file upload."""
    return SecureFileHandler.validate_and_save_upload(
        file_content, filename, content_type
    )


def cleanup_temp_file(file_path: Union[str, Path]) -> bool:
    """Convenience function for file cleanup."""
    return SecureFileHandler.cleanup_temp_file(file_path)


def get_file_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Convenience function for metadata extraction."""
    return FileMetadataExtractor.extract_metadata(file_path)


def analyze_text_content(content: str) -> Dict[str, Any]:
    """Convenience function for text content analysis."""
    return FileContentAnalyzer.analyze_text_content(content)
