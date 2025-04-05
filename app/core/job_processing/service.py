# resume_matcher/core/job_processing/service.py
import logging
from typing import List, Dict, Any
from pathlib import Path
import json

from .models import JobPosting
from .parser import JobParser
from .analyzer import JobAnalyzer
from ..document_processing.sanitizer import ContentSanitizer
from ...utils.logger import logger


class JobProcessingService:
    """
    High-level service for processing job postings from various sources.
    Handles both single jobs and batch processing.
    """

    def __init__(self):
        self.parser = JobParser()
        self.analyzer = JobAnalyzer()
        self.sanitizer = ContentSanitizer()

    def process_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single job posting from raw data to analyzed format.

        Args:
            job_data: Dictionary containing raw job posting data

        Returns:
            Dictionary containing processed job data
        """
        try:
            # Parse and validate
            job_posting = self.parser.parse_job(job_data)

            # Analyze and add features
            analyzed_job = self.analyzer.analyze_jobs([job_posting])[0]

            return self._serialize_job(analyzed_job)

        except Exception as e:
            logger.error(f"Job processing failed: {str(e)}")
            raise

    def process_job_batch(
        self, jobs_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple job postings in batch.

        Args:
            jobs_data: List of dictionaries containing raw job posting data

        Returns:
            List of dictionaries containing processed job data
        """
        try:
            # Parse all jobs first
            job_postings = []
            for job_data in jobs_data:
                try:
                    job_postings.append(self.parser.parse_job(job_data))
                except Exception as e:
                    logger.warning(f"Skipping invalid job: {str(e)}")

            # Analyze all jobs together (better for TF-IDF)
            analyzed_jobs = self.analyzer.analyze_jobs(job_postings)

            return [self._serialize_job(job) for job in analyzed_jobs]

        except Exception as e:
            logger.error(f"Job batch processing failed: {str(e)}")
            raise

    def process_job_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Process jobs from a JSON file.

        Args:
            file_path: Path to JSON file containing job data

        Returns:
            List of dictionaries containing processed job data
        """
        try:
            self.sanitizer.validate_file(file_path)

            with open(file_path, "r", encoding="utf-8") as f:
                jobs_data = json.load(f)

            if not isinstance(jobs_data, list):
                jobs_data = [jobs_data]

            return self.process_job_batch(jobs_data)

        except Exception as e:
            logger.error(f"Job file processing failed: {str(e)}")
            raise

    def _serialize_job(self, job: JobPosting) -> Dict[str, Any]:
        """Convert JobPosting object to serializable dictionary"""
        return {
            "job_id": job.job_id,
            "job_title": job.job_title,
            "company_name": job.company_name,
            "job_type": job.job_type,
            "salary": job.salary,
            "experience": job.experience,
            "location": job.location,
            "description": job.description,
            "apply_link": job.apply_link,
            "posting_date": job.posting_date.isoformat(),
            "skills": job.skills,
            "requirements": job.requirements,
            "responsibilities": job.responsibilities,
            "benefits": job.benefits,
            "qualifications": job.qualifications,
            "normalized_features": job.normalized_features,
        }
