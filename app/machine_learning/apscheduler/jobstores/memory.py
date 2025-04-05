# apscheduler/jobstores/memory.py
from typing import Dict, List, Optional
from ..job import Job
from .base import BaseJobStore, JobLookupError


class MemoryJobStore(BaseJobStore):
    """Stores jobs in an in-memory dictionary."""

    def __init__(self):
        self.jobs = {}  # job_id -> Job mapping
        super().__init__()

    def add_job(self, job: Job):
        """Add a job to the store."""
        if job.id in self.jobs:
            raise ValueError(f"Job with ID {job.id} already exists")
        self.jobs[job.id] = job

    def update_job(self, job: Job):
        """Update a job in the store."""
        if job.id not in self.jobs:
            raise JobLookupError(job.id)
        self.jobs[job.id] = job

    def remove_job(self, job_id: str):
        """Remove a job from the store."""
        try:
            del self.jobs[job_id]
        except KeyError:
            raise JobLookupError(job_id)

    def load_jobs(self) -> List[Job]:
        """Load all jobs from the store."""
        return list(self.jobs.values())

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a specific job from the store."""
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> List[Job]:
        """Get all jobs from the store."""
        return list(self.jobs.values())

    def remove_all_jobs(self):
        """Remove all jobs from the store."""
        self.jobs.clear()

    def __repr__(self):
        return f"<MemoryJobStore (jobs={len(self.jobs)})>"
