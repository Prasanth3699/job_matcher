# apscheduler/jobstores/base.py
from typing import List, Optional
from ..job import Job


class JobLookupError(LookupError):
    """Raised when a job is not found in a job store."""

    def __init__(self, job_id):
        super().__init__(f'Job "{job_id}" not found in job store')


class BaseJobStore:
    """Abstract base class for all job stores."""

    def start(self, scheduler, alias):
        """Start the job store.

        Args:
            scheduler: The scheduler instance that owns this job store
            alias: The alias this job store was referenced by in the scheduler
        """
        self.scheduler = scheduler
        self.alias = alias

    def shutdown(self):
        """Shutdown the job store."""
        pass

    def add_job(self, job: Job):
        """Add a job to the store."""
        raise NotImplementedError

    def update_job(self, job: Job):
        """Update a job in the store."""
        raise NotImplementedError

    def remove_job(self, job_id: str):
        """Remove a job from the store."""
        raise NotImplementedError

    def load_jobs(self) -> List[Job]:
        """Load all jobs from the store."""
        raise NotImplementedError

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a specific job from the store."""
        raise NotImplementedError

    def get_all_jobs(self) -> List[Job]:
        """Get all jobs from the store."""
        return self.load_jobs()

    def remove_all_jobs(self):
        """Remove all jobs from the store."""
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.__class__.__name__}>"
