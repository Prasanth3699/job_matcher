# apscheduler/executors/pool.py
import concurrent.futures
from typing import Any
from .base import BaseExecutor
import logging

logger = logging.getLogger("apscheduler.executors.pool")


class ThreadPoolExecutor(BaseExecutor):
    """An executor that runs jobs in a thread pool."""

    def __init__(self, max_workers=10):
        self.max_workers = max_workers
        self._pool = None

    def start(self, scheduler, alias):
        super().start(scheduler, alias)
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        logger.info(f"ThreadPoolExecutor started with {self.max_workers} workers")

    def shutdown(self, wait=True):
        if self._pool:
            self._pool.shutdown(wait=wait)
        super().shutdown(wait)

    def submit_job(self, job):
        """Submit a job to the thread pool."""
        try:
            future = self._pool.submit(job.func, *job.args, **job.kwargs)
            future.add_done_callback(self._job_done_callback)
        except Exception as e:
            logger.error(f"Failed to submit job {job.id}: {str(e)}")
            raise

    def _job_done_callback(self, future):
        """Callback when a job completes."""
        try:
            future.result()
        except Exception as e:
            logger.error(f"Job failed: {str(e)}")
