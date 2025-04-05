# apscheduler/executors/base.py
from ..job import Job


class BaseExecutor:
    """Abstract base class for all executors."""

    def start(self, scheduler, alias):
        """Start the executor.

        Args:
            scheduler: The scheduler instance that owns this executor
            alias: The alias this executor was referenced by in the scheduler
        """
        self.scheduler = scheduler
        self.alias = alias
        # logger.info(f"Started executor '{alias}'")

    def shutdown(self, wait=True):
        """Shutdown the executor."""
        pass

    def submit_job(self, job: Job):
        """Submit job for execution."""
        raise NotImplementedError

    def __repr__(self):
        return "<%s>" % self.__class__.__name__
