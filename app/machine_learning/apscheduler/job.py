# apscheduler/job.py
from typing import Any, Optional, Callable, Tuple, Dict
from datetime import datetime
from .triggers.base import BaseTrigger


class Job:
    """Container for job details and state."""

    def __init__(
        self,
        scheduler,
        func: Callable,
        trigger: Optional[BaseTrigger] = None,
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict] = None,
        **options
    ):
        self.scheduler = scheduler
        self.func = func
        self.trigger = trigger
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.options = options
        self.id = options.get("id")
        self.name = options.get("name")
        self.next_run_time = None
        self._pending = True

    def should_run(self, now: datetime) -> bool:
        """Return whether this job should run now."""
        if self._pending:
            return False

        if self.next_run_time is None:
            self.next_run_time = self.trigger.get_next_fire_time(None, now)
            return False

        return now >= self.next_run_time

    def modify(self, **changes):
        """Modify the job's properties."""
        for attr, value in changes.items():
            setattr(self, attr, value)

    def pause(self):
        """Pause the job."""
        self._pending = True

    def resume(self):
        """Resume the job."""
        self._pending = False

    def __repr__(self):
        return "<Job (id=%s name=%s)>" % (self.id, self.name)
