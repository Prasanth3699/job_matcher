# apscheduler/events.py
from enum import IntEnum


class Event(IntEnum):
    """Base class for scheduler events."""

    pass


class SchedulerEvent(Event):
    """Events related to scheduler operation."""

    SCHEDULER_START = 1
    SCHEDULER_SHUTDOWN = 2
    SCHEDULER_PAUSED = 3
    SCHEDULER_RESUMED = 4


class JobEvent(Event):
    """Events related to job execution."""

    JOB_ADDED = 10
    JOB_REMOVED = 11
    JOB_MODIFIED = 12
    JOB_EXECUTED = 20
    JOB_ERROR = 21
    JOB_MISSED = 22


# Common event masks
EVENT_ALL = (
    SchedulerEvent.SCHEDULER_START
    | SchedulerEvent.SCHEDULER_SHUTDOWN
    | JobEvent.JOB_ADDED
    | JobEvent.JOB_REMOVED
    | JobEvent.JOB_MODIFIED
    | JobEvent.JOB_EXECUTED
    | JobEvent.JOB_ERROR
    | JobEvent.JOB_MISSED
)
EVENT_JOB_EXECUTED = JobEvent.JOB_EXECUTED
EVENT_JOB_ERROR = JobEvent.JOB_ERROR
