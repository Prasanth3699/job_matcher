# apscheduler/triggers/interval.py
from datetime import datetime, timedelta
from typing import Optional, Union
from ..triggers.base import BaseTrigger


class IntervalTrigger(BaseTrigger):
    """
    Triggers on specified intervals.
    """

    def __init__(
        self,
        weeks=0,
        days=0,
        hours=0,
        minutes=0,
        seconds=0,
        start_date=None,
        end_date=None,
        timezone=None,
        jitter=None,
    ):
        self.interval = timedelta(
            weeks=weeks, days=days, hours=hours, minutes=minutes, seconds=seconds
        )
        self.start_date = start_date
        self.end_date = end_date
        self.timezone = timezone
        self.jitter = jitter

    def get_next_fire_time(self, previous_fire_time, now):
        if previous_fire_time:
            next_fire_time = previous_fire_time + self.interval
        else:
            next_fire_time = now + self.interval

        if self.end_date and next_fire_time > self.end_date:
            return None

        return next_fire_time

    def __str__(self):
        return "interval[%s]" % str(self.interval)
