# apscheduler/triggers/cron.py
from datetime import datetime, timedelta
from typing import Optional, Union, List
from ..triggers.base import BaseTrigger


class CronTrigger(BaseTrigger):
    """
    Triggers when current time matches all specified time constraints.
    """

    def __init__(
        self,
        year=None,
        month=None,
        day=None,
        week=None,
        day_of_week=None,
        hour=None,
        minute=None,
        second=None,
        start_date=None,
        end_date=None,
        timezone=None,
        jitter=None,
    ):
        self.year = year
        self.month = month
        self.day = day
        self.week = week
        self.day_of_week = day_of_week
        self.hour = hour
        self.minute = minute
        self.second = second
        self.start_date = start_date
        self.end_date = end_date
        self.timezone = timezone
        self.jitter = jitter

    def get_next_fire_time(self, previous_fire_time, now):
        # Simplified implementation - actual cron would be more complex
        if previous_fire_time:
            next_fire_time = previous_fire_time + timedelta(days=1)
        else:
            next_fire_time = now.replace(
                hour=self.hour or 0,
                minute=self.minute or 0,
                second=self.second or 0,
                microsecond=0,
            )
            if next_fire_time < now:
                next_fire_time += timedelta(days=1)

        if self.end_date and next_fire_time > self.end_date:
            return None

        return next_fire_time

    def __str__(self):
        return "cron[%s %s %s %s %s]" % (
            self.minute or "*",
            self.hour or "*",
            self.day or "*",
            self.month or "*",
            self.day_of_week or "*",
        )
