# apscheduler/triggers/base.py
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional


class BaseTrigger(ABC):
    """Abstract base class for all triggers."""

    @abstractmethod
    def get_next_fire_time(
        self, previous_fire_time: Optional[datetime], now: datetime
    ) -> Optional[datetime]:
        """Return the next datetime to fire on."""
        pass

    def __str__(self):
        return self.__class__.__name__
