# apscheduler/util.py
import time
from typing import Any, Optional

TIMEOUT_MAX = 4294967.0  # Maximum timeout value for various operations


def maybe_ref(obj):
    """Return the object itself if it's not a string, otherwise try to import it."""
    if isinstance(obj, str):
        module, _, attr = obj.rpartition(".")
        try:
            return getattr(__import__(module, fromlist=[attr]), attr)
        except (ImportError, AttributeError):
            raise ImportError("Cannot import %s" % obj)
    return obj


def timedelta_seconds(delta):
    """Convert a timedelta to seconds."""
    return delta.days * 24 * 60 * 60 + delta.seconds + delta.microseconds / 1000000.0


def to_unicode(obj):
    """Convert an object to unicode."""
    if isinstance(obj, str):
        return obj
    try:
        return str(obj)
    except UnicodeDecodeError:
        return obj.decode("utf-8", "replace")
