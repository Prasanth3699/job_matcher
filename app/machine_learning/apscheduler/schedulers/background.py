# apscheduler/schedulers/background.py
import threading
from typing import Any, Optional
from datetime import datetime
from ..schedulers.base import BaseScheduler
from ..util import maybe_ref, TIMEOUT_MAX
from ..job import Job


class BackgroundScheduler(BaseScheduler):
    """
    A scheduler that runs in the background using a separate thread.
    """

    def __init__(self, **kwargs):
        super(BackgroundScheduler, self).__init__(**kwargs)
        self._thread: Optional[threading.Thread] = None
        self._event = threading.Event()

    @property
    def running(self) -> bool:
        """Return whether the scheduler is running."""
        return getattr(self, "_running", False)

    def _configure(self, **kwargs):
        self._daemon = kwargs.pop("daemon", False)
        super(BackgroundScheduler, self)._configure(**kwargs)

    def start(self, paused=False):
        if self.running:
            raise RuntimeError("Scheduler is already running")

        super(BackgroundScheduler, self).start(paused)
        self._event.clear()
        self._thread = threading.Thread(
            target=self._main_loop, name="APScheduler", daemon=self._daemon
        )
        self._thread.start()

    def shutdown(self, wait=True):
        super(BackgroundScheduler, self).shutdown(wait)
        self._event.set()
        if wait and self._thread and self._thread.is_alive():
            self._thread.join(TIMEOUT_MAX)

    def _main_loop(self):
        while not self._event.wait(0.5):
            if not self._event.is_set():
                self._process_jobs()

    def _create_default_executor(self):
        from ..executors.pool import ThreadPoolExecutor

        return ThreadPoolExecutor()


# apscheduler/schedulers/background.py
# import threading
# from typing import Any, Optional
# from datetime import datetime
# from ..schedulers.base import BaseScheduler
# from ..util import maybe_ref, TIMEOUT_MAX
# from ..job import Job


# class BackgroundScheduler(BaseScheduler):
#     """
#     A scheduler that runs in the background using a separate thread.
#     """

#     def __init__(self, **kwargs):
#         super(BackgroundScheduler, self).__init__(**kwargs)
#         self._thread: Optional[threading.Thread] = None
#         self._event = threading.Event()

#     def _configure(self, **kwargs):
#         self._daemon = kwargs.pop("daemon", False)
#         super(BackgroundScheduler, self)._configure(**kwargs)

#     def start(self, paused=False):
#         if self._thread is not None and self._thread.is_alive():
#             raise RuntimeError("Scheduler is already running")

#         super(BackgroundScheduler, self).start(paused)
#         self._event.clear()
#         self._thread = threading.Thread(target=self._main_loop, name="APScheduler")
#         self._thread.daemon = self._daemon
#         self._thread.start()

#     def shutdown(self, wait=True):
#         super(BackgroundScheduler, self).shutdown(wait)
#         self._event.set()
#         if wait and self._thread and self._thread.is_alive():
#             self._thread.join(TIMEOUT_MAX)

#     def _main_loop(self):
#         while not self._event.wait(0.5):
#             if not self._event.is_set():
#                 self._process_jobs()

#     def _create_default_executor(self):
#         from apscheduler.executors.pool import ThreadPoolExecutor

#         return ThreadPoolExecutor()

#     def _create_default_jobstore(self):
#         from .jobstores.memory import MemoryJobStore

#         return MemoryJobStore()
