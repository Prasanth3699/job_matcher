import atexit
from typing import Optional
from app.machine_learning.apscheduler.jobstores.memory import MemoryJobStore
from app.machine_learning.apscheduler.schedulers.background import BackgroundScheduler
from app.machine_learning.apscheduler.triggers.interval import IntervalTrigger
from app.machine_learning.apscheduler.triggers.cron import CronTrigger
from app.machine_learning.apscheduler.jobstores.base import JobLookupError
from app.machine_learning.apscheduler.executors.pool import ThreadPoolExecutor
from .tasks import ScheduledTasks
from app.config.settings import settings
from app.utils.logger import logger


class TaskScheduler:
    """Manages scheduled background tasks with enhanced reliability"""

    def __init__(self):
        if not hasattr(settings, "SCHEDULER_JOBSTORE"):
            raise ValueError("Missing SCHEDULER_JOBSTORE in settings")
        if not hasattr(settings, "timezone_obj"):
            raise ValueError("Missing timezone_obj in settings")

        self.tasks = ScheduledTasks()

        # Configure job store and executors
        jobstores = {"default": MemoryJobStore()}

        try:
            if hasattr(settings, "SCHEDULER_JOBSTORE"):
                if settings.SCHEDULER_JOBSTORE == "sqlalchemy":
                    from app.machine_learning.apscheduler.jobstores.sqlalchemy import (
                        SQLAlchemyJobStore,
                    )

                    jobstores["default"] = SQLAlchemyJobStore(url=settings.DATABASE_URL)
                elif settings.SCHEDULER_JOBSTORE == "mongodb":
                    from app.machine_learning.apscheduler.jobstores.mongodb import (
                        MongoDBJobStore,
                    )

                    jobstores["default"] = MongoDBJobStore(
                        database="resume_matcher",
                        collection="jobs",
                        host=settings.DATABASE_URL,
                    )
        except ImportError as e:
            logger.warning(f"Could not initialize configured jobstore: {e}")
            logger.warning("Falling back to MemoryJobStore")
            jobstores["default"] = MemoryJobStore()

        executors = {
            "default": ThreadPoolExecutor(
                max_workers=getattr(settings, "SCHEDULER_MAX_WORKERS", 20)
            )
        }

        job_defaults = {
            "coalesce": settings.SCHEDULER_COALESCE,
            "max_instances": settings.SCHEDULER_MAX_INSTANCES,
            "misfire_grace_time": settings.SCHEDULER_MISFIRE_GRACE_TIME,
        }

        self.scheduler = BackgroundScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone=settings.timezone_obj,
        )
        self.scheduler._daemon = True  # Set daemon flag directly
        self._register_shutdown_hook()

    def _register_shutdown_hook(self):
        """Register shutdown handler for proper cleanup"""
        atexit.register(self.shutdown)

    def start(self):
        """Start the scheduler with configured jobs"""
        try:
            if not hasattr(self, "tasks"):
                raise AttributeError(
                    "TaskScheduler is missing required 'tasks' attribute"
                )

            if not self.scheduler.running:
                # Verify and initialize jobstores properly
                for alias, store in self.scheduler.jobstores.items():
                    if store is None:
                        raise ValueError(
                            f"Job store '{alias}' is not properly initialized"
                        )
                    # Correctly call start() with both arguments
                    store.start(
                        self.scheduler, alias
                    )  # Fixed: passing both scheduler and alias

                self._configure_jobs()
                self.scheduler.start()
                logger.info("Task scheduler started successfully")
            else:
                logger.warning("Scheduler is already running")
        except Exception as e:
            logger.error(f"Failed to start scheduler: {str(e)}", exc_info=True)
            raise RuntimeError("Scheduler startup failed") from e

    def _configure_jobs(self):
        """Configure all scheduled jobs with proper error handling"""
        try:
            # Model retraining - daily at 2 AM
            self._add_cron_job(
                func=self.tasks.retrain_model_if_needed,
                hour=2,
                name="daily_model_retraining",
                description="Retrain ML models if new data is available",
            )

            # Database backup - daily at 1 AM
            self._add_cron_job(
                func=self.tasks.backup_database,
                hour=1,
                name="daily_database_backup",
                description="Create database backup",
            )

            # Backup cleanup - weekly on Sunday at 3 AM
            self._add_cron_job(
                func=self.tasks.clean_old_backups,
                day_of_week="sun",
                hour=3,
                name="weekly_backup_cleanup",
                description="Remove old backup files",
            )

            # Performance monitoring - hourly
            self._add_interval_job(
                func=self.monitor_performance,
                hours=1,
                name="hourly_performance_monitoring",
                description="System performance metrics collection",
            )

            # Optional: Add health check job if needed
            if settings.ENABLE_HEALTH_CHECKS:
                self._add_interval_job(
                    func=self.check_system_health,
                    minutes=30,
                    name="system_health_check",
                    description="System health status check",
                )

        except Exception as e:
            logger.error(f"Failed to configure jobs: {str(e)}")
            raise

    def _add_cron_job(self, func, name: str, description: str = "", **trigger_args):
        """Safely add a cron-scheduled job"""
        try:
            self.scheduler.add_job(
                func=func,
                trigger=CronTrigger(**trigger_args),
                name=name,
                description=description,
                replace_existing=True,
            )
            logger.info(f"Added cron job: {name}")
        except Exception as e:
            logger.error(f"Failed to add cron job {name}: {str(e)}")
            raise

    def _add_interval_job(self, func, name: str, description: str = "", **trigger_args):
        """Safely add an interval-based job"""
        try:
            self.scheduler.add_job(
                func=func,
                trigger=IntervalTrigger(**trigger_args),
                name=name,
                description=description,
                replace_existing=True,
            )
            logger.info(f"Added interval job: {name}")
        except Exception as e:
            logger.error(f"Failed to add interval job {name}: {str(e)}")
            raise

    def monitor_performance(self):
        """Check system performance and log metrics"""
        try:
            # Implement actual performance monitoring logic here
            logger.info("Performance monitoring check completed")
        except Exception as e:
            logger.error(f"Performance monitoring failed: {str(e)}")

    def check_system_health(self):
        """Check critical system components"""
        try:
            # Implement health check logic
            logger.info("System health check completed")
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")

    def shutdown(self):
        """Gracefully shutdown the scheduler"""
        try:
            if hasattr(self.scheduler, "running") and self.scheduler.running:
                self.scheduler.shutdown(wait=True)
                logger.info("Task scheduler stopped gracefully")
        except Exception as e:
            logger.error(f"Error during scheduler shutdown: {str(e)}")
            raise

    def get_job_status(self, job_name: str) -> Optional[dict]:
        """Get status information for a specific job"""
        try:
            job = self.scheduler.get_job(job_name)
            if job:
                return {
                    "name": job.name,
                    "next_run": job.next_run_time,
                    "enabled": not job.pending,
                }
            return None
        except JobLookupError:
            return None
        except Exception as e:
            logger.error(f"Failed to get job status for {job_name}: {str(e)}")
            raise

    def list_jobs(self) -> list:
        """List all scheduled jobs with their status"""
        try:
            return [
                {
                    "id": job.id,
                    "name": job.name,
                    "trigger": str(job.trigger),
                    "next_run": job.next_run_time,
                    "description": job.description,
                }
                for job in self.scheduler.get_jobs()
            ]
        except Exception as e:
            logger.error(f"Failed to list jobs: {str(e)}")
            raise
