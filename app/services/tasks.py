from datetime import datetime, timedelta
from pathlib import Path
import subprocess
from app.data.database import db
from app.data.repositories import (
    FeedbackRepository,
    ModelVersionRepository,
    BackupRepository,
)
from app.core.learning.model_updater import ModelUpdater
from app.core.learning.performance_monitor import PerformanceMonitor
from app.utils.logger import logger


class ScheduledTasks:
    """Manager for scheduled background tasks"""

    def __init__(self):
        self.model_updater = ModelUpdater()
        self.performance_monitor = PerformanceMonitor()

    def retrain_model_if_needed(self):
        """Check if model needs retraining and execute if needed"""
        with db.session_scope() as session:
            feedback_repo = FeedbackRepository(session)
            model_repo = ModelVersionRepository(session)

            # Check if we have enough new feedback
            feedback_count = len(
                feedback_repo.get_feedback_for_training(days=7, sample_size=100)
            )
            if feedback_count < 100:
                logger.info(
                    f"Not enough feedback for retraining: {feedback_count} samples"
                )
                return None

            # Check current model performance
            current_version = model_repo.get_active_version()
            if current_version:
                alert = self.performance_monitor.detect_performance_drop(
                    current_version.version_name, threshold=0.1
                )
                if not alert:
                    logger.info("No performance drop detected, skipping retraining")
                    return None

            # Proceed with retraining
            logger.info("Starting model retraining...")
            try:
                new_version = self.model_updater.retrain_model(
                    feedback_repo, epochs=3, train_size=0.8
                )

                # Save new version to database
                db_version = model_repo.create_version(
                    version_name=new_version.version_id,
                    model_type="hybrid",
                    storage_path=str(
                        self.model_updater.model_storage / new_version.version_id
                    ),
                    metrics=new_version.metrics,
                    description=new_version.description,
                )

                # Activate the new version
                model_repo.activate_version(db_version.id)

                # Record performance metrics
                self.performance_monitor.record_metrics(
                    new_version.version_id, new_version.metrics
                )

                logger.info(
                    f"Model retraining completed. New version: {new_version.version_id}"
                )
                return new_version

            except Exception as e:
                logger.error(f"Model retraining failed: {str(e)}")
                raise

    def backup_database(self):
        """Perform database backup"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"backups/resume_matcher_{timestamp}.dump"

        with db.session_scope() as session:
            backup_repo = BackupRepository(session)

            # Log backup start
            backup_repo.log_backup(
                backup_type="full", file_path=backup_file, status="started"
            )

            try:
                # Execute pg_dump (PostgreSQL specific)
                result = subprocess.run(
                    [
                        "pg_dump",
                        "-Fc",  # Custom format
                        "-f",
                        backup_file,
                        db._engine.url.database,
                    ],
                    check=True,
                    capture_output=True,
                )

                # Log successful completion
                backup_repo.log_backup(
                    backup_type="full",
                    file_path=backup_file,
                    status="completed",
                    file_size=Path(backup_file).stat().st_size,
                )

                logger.info(f"Database backup completed: {backup_file}")
                return backup_file

            except subprocess.CalledProcessError as e:
                logger.error(f"Database backup failed: {e.stderr.decode()}")
                backup_repo.log_backup(
                    backup_type="full", file_path=backup_file, status="failed"
                )
                raise

    def clean_old_backups(self, days_to_keep: int = 7):
        """Clean up old backup files"""
        cutoff = datetime.now() - timedelta(days=days_to_keep)
        backup_dir = Path("backups")

        if not backup_dir.exists():
            return

        for backup_file in backup_dir.glob("*.dump"):
            if backup_file.stat().st_mtime < cutoff.timestamp():
                try:
                    backup_file.unlink()
                    logger.info(f"Deleted old backup: {backup_file}")
                except Exception as e:
                    logger.warning(
                        f"Failed to delete old backup {backup_file}: {str(e)}"
                    )
