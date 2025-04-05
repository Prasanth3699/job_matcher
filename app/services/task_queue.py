from typing import Dict
from celery import Celery
from kombu import Queue, Exchange
from app.utils.logger import logger
from app.data.database import db
from app.core.learning import ModelUpdater, PerformanceMonitor
from app.data.repositories import FeedbackRepository, ModelVersionRepository

# Configure Celery
app = Celery(
    "resume_matcher",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

# Custom task queue configuration
app.conf.task_queues = (
    Queue("default", Exchange("default"), routing_key="default"),
    Queue("model_training", Exchange("model_training"), routing_key="model.training"),
    Queue("analysis", Exchange("analysis"), routing_key="analysis"),
)

app.conf.task_routes = {
    "retrain_model": {"queue": "model_training"},
    "analyze_resume": {"queue": "analysis"},
}

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
)


@app.task(bind=True, max_retries=3)
def retrain_model(self, days_of_feedback: int = 30):
    """Celery task for model retraining"""
    try:
        with db.session_scope() as session:
            feedback_repo = FeedbackRepository(session)
            model_repo = ModelVersionRepository(session)
            model_updater = ModelUpdater()
            performance_monitor = PerformanceMonitor()

            # Training logic
            new_version = model_updater.retrain_model(
                feedback_repo, epochs=3, train_size=0.8
            )

            # Save and activate new version
            db_version = model_repo.create_version(
                version_name=new_version.version_id,
                model_type="hybrid",
                storage_path=str(model_updater.model_storage / new_version.version_id),
                metrics=new_version.metrics,
                description=new_version.description,
            )

            model_repo.activate_version(db_version.id)
            performance_monitor.record_metrics(
                new_version.version_id, new_version.metrics
            )

            return {
                "status": "success",
                "version_id": new_version.version_id,
                "metrics": new_version.metrics,
            }

    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        self.retry(exc=e, countdown=60)


@app.task(bind=True)
def analyze_resume(self, resume_data: Dict, job_data: Dict = None):
    """Celery task for comprehensive resume analysis"""
    try:
        from app.core.analysis.gap_analyzer import GapAnalyzer

        analyzer = GapAnalyzer()

        # Work history analysis
        work_history = resume_data.get("experiences", [])
        gap_analysis = analyzer.analyze_work_history(work_history)

        # Skill gap analysis (if job data provided)
        skill_analysis = {"missing_skills": [], "categorized_skills": {}}
        if job_data:
            resume_skills = resume_data.get("skills", [])
            job_skills = job_data.get("skills", [])
            skill_analysis = analyzer.analyze_skill_gaps(resume_skills, job_skills)

        # Generate recommendations
        recommendations = analyzer.generate_recommendations(
            gap_analysis, skill_analysis
        )

        return {
            "gap_analysis": gap_analysis,
            "skill_analysis": skill_analysis,
            "recommendations": recommendations,
        }

    except Exception as e:
        logger.error(f"Resume analysis failed: {str(e)}")
        return {"status": "error", "error": str(e)}


def start_workers():
    """Start Celery workers programmatically (for development)"""
    from celery.bin import worker

    worker = worker.worker(app=app)
    worker.run(loglevel="INFO")
