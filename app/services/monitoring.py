from .metrics_manager import MetricsManager
from app.utils.logger import logger


class Monitoring:
    """Handles system monitoring and metrics collection"""

    def __init__(self, port: int = 8002):
        self.metrics = MetricsManager()
        self.port = port

    def start_metrics_server(self):
        """Start the metrics server"""
        self.metrics.start_metrics_server(self.port)
        logger.info(f"Metrics server started on port {self.port}")

    def update_system_metrics(
        self, job_count: int, resume_count: int, vector_db_size: int
    ):
        """Update system resource metrics"""
        self.metrics.update_metric("jobs_in_database", job_count)
        self.metrics.update_metric("vector_db_size", vector_db_size)
        self.metrics.update_metric("resumes_processed", resume_count)
        logger.debug(
            f"Updated system metrics - jobs: {job_count}, vectors: {vector_db_size}"
        )

    def record_match(
        self,
        endpoint: str,
        method: str,
        duration: float,
        score: float,
        job_type: str = None,
    ):
        """Record match processing metrics"""
        self.metrics.update_metric(
            "requests_total", 1, {"endpoint": endpoint, "method": method}
        )
        self.metrics.update_metric("matches_processed", 1)
        self.metrics.update_metric("match_duration", duration)
        if score is not None:
            self.metrics.update_metric(
                "match_score", score, {"job_type": job_type or "unknown"}
            )
