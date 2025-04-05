from prometheus_client import REGISTRY, Counter, Gauge, Summary, start_http_server
from typing import Dict, Any, Set
from app.utils.logger import logger


class MetricsManager:
    """Handles Prometheus metrics registration and management"""

    _instance = None
    _registered_metrics: Set[str] = set()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.metrics: Dict[str, Any] = {}
        self._setup_metrics()

    def _unregister_metric(self, name: str):
        """Safely unregister a metric and all its related time series"""
        try:
            # Unregister all variants of the metric name
            for metric_name in list(REGISTRY._names_to_collectors.keys()):
                if (
                    metric_name == name
                    or metric_name.startswith(f"{name}_")
                    or metric_name.endswith(f"_{name}")
                ):
                    try:
                        REGISTRY.unregister(REGISTRY._names_to_collectors[metric_name])
                        if metric_name in self._registered_metrics:
                            self._registered_metrics.remove(metric_name)
                    except Exception as e:
                        logger.warning(
                            f"Failed to unregister metric {metric_name}: {str(e)}"
                        )
        except Exception as e:
            logger.warning(f"Error during metric cleanup for {name}: {str(e)}")

    def _setup_metrics(self):
        """Initialize all metrics with proper registration"""
        # First unregister all possible metrics
        for name in [
            "jobs_in_database",
            "vector_db_size",
            "resumes_processed",
            "requests",
            "matches_processed",
            "match_duration",
            "match_score",
        ]:
            self._unregister_metric(name)

        # Define metrics without manual prefixing
        metrics_def = [
            # System metrics
            ("jobs_in_database", Gauge, "Number of jobs in database"),
            ("vector_db_size", Gauge, "Number of vectors in vector database"),
            ("resumes_processed", Counter, "Total resumes processed"),
            # Matching metrics
            ("requests", Counter, "Total API requests", ["endpoint", "method"]),
            ("matches_processed", Counter, "Total resume-job matches processed"),
            ("match_duration", Summary, "Time spent processing matches"),
            ("match_score", Gauge, "Match score distribution", ["job_type"]),
        ]

        for metric in metrics_def:
            name = metric[0]
            metric_type = metric[1]
            help_text = metric[2]
            labels = metric[3] if len(metric) > 3 else []

            try:
                # Create metric with just the base name
                if metric_type == Counter:
                    self.metrics[name] = metric_type(
                        name,
                        help_text,
                        labels,
                        namespace="resume_matcher",
                    )
                elif metric_type == Gauge:
                    self.metrics[name] = metric_type(
                        name,
                        help_text,
                        labels,
                        namespace="resume_matcher",
                    )
                elif metric_type == Summary:
                    self.metrics[name] = metric_type(
                        name, help_text, namespace="resume_matcher"
                    )

                # Register the metric
                REGISTRY.register(self.metrics[name])
                self._registered_metrics.add(name)
                logger.debug(f"Registered metric: {name}")

            except ValueError as e:
                logger.error(f"Failed to register metric {metric_type}: {str(e)}")
                continue

    def update_metric(self, name: str, value: Any, labels: Dict[str, str] = None):
        """Update a metric value"""
        if name not in self.metrics:
            logger.error(f"Metric {name} not found")
            return

        metric = self.metrics[name]

        try:
            if isinstance(metric, Gauge):
                if labels:
                    metric.labels(**labels).set(value)
                else:
                    metric.set(value)
            elif isinstance(metric, Counter):
                if labels:
                    metric.labels(**labels).inc(value)
                else:
                    metric.inc(value)
            elif isinstance(metric, Summary):
                metric.observe(value)
        except Exception as e:
            logger.error(f"Failed to update metric {name}: {str(e)}")

    def start_metrics_server(self, port: int = 8001):
        """Start the metrics server"""
        try:
            start_http_server(port)
            logger.info(f"Metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {str(e)}")
