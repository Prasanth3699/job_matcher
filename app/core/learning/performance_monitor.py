# resume_matcher/core/learning/performance_monitor.py
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from ...utils.logger import logger
from .models import ModelVersion


class PerformanceMonitor:
    """
    Tracks model performance metrics over time and detects anomalies.
    """

    def __init__(self, storage_path: Path = Path("performance_metrics")):
        self.storage_path = storage_path
        self.storage_path.mkdir(exist_ok=True)

    def record_metrics(
        self,
        version_id: str,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record performance metrics for a model version"""
        if not timestamp:
            timestamp = datetime.now()

        record = {
            "version_id": version_id,
            "timestamp": timestamp.isoformat(),
            "metrics": metrics,
        }

        file_path = self.storage_path / f"{timestamp.date()}_{version_id}.json"
        with open(file_path, "w") as f:
            json.dump(record, f)

    def get_metrics_history(
        self, version_id: str, days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get performance metrics history for a version"""
        cutoff_date = datetime.now() - timedelta(days=days)
        metrics_files = self.storage_path.glob(f"*_{version_id}.json")
        history = []

        for file in metrics_files:
            try:
                file_date = datetime.strptime(
                    file.name.split("_")[0], "%Y-%m-%d"
                ).date()
                if file_date >= cutoff_date.date():
                    with open(file, "r") as f:
                        history.append(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to read metrics file {file}: {str(e)}")

        return sorted(history, key=lambda x: x["timestamp"])

    def detect_performance_drop(
        self, version_id: str, threshold: float = 0.1
    ) -> Optional[Dict[str, Any]]:
        """
        Detect significant performance drops.

        Args:
            version_id: Model version to check
            threshold: Relative drop threshold to trigger alert

        Returns:
            Alert details if drop detected, None otherwise
        """
        history = self.get_metrics_history(version_id)
        if len(history) < 2:
            return None

        # Get baseline metrics (median of first week)
        baseline_period = [
            h
            for h in history
            if (datetime.now() - datetime.fromisoformat(h["timestamp"])).days > 7
        ]
        if not baseline_period:
            baseline_period = history[:7]

        baseline = {}
        for metric in baseline_period[0]["metrics"].keys():
            values = [
                h["metrics"][metric] for h in baseline_period if metric in h["metrics"]
            ]
            baseline[metric] = np.median(values)

        # Compare with recent performance (last 3 days)
        recent_period = [
            h
            for h in history
            if (datetime.now() - datetime.fromisoformat(h["timestamp"])).days <= 3
        ]
        if not recent_period:
            return None

        recent = {}
        for metric in recent_period[0]["metrics"].keys():
            values = [
                h["metrics"][metric] for h in recent_period if metric in h["metrics"]
            ]
            recent[metric] = np.median(values)

        # Check for significant drops
        alerts = {}
        for metric in baseline:
            if metric in recent:
                drop = (baseline[metric] - recent[metric]) / baseline[metric]
                if drop > threshold:
                    alerts[metric] = {
                        "baseline": baseline[metric],
                        "current": recent[metric],
                        "drop_pct": drop * 100,
                    }

        if alerts:
            return {
                "version_id": version_id,
                "timestamp": datetime.now().isoformat(),
                "alerts": alerts,
                "suggestion": "Consider retraining the model",
            }

        return None

    def generate_performance_report(
        self, version_id: str, days: int = 30
    ) -> Dict[str, Any]:
        """Generate a comprehensive performance report"""
        history = self.get_metrics_history(version_id, days)
        if not history:
            return {}

        # Convert to DataFrame
        df = pd.DataFrame(
            [
                {"date": datetime.fromisoformat(h["timestamp"]).date(), **h["metrics"]}
                for h in history
            ]
        )

        # Calculate stats
        report = {
            "version_id": version_id,
            "time_period": f"Last {days} days",
            "metrics": {},
        }

        for metric in df.columns[1:]:
            report["metrics"][metric] = {
                "mean": float(df[metric].mean()),
                "median": float(df[metric].median()),
                "min": float(df[metric].min()),
                "max": float(df[metric].max()),
                "trend": self._calculate_trend(df, metric),
                "last_value": float(df[metric].iloc[-1]),
            }

        return report

    def _calculate_trend(self, df: pd.DataFrame, metric: str) -> float:
        """Calculate trend line slope for a metric"""
        if len(df) < 2:
            return 0.0

        x = np.arange(len(df))
        y = df[metric].values
        slope = np.polyfit(x, y, 1)[0]
        return float(slope / np.mean(y) if np.mean(y) != 0 else 0.0)
