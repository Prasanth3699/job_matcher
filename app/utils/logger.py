# resume_matcher/utils/logger.py
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
import json
from datetime import datetime
import structlog


def setup_logging(log_dir: Path = Path("logs")):
    """Configure structured logging for the application"""
    log_dir.mkdir(exist_ok=True)

    # Standard library logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            RotatingFileHandler(
                log_dir / "resume_matcher.log",
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Structlog configuration
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()


# Initialize logger
logger = setup_logging()
