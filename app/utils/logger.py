import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import structlog

# Configure logging before importing other modules
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = Path("logs")
LOG_FILE = "application.log"
MAX_BYTES = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5


def setup_logging():
    """Configure structured logging with structlog."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ]

    # Configure logging handlers with proper error handling and rotation
    try:
        logging.basicConfig(
            level=LOG_LEVEL,
            handlers=[
                logging.StreamHandler(sys.stdout),
                RotatingFileHandler(
                    LOG_DIR / LOG_FILE,
                    maxBytes=MAX_BYTES,
                    backupCount=BACKUP_COUNT,
                    encoding="utf-8",
                    delay=True,  # Delay file opening until first log
                ),
            ],
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    except Exception as e:
        # Use structlog's native error logging with full context
        structlog.get_logger().error(
            "Failed to configure logging handlers",
            error=str(e),
            exc_info=True,
            log_level=LOG_LEVEL,
            log_file=LOG_FILE,
            max_bytes=MAX_BYTES,
            backup_count=BACKUP_COUNT,
        )
        raise

    structlog.configure(
        processors=processors,
        context_class=dict,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()


# Initialize logger
logger = setup_logging()
