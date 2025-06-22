import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import structlog


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "application.log"
ERROR_LOG_FILE = LOG_DIR / "error.log"
MAX_BYTES = 50 * 1024 * 1024  # 50 MB
BACKUP_COUNT = 14  # Two weeks of logs if you rotate daily


def configure_logging():
    """Configure rich, production-ready logging and structlog."""
    # Plain formatter for console/file
    plain_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(processName)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Detailed formatter for errors
    error_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(processName)s | %(name)s | %(pathname)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handlers
    handlers = []

    # Console (stdout) handler, for all levels
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(plain_formatter)
    handlers.append(console_handler)

    # Rotating file handler for all logs
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(plain_formatter)
    handlers.append(file_handler)

    # Rotating file handler for errors only
    error_file_handler = RotatingFileHandler(
        ERROR_LOG_FILE,
        maxBytes=MAX_BYTES // 2,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(error_formatter)
    handlers.append(error_file_handler)

    # Root logger config
    logging.basicConfig(level=LOG_LEVEL, handlers=handlers)

    # Silence third-party noise if needed
    for noisy in ("urllib3", "httpx", "botocore", "asyncio", "chardet"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ---- structlog configuration (JSON output great for ingestion) ---- #
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # Choose one renderer!
            # structlog.dev.ConsoleRenderer(colors=True),   # for dev
            structlog.processors.JSONRenderer(),  # for prod
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger("myapp")


# --- Outside the function, so importers get a logger ---
logger = configure_logging()
