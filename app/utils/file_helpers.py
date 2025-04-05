from pathlib import Path
from app.utils.logger import logger


def cleanup_temp_file(path: Path):
    """Handle temporary file cleanup with logging"""
    try:
        if path.exists():
            path.unlink()
            logger.debug(f"Successfully cleaned up temp file: {path}")
    except Exception as e:
        logger.warning(f"Failed to clean up temp file {path}: {str(e)}")
