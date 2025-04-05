import hashlib
import tempfile
from pathlib import Path
from typing import Optional
from .logger import logger


class SecurityUtils:
    """Handles security-related utilities"""

    @staticmethod
    def calculate_file_hash(
        file_path: Path, algorithm: str = "sha256"
    ) -> Optional[str]:
        """Calculate file hash for verification"""
        try:
            hash_func = getattr(hashlib, algorithm)()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate file hash: {str(e)}")
            return None

    @staticmethod
    def create_secure_temp_file(content: bytes) -> Path:
        """Create a secure temporary file with the given content"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)

            # Set restrictive permissions
            tmp_path.chmod(0o600)
            return tmp_path
        except Exception as e:
            logger.error(f"Failed to create secure temp file: {str(e)}")
            raise

    @staticmethod
    def safe_file_operations(file_path: Path) -> bool:
        """Verify file is safe for operations"""
        try:
            # Check file permissions
            if file_path.stat().st_mode & 0o777 > 0o600:
                logger.warning(
                    f"Insecure file permissions: {oct(file_path.stat().st_mode)}"
                )
                return False

            # Check file owner (simplified)
            import os

            if os.name == "posix" and file_path.owner() != os.getlogin():
                logger.warning(f"File owned by different user: {file_path.owner()}")
                return False

            return True
        except Exception as e:
            logger.error(f"File safety check failed: {str(e)}")
            return False
