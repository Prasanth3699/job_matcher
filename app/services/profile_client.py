import httpx, json
from datetime import datetime
from ..core.config import get_settings
from typing import Optional, Dict, Any
from app.utils.logger import logger


settings = get_settings()


class ProfileServiceClient:

    @staticmethod
    def _serialize_json_safe(data: Dict[str, Any]) -> str:
        """
        Safely serialize data to JSON, handling datetime objects and other non-serializable types.
        """

        def json_serial(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, "__dict__"):
                return obj.__dict__
            else:
                return str(obj)

        try:
            return json.dumps(data or {}, default=json_serial)
        except Exception as e:
            logger.warning(f"JSON serialization failed, using fallback: {e}")
            # Fallback: convert all values to strings
            safe_data = {}
            for key, value in (data or {}).items():
                try:
                    safe_data[key] = json.dumps(value, default=json_serial)
                except:
                    safe_data[key] = str(value)
            return json.dumps(safe_data)

    @staticmethod
    async def push_parsed_resume(
        token: str,
        filename: str,
        file_bytes: bytes,
        raw_text: str,
        parsed_data: dict,
        metadata: dict,
    ) -> Optional[Dict[str, Any]]:
        """
        Uploads the file and its parsed data, returning the response JSON on success.
        Returns None on failure or exception.
        """
        try:
            async with httpx.AsyncClient(base_url=settings.JOBS_SERVICE_URL) as cl:
                files = {
                    "file": (filename, file_bytes, "application/octet-stream"),
                    "raw_text": (None, raw_text),
                    "parsed_data": (
                        None,
                        ProfileServiceClient._serialize_json_safe(parsed_data),
                    ),
                    "metadata": (
                        None,
                        ProfileServiceClient._serialize_json_safe(metadata),
                    ),
                }
                print("--------------------", files)
                resp = await cl.post(
                    "/api/v1/parsed-resume/upload-parsed-resume",
                    files=files,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=20.0,
                )
            if resp.status_code in (200, 201):
                try:
                    # Successfully uploaded, return the response body as a dict
                    return resp.json()
                except json.JSONDecodeError as json_err:
                    logger.error(
                        f"[ProfileService] push_parsed_resume failed to decode JSON response: {json_err}. Response text: {resp.text[:200]}"
                    )
                    return None  # Indicate failure if JSON is bad
            else:
                logger.warning(
                    f"[ProfileService] push_parsed_resume failed "
                    f"({resp.status_code}): {resp.text[:200]}"
                )
                return None  # Indicate failure
        except httpx.RequestError as req_err:
            logger.error(
                f"[ProfileService] request error during push_parsed_resume: {req_err}"
            )
            return None  # Indicate failure
        except Exception as exc:
            logger.error(
                f"[ProfileService] exception during push_parsed_resume: {exc}",
                exc_info=True,
            )  # Add exc_info for more details
            return None  # Indicate failure

    @staticmethod
    def push_parsed_resume_sync(
        token: str,
        filename: str,
        file_bytes: bytes,
        raw_text: str,
        parsed_data: dict,
        metadata: dict,
    ) -> Optional[Dict[str, Any]]:
        """
        Synchronous version for Celery workers to avoid asyncio event loop conflicts.
        Uploads the file and its parsed data, returning the response JSON on success.
        Returns None on failure or exception.
        """
        try:
            with httpx.Client(base_url=settings.JOBS_SERVICE_URL) as cl:
                files = {
                    "file": (filename, file_bytes, "application/octet-stream"),
                    "raw_text": (None, raw_text),
                    "parsed_data": (
                        None,
                        ProfileServiceClient._serialize_json_safe(parsed_data),
                    ),
                    "metadata": (
                        None,
                        ProfileServiceClient._serialize_json_safe(metadata),
                    ),
                }
                resp = cl.post(
                    "/api/v1/parsed-resume/upload-parsed-resume",
                    files=files,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=20.0,
                )
            if resp.status_code in (200, 201):
                try:
                    # Successfully uploaded, return the response body as a dict
                    return resp.json()
                except json.JSONDecodeError as json_err:
                    logger.error(
                        f"[ProfileService] push_parsed_resume_sync failed to decode JSON response: {json_err}. Response text: {resp.text[:200]}"
                    )
                    return None  # Indicate failure if JSON is bad
            else:
                logger.warning(
                    f"[ProfileService] push_parsed_resume_sync failed "
                    f"({resp.status_code}): {resp.text[:200]}"
                )
                return None  # Indicate failure
        except httpx.RequestError as req_err:
            logger.error(
                f"[ProfileService] request error during push_parsed_resume_sync: {req_err}"
            )
            return None  # Indicate failure
        except Exception as exc:
            logger.error(
                f"[ProfileService] exception during push_parsed_resume_sync: {exc}",
                exc_info=True,
            )  # Add exc_info for more details
            return None  # Indicate failure
