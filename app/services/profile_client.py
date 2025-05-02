import httpx, json
from ..core.config import get_settings
from typing import Optional, Dict, Any
from app.utils.logger import logger


settings = get_settings()


class ProfileServiceClient:

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
                        json.dumps(parsed_data or {}),
                    ),  # Ensure empty dict if None
                    "metadata": (
                        None,
                        json.dumps(metadata or {}),
                    ),  # Ensure empty dict if None
                }
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
