"""
Job service for fetching and processing job data from external services.
Implements RabbitMQ-first approach with HTTP fallback for reliable data retrieval.
"""

import asyncio
import json
from typing import List, Dict, Any
import re
from bs4 import BeautifulSoup
import httpx
import aio_pika
from aio_pika import connect, Message, DeliveryMode, ExchangeType
from contextlib import asynccontextmanager

from app.services.rabbitmq_client import rabbitmq_client_context
from app.utils.serialization import make_json_serializable
from app.core.constants import (
    TimeoutSettings,
    ProcessingLimits,
    BusinessRules,
    ErrorCodes,
    ErrorMessages,
)

from ..core.config import get_settings
from ..schemas.resume import Job
from app.utils.logger import logger
from app.core.events import (
    EventMessage,
    EventType,
    JobDataRequestBuilder,
    RoutingKeyGenerator,
)


settings = get_settings()


class JobDataRequestClient:
    """RabbitMQ client implementation matching the working simulation pattern"""

    def __init__(self, service_id: str = "ml-service"):
        self.service_id = service_id
        self.rabbitmq_url = settings.RABBITMQ_URL
        self.connection = None
        self.channel = None
        self.exchange = None
        self.response_queue = None
        self.response_handlers = {}

    async def connect(self):
        """Connect to RabbitMQ using the exact pattern from simulation"""
        try:
            logger.info(f"Connecting to RabbitMQ: {self.rabbitmq_url}")

            self.connection = await connect(self.rabbitmq_url)
            self.channel = await self.connection.channel()

            # Declare exchange
            self.exchange = await self.channel.declare_exchange(
                "job_scraper_events", ExchangeType.TOPIC, durable=True
            )

            # Create response queue (use unique name to avoid conflicts)
            import time

            unique_suffix = str(int(time.time()))
            self.response_queue = await self.channel.declare_queue(
                f"task.{self.service_id}.responses.{unique_suffix}",
                durable=False,
                auto_delete=True,
            )

            # Bind response queue
            await self.response_queue.bind(
                self.exchange, f"ml.{self.service_id}.job_data_response"
            )
            await self.response_queue.bind(
                self.exchange, f"ml.{self.service_id}.bulk_job_data_response"
            )

            # Start consuming responses
            await self.response_queue.consume(self._handle_response, no_ack=False)

            logger.info("Connected to RabbitMQ successfully")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    async def _handle_response(self, message: aio_pika.Message):
        """Handle response messages using simulation pattern"""
        try:
            message_data = json.loads(message.body.decode())
            event_message = EventMessage.from_dict(message_data)

            request_id = event_message.data.get("request_id")

            if request_id in self.response_handlers:
                future = self.response_handlers.pop(request_id)

                if event_message.data.get("status") == "error":
                    error_msg = event_message.data.get("error", "Unknown error")
                    future.set_exception(Exception(error_msg))
                else:
                    future.set_result(event_message.data)

            await message.ack()

        except Exception as e:
            logger.error(f"Error handling response: {e}")
            await message.nack(requeue=False)

    async def request_single_job(
        self, job_id: int, timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Request single job data using simulation pattern"""
        logger.info(f"Requesting single job: {job_id}")

        # Create request message
        event_message = JobDataRequestBuilder.create_single_job_request(
            job_id=job_id,
            ml_service_id=self.service_id,
            additional_fields=["category", "skills", "requirements"],
        )

        # Setup response handler
        request_id = event_message.data["request_id"]
        response_future = asyncio.Future()
        self.response_handlers[request_id] = response_future

        try:
            # Send message
            message = Message(
                json.dumps(event_message.to_dict()).encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
                message_id=event_message.event_id,
                correlation_id=event_message.correlation_id,
            )

            await self.exchange.publish(
                message, routing_key=RoutingKeyGenerator.for_job_data_request()
            )

            logger.info(f"Single job request sent - Request ID: {request_id}")

            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response

        except asyncio.TimeoutError:
            self.response_handlers.pop(request_id, None)
            raise TimeoutError(f"Request timeout for job_id: {job_id}")
        except Exception as e:
            self.response_handlers.pop(request_id, None)
            raise

    async def request_bulk_jobs(
        self, job_ids: List[int], timeout: float = 45.0
    ) -> Dict[str, Any]:
        """Request bulk job data using simulation pattern"""
        logger.info(f"Requesting bulk jobs: {job_ids}")

        # Create request message
        event_message = JobDataRequestBuilder.create_bulk_job_request(
            job_ids=job_ids,
            ml_service_id=self.service_id,
            additional_fields=["category", "skills", "requirements"],
        )

        # Setup response handler
        request_id = event_message.data["request_id"]
        response_future = asyncio.Future()
        self.response_handlers[request_id] = response_future

        try:
            # Send message
            message = Message(
                json.dumps(event_message.to_dict()).encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
                message_id=event_message.event_id,
                correlation_id=event_message.correlation_id,
            )

            await self.exchange.publish(
                message, routing_key=RoutingKeyGenerator.for_bulk_job_data_request()
            )

            logger.info(f"Bulk job request sent - Request ID: {request_id}")

            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response

        except asyncio.TimeoutError:
            self.response_handlers.pop(request_id, None)
            raise TimeoutError(f"Bulk request timeout for {len(job_ids)} jobs")
        except Exception as e:
            self.response_handlers.pop(request_id, None)
            raise

    async def close(self):
        """Close connection"""
        try:
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
            logger.info("Disconnected from RabbitMQ")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")


class JobService:
    """
    Service for fetching job data from external services.

    Implements RabbitMQ-first approach with HTTP fallback for reliable job data retrieval.
    Includes content cleaning and error handling for production use.
    """

    @staticmethod
    async def fetch_jobs(job_ids: List[int], token: str = None) -> List[Job]:
        """
        Fetch job details with RabbitMQ-first approach and HTTP fallback (async).
        """
        if not job_ids:
            logger.warning("No job IDs provided to fetch")
            return []

        unique_job_ids = list(dict.fromkeys(job_ids))

        # Try RabbitMQ first (async)
        try:
            jobs_data = await JobService._fetch_jobs_rabbitmq(unique_job_ids)

            jobs: List[Job] = []
            for job_data in jobs_data:
                try:
                    serializable_job_data = make_json_serializable(job_data)
                    cleaned_job = JobService._clean_job_data(serializable_job_data)
                    jobs.append(Job(**cleaned_job))
                except Exception as e:
                    logger.error(f"Error processing job data: {str(e)}")
                    continue

            if jobs:
                logger.info(f"Successfully fetched {len(jobs)} jobs via RabbitMQ")
                return jobs
            else:
                logger.warning("No valid jobs received from RabbitMQ")

        except Exception as e:
            logger.error(f"RabbitMQ fetch failed: {str(e)}")

        # Fallback to HTTP (async)
        logger.info("Falling back to HTTP requests")
        try:
            jobs = await JobService._fetch_jobs_http_fallback(unique_job_ids, token)
            if jobs:
                logger.info(f"Successfully fetched {len(jobs)} jobs via HTTP fallback")
                return jobs
        except Exception as e:
            logger.error(f"HTTP fallback failed: {str(e)}")

        raise Exception(f"No job details found for IDs: {unique_job_ids}")

    @staticmethod
    def fetch_jobs_sync(job_ids: List[int], token: str = None) -> List[Job]:
        """
        Synchronous variant used by Celery worker threads on Windows.
        RabbitMQ-first via a temporary event loop, with HTTP fallback using httpx.Client.
        """
        if not job_ids:
            logger.warning("No job IDs provided to fetch (sync)")
            return []

        unique_job_ids = list(dict.fromkeys(job_ids))

        # Try RabbitMQ first by driving the async client in a temporary loop
        try:

            def _run_async_rabbit(ids: List[int]) -> List[Dict[str, Any]]:
                # Create a dedicated event loop for this blocking call
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    return loop.run_until_complete(JobService._fetch_jobs_rabbitmq(ids))
                finally:
                    try:
                        loop.run_until_complete(asyncio.sleep(0))
                    except Exception:
                        pass
                    loop.close()

            jobs_data = _run_async_rabbit(unique_job_ids)

            jobs: List[Job] = []
            for job_data in jobs_data:
                try:
                    serializable_job_data = make_json_serializable(job_data)
                    cleaned_job = JobService._clean_job_data(serializable_job_data)
                    jobs.append(Job(**cleaned_job))
                except Exception as e:
                    logger.error(f"Error processing job data (sync): {str(e)}")
                    continue

            if jobs:
                logger.info(
                    f"Successfully fetched {len(jobs)} jobs via RabbitMQ (sync)"
                )
                return jobs
            else:
                logger.warning("No valid jobs received from RabbitMQ (sync)")

        except Exception as e:
            logger.error(f"RabbitMQ fetch failed (sync): {str(e)}")

        # HTTP fallback using synchronous client
        logger.info("Falling back to HTTP requests (sync)")
        try:
            jobs = JobService._fetch_jobs_http_fallback_sync(unique_job_ids, token)
            if jobs:
                logger.info(
                    f"Successfully fetched {len(jobs)} jobs via HTTP fallback (sync)"
                )
                return jobs
        except Exception as e:
            logger.error(f"HTTP fallback failed (sync): {str(e)}")

        raise Exception(f"No job details found for IDs: {unique_job_ids}")

    @staticmethod
    async def _fetch_jobs_rabbitmq(job_ids: List[int]) -> List[Dict]:
        """
        Fetch jobs via RabbitMQ using the simulation pattern.

        Args:
            job_ids: List of job IDs to fetch

        Returns:
            List[Dict]: List of job data dictionaries

        Raises:
            Exception: If RabbitMQ request fails
        """
        client = JobDataRequestClient()

        try:
            # Connect to RabbitMQ
            await client.connect()

            # Wait a moment for consumers to be ready
            await asyncio.sleep(1)

            # Use bulk request for multiple jobs, individual for single job
            if len(job_ids) > 1:
                logger.debug(f"Fetching {len(job_ids)} jobs via bulk RabbitMQ request")
                response = await client.request_bulk_jobs(
                    job_ids, timeout=TimeoutSettings.RABBITMQ_BULK_REQUEST_TIMEOUT
                )

                if response.get("status") == "success":
                    bulk_data = response.get("bulk_data", {})
                    jobs_data = bulk_data.get("jobs", [])
                    logger.debug(f"Received {len(jobs_data)} jobs from bulk request")
                    return jobs_data
                else:
                    logger.error(f"Bulk request failed: {response}")
                    raise Exception(
                        f"Bulk request failed: {response.get('error', 'Unknown error')}"
                    )
            else:
                logger.debug(f"Fetching single job {job_ids[0]} via RabbitMQ")
                response = await client.request_single_job(
                    job_ids[0], timeout=TimeoutSettings.RABBITMQ_REQUEST_TIMEOUT
                )

                if response.get("status") == "success":
                    job_data = response.get("job_data")
                    if job_data:
                        return [job_data]
                    else:
                        logger.warning("No job data in successful response")
                        return []
                else:
                    logger.error(f"Single job request failed: {response}")
                    raise Exception(
                        f"Single job request failed: {response.get('error', 'Unknown error')}"
                    )

        except (TimeoutError, ConnectionError) as e:
            logger.error(f"RabbitMQ communication error: {str(e)}")
            raise Exception(f"RabbitMQ request failed: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error in RabbitMQ fetch: {str(e)}")
            raise
        finally:
            await client.close()

    @staticmethod
    async def _fetch_jobs_http_fallback(
        job_ids: List[int], token: str = None
    ) -> List[Job]:
        """
        Fallback HTTP method for fetching jobs when RabbitMQ fails (async).
        """
        if not settings.JOBS_SERVICE_URL:
            logger.error("JOBS_SERVICE_URL not configured for HTTP fallback")
            return []

        jobs: List[Job] = []
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        timeout = httpx.Timeout(
            TimeoutSettings.HTTP_REQUEST_TIMEOUT,
            connect=TimeoutSettings.HTTP_CONNECT_TIMEOUT,
        )

        async with httpx.AsyncClient(
            base_url=settings.JOBS_SERVICE_URL,
            timeout=timeout,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        ) as client:

            semaphore = asyncio.Semaphore(ProcessingLimits.MAX_CONCURRENT_REQUESTS)

            async def fetch_single_job(job_id: int) -> Job | None:
                async with semaphore:
                    try:
                        response = await client.get(
                            f"/api/v1/jobs/match/{job_id}", headers=headers
                        )
                        if response.status_code == 200:
                            job_data = response.json()
                            cleaned_job = JobService._clean_job_data(job_data)
                            return Job(**cleaned_job)
                        else:
                            logger.error(
                                f"HTTP error fetching job {job_id}: Status {response.status_code}"
                            )
                            return None
                    except httpx.TimeoutException:
                        logger.error(f"Timeout fetching job {job_id}")
                        return None
                    except Exception as e:
                        logger.error(f"Exception fetching job {job_id}: {str(e)}")
                        return None

            tasks = [fetch_single_job(job_id) for job_id in job_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Job):
                    jobs.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {str(result)}")

        return jobs

    @staticmethod
    def _fetch_jobs_http_fallback_sync(
        job_ids: List[int], token: str = None
    ) -> List[Job]:
        """
        Fallback HTTP method for fetching jobs in synchronous context (Celery worker).
        """
        if not settings.JOBS_SERVICE_URL:
            logger.error("JOBS_SERVICE_URL not configured for HTTP fallback (sync)")
            return []

        jobs: List[Job] = []
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        timeout = httpx.Timeout(
            TimeoutSettings.HTTP_REQUEST_TIMEOUT,
            connect=TimeoutSettings.HTTP_CONNECT_TIMEOUT,
        )

        with httpx.Client(
            base_url=settings.JOBS_SERVICE_URL,
            timeout=timeout,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        ) as client:

            for job_id in job_ids:
                try:
                    response = client.get(
                        f"/api/v1/jobs/match/{job_id}", headers=headers
                    )
                    if response.status_code == 200:
                        job_data = response.json()
                        cleaned_job = JobService._clean_job_data(job_data)
                        jobs.append(Job(**cleaned_job))
                    else:
                        logger.error(
                            f"(sync) HTTP error fetching job {job_id}: Status {response.status_code}"
                        )
                except httpx.TimeoutException:
                    logger.error(f"(sync) Timeout fetching job {job_id}")
                except Exception as e:
                    logger.error(f"(sync) Exception fetching job {job_id}: {str(e)}")

        return jobs

    @staticmethod
    def _clean_job_data(job_data: Dict) -> Dict:
        """
        Clean and validate job data for processing.

        Args:
            job_data: Raw job data dictionary

        Returns:
            Dict: Cleaned job data dictionary

        Raises:
            ValueError: If required fields are missing
        """
        if not job_data:
            raise ValueError("Empty job data provided")

        cleaned_data = job_data.copy()

        # Map field names from job service to resume matcher schema
        field_mapping = {
            "title": "job_title",  # Map title -> job_title
            "job_title": "job_title",  # Keep job_title as is
            "company": "company_name",  # Map company -> company_name
            "company_name": "company_name",  # Keep company_name as is
        }

        # Apply field mapping
        for old_field, new_field in field_mapping.items():
            if old_field in cleaned_data and old_field != new_field:
                cleaned_data[new_field] = cleaned_data[old_field]
                if old_field != new_field:  # Only remove if it's actually different
                    del cleaned_data[old_field]

        # Validate required fields (using the new field names)
        required_fields = ["id", "job_title"]
        for field in required_fields:
            if field not in cleaned_data or not cleaned_data[field]:
                # Try to extract from alternative field names
                if field == "job_title" and ("title" in job_data):
                    cleaned_data["job_title"] = job_data["title"]
                elif field == "company_name" and ("company" in job_data):
                    cleaned_data["company_name"] = job_data["company"]
                else:
                    raise ValueError(f"Missing required field: {field}")

        # Clean HTML content from text fields
        text_fields = ["description", "requirements", "responsibilities", "summary"]
        for field in text_fields:
            if field in cleaned_data and cleaned_data[field]:
                cleaned_data[field] = JobService._clean_html_content(
                    cleaned_data[field]
                )

        # Handle salary with better error tolerance
        if "salary" in cleaned_data and cleaned_data["salary"]:
            try:
                salary_str = str(cleaned_data["salary"]).strip()

                # Clean common salary format issues
                salary_str = re.sub(
                    r"[^\d\-\.,\s]", "", salary_str
                )  # Remove non-numeric chars except dash, comma, dot, space
                salary_str = re.sub(
                    r"\s+", " ", salary_str
                ).strip()  # Normalize whitespace

                # Handle different salary formats
                if not salary_str or salary_str in ["", "0", "null", "None"]:
                    cleaned_data["salary"] = None
                elif "-" in salary_str or "to" in salary_str.lower():
                    # Keep as string for range format
                    cleaned_data["salary"] = salary_str
                else:
                    # Keep all salary values as strings for schema compatibility
                    cleaned_data["salary"] = salary_str

            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid salary format: {cleaned_data.get('salary')} - keeping as string"
                )
                cleaned_data["salary"] = (
                    str(cleaned_data["salary"]) if cleaned_data.get("salary") else None
                )

        # Normalize string fields (using new field names)
        string_fields = [
            "job_title",
            "company_name",
            "location",
            "job_type",
            "experience",
        ]
        for field in string_fields:
            if field in cleaned_data and cleaned_data[field]:
                cleaned_data[field] = str(cleaned_data[field]).strip()

        # Normalize numeric fields to prevent comparison errors
        numeric_fields = ["id", "job_id"]
        for field in numeric_fields:
            if field in cleaned_data and cleaned_data[field] is not None:
                try:
                    # Convert to int if it's a numeric string or number
                    if (
                        isinstance(cleaned_data[field], str)
                        and cleaned_data[field].isdigit()
                    ):
                        cleaned_data[field] = int(cleaned_data[field])
                    elif isinstance(cleaned_data[field], (int, float)):
                        cleaned_data[field] = int(cleaned_data[field])
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not convert {field} to integer: {cleaned_data[field]}"
                    )
                    # Keep as string if conversion fails
                    cleaned_data[field] = str(cleaned_data[field])

        # Ensure list fields are actually lists
        list_fields = [
            "skills",
            "requirements",
            "responsibilities",
            "benefits",
            "qualifications",
        ]
        for field in list_fields:
            if field in cleaned_data:
                if isinstance(cleaned_data[field], str):
                    # Try to parse comma-separated string
                    try:
                        cleaned_data[field] = [
                            item.strip()
                            for item in cleaned_data[field].split(",")
                            if item.strip()
                        ]
                    except Exception:
                        cleaned_data[field] = (
                            [cleaned_data[field]] if cleaned_data[field] else []
                        )
                elif not isinstance(cleaned_data[field], list):
                    cleaned_data[field] = []

        # Ensure we have apply_link field
        if "apply_link" not in cleaned_data and "url" in cleaned_data:
            cleaned_data["apply_link"] = cleaned_data["url"]

        # Add default values for missing optional fields
        default_fields = {
            "location": "Not specified",
            "job_type": "Not specified",
            "experience": "Not specified",
            "description": "",
            "salary": None,
            "apply_link": None,
            "skills": [],
            "requirements": [],
            "responsibilities": [],
            "benefits": [],
            "qualifications": [],
        }

        for field, default_value in default_fields.items():
            if field not in cleaned_data:
                cleaned_data[field] = default_value

        return cleaned_data

    @staticmethod
    def _clean_html_content(html_content: str) -> str:
        """
        Clean HTML content from job descriptions with robust error handling.

        Args:
            html_content: Raw HTML content string

        Returns:
            str: Cleaned plain text content
        """
        if not html_content or not isinstance(html_content, str):
            return ""

        # Early return for non-HTML content
        if "<" not in html_content or ">" not in html_content:
            return html_content.strip()

        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Replace <br> and <hr> with newlines
            for element in soup.find_all(["br", "hr"]):
                element.replace_with("\n")

            # Convert lists to formatted text
            for ul in soup.find_all("ul"):
                for li in ul.find_all("li"):
                    li_text = li.get_text().strip()
                    if li_text:
                        li.replace_with(f"• {li_text}\n")

            for ol in soup.find_all("ol"):
                for i, li in enumerate(ol.find_all("li"), 1):
                    li_text = li.get_text().strip()
                    if li_text:
                        li.replace_with(f"{i}. {li_text}\n")

            # Handle paragraphs and divs
            for element in soup.find_all(["p", "div"]):
                element_text = element.get_text().strip()
                if element_text:
                    element.replace_with(f"{element_text}\n\n")

            # Handle headings
            for level in range(1, 7):
                for heading in soup.find_all(f"h{level}"):
                    heading_text = heading.get_text().strip()
                    if heading_text:
                        heading.replace_with(f"\n{heading_text}\n")

            # Get the cleaned text
            text = soup.get_text()

            # Post-processing cleanup
            text = re.sub(r"\n{3,}", "\n\n", text)  # Normalize multiple newlines
            text = re.sub(r"[ \t]{2,}", " ", text)  # Normalize multiple spaces/tabs
            text = re.sub(
                r"\n[ \t]+", "\n", text
            )  # Remove leading whitespace on new lines

            return text.strip()

        except Exception as e:
            logger.error(f"Error cleaning HTML content: {str(e)}")
            # Robust fallback: strip all HTML tags and normalize whitespace
            try:
                clean_text = re.sub(r"<[^>]+>", " ", html_content)
                clean_text = re.sub(r"\s+", " ", clean_text)
                return clean_text.strip()
            except Exception:
                # Ultimate fallback
                return str(html_content).strip()

    @staticmethod
    async def get_job_count() -> int:
        """
        Get the total number of available jobs (if supported by the service).

        Returns:
            int: Total job count, or 0 if unable to retrieve
        """
        try:
            async with rabbitmq_client_context() as client:
                # This would require additional event types in the original service
                # For now, return 0 as it's not implemented
                logger.debug("Job count retrieval not implemented via RabbitMQ")
                return 0
        except Exception as e:
            logger.error(f"Error getting job count: {str(e)}")
            return 0

    @staticmethod
    def validate_job_ids(job_ids: List[int]) -> List[int]:
        """
        Validate and filter job IDs.

        Args:
            job_ids: List of job IDs to validate

        Returns:
            List[int]: List of valid job IDs
        """
        if not job_ids:
            return []

        valid_ids = []
        for job_id in job_ids:
            try:
                job_id_int = int(job_id)
                if job_id_int > 0:
                    valid_ids.append(job_id_int)
                else:
                    logger.warning(f"Invalid job ID (non-positive): {job_id}")
            except (ValueError, TypeError):
                logger.warning(f"Invalid job ID (non-numeric): {job_id}")

        return valid_ids
