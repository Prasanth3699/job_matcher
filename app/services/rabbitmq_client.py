"""
Production-ready RabbitMQ client for job data requests.
Implements event-driven communication with proper error handling and connection management.
"""

import asyncio
import json
from typing import Dict, List, Optional
import aio_pika
from aio_pika import connect, Message, DeliveryMode, ExchangeType
from contextlib import asynccontextmanager

from app.utils.logger import logger
from app.core.config import get_settings
from app.core.circuit_breaker import rabbitmq_breaker
from app.core.events import (
    EventMessage, 
    EventType, 
    JobDataRequestBuilder, 
    RoutingKeyGenerator
)

settings = get_settings()


class RabbitMQError(Exception):
    """Base exception for RabbitMQ-related errors."""
    pass


class MessageTimeoutError(RabbitMQError):
    """Raised when a message request times out."""
    pass


class ConnectionLostError(RabbitMQError):
    """Raised when RabbitMQ connection is lost."""
    pass


class JobDataClient:
    """
    Production-ready RabbitMQ client for job data requests.
    
    Handles connection management, message routing, and response handling
    with proper error recovery and circuit breaker protection.
    """
    
    EXCHANGE_NAME = "job_scraper_events"
    RESPONSE_TIMEOUT = 30.0
    CONNECTION_TIMEOUT = 10.0
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    
    def __init__(self, rabbitmq_url: str = None):
        self.rabbitmq_url = rabbitmq_url or settings.RABBITMQ_URL
        self.connection = None
        self.channel = None
        self.service_id = "ml-service"
        self.response_handlers = {}
        self.exchange = None
        self.response_queue = None
        self.is_connected = False
        self.is_consuming = False
        self._connection_lock = asyncio.Lock()

    async def connect(self) -> None:
        """
        Initialize RabbitMQ connection with retry logic and proper error handling.
        
        Raises:
            ConnectionError: If unable to establish connection after max retries
        """
        async with self._connection_lock:
            if self.is_connected:
                return
            
            retry_delay = self.RETRY_DELAY
            
            for attempt in range(self.MAX_RETRIES):
                try:
                    logger.info(f"Establishing RabbitMQ connection (attempt {attempt + 1}/{self.MAX_RETRIES})")
                    
                    # Create connection with timeout
                    self.connection = await asyncio.wait_for(
                        connect(self.rabbitmq_url),
                        timeout=self.CONNECTION_TIMEOUT
                    )
                    
                    self.channel = await self.connection.channel()
                    await self.channel.set_qos(prefetch_count=10)
                    
                    # Declare exchange matching original service expectations
                    self.exchange = await self.channel.declare_exchange(
                        self.EXCHANGE_NAME,
                        ExchangeType.TOPIC,
                        durable=True
                    )
                    
                    # Setup response queue with proper naming convention
                    # Use a new queue name to avoid conflicts with existing queues
                    queue_name = f"ml.{self.service_id}.responses.v2"
                    self.response_queue = await self.channel.declare_queue(
                        queue_name,
                        durable=True,
                        auto_delete=False,
                        arguments={
                            "x-message-ttl": 300000,  # 5 minutes TTL
                            "x-max-length": 1000,     # Max queue length
                        }
                    )
                    
                    # Bind response queue to appropriate routing keys
                    await self.response_queue.bind(
                        self.exchange,
                        RoutingKeyGenerator.for_job_data_response(self.service_id)
                    )
                    await self.response_queue.bind(
                        self.exchange,
                        RoutingKeyGenerator.for_bulk_job_data_response(self.service_id)
                    )
                    
                    # Start consuming responses
                    await self._start_consuming()
                    
                    self.is_connected = True
                    logger.info("RabbitMQ connection established successfully")
                    return
                    
                except asyncio.TimeoutError:
                    logger.error(f"Connection timeout on attempt {attempt + 1}")
                except Exception as e:
                    logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
                    
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5
            
            raise ConnectionError(f"Failed to connect to RabbitMQ after {self.MAX_RETRIES} attempts")
    
    async def _start_consuming(self) -> None:
        """Start consuming messages from response queue."""
        if not self.is_consuming and self.response_queue:
            await self.response_queue.consume(
                self._handle_response,
                no_ack=False
            )
            self.is_consuming = True
            logger.debug("Started consuming RabbitMQ responses")

    async def request_job_data(
        self, 
        job_id: int, 
        additional_fields: Optional[List[str]] = None,
        timeout: float = None
    ) -> Dict:
        """
        Request single job data with circuit breaker protection and response handling.
        
        Args:
            job_id: ID of the job to request
            additional_fields: Optional list of additional fields to include
            timeout: Response timeout in seconds
            
        Returns:
            Dict: Job data response
            
        Raises:
            TimeoutError: If request times out
            ValueError: If job_id is invalid
            ConnectionError: If RabbitMQ connection fails
        """
        return await rabbitmq_breaker.call(
            self._send_job_data_request, job_id, additional_fields, timeout
        )

    async def _send_job_data_request(
        self, 
        job_id: int, 
        additional_fields: Optional[List[str]] = None,
        timeout: float = None
    ) -> Dict:
        """
        Internal method to send job data request and wait for response.
        
        Args:
            job_id: ID of the job to request
            additional_fields: Optional list of additional fields to include
            timeout: Response timeout in seconds
            
        Returns:
            Dict: Job data response
        """
        if not self.is_connected:
            await self.connect()
        
        # Create standardized event message
        event_message = JobDataRequestBuilder.create_single_job_request(
            job_id=job_id,
            ml_service_id=self.service_id,
            additional_fields=additional_fields
        )
        
        # Setup response future
        response_future = asyncio.Future()
        request_id = event_message.data["request_id"]
        self.response_handlers[request_id] = response_future
        
        try:
            # Send message
            message = Message(
                json.dumps(event_message.to_dict()).encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
                message_id=event_message.event_id,
                correlation_id=event_message.correlation_id,
                headers={
                    "source_service": self.service_id,
                    "event_type": event_message.event_type.value
                }
            )
            
            await self.exchange.publish(
                message,
                routing_key=RoutingKeyGenerator.for_job_data_request()
            )
            
            logger.debug(f"Job data request sent: job_id={job_id}, request_id={request_id}")
            
            # Wait for response
            response_timeout = timeout or self.RESPONSE_TIMEOUT
            response = await asyncio.wait_for(response_future, timeout=response_timeout)
            
            return response
            
        except asyncio.TimeoutError:
            self.response_handlers.pop(request_id, None)
            logger.error(f"Request timeout for job_id: {job_id}")
            raise TimeoutError(f"Request timeout for job_id: {job_id}")
        except Exception as e:
            self.response_handlers.pop(request_id, None)
            logger.error(f"Failed to send job data request: {str(e)}")
            
            if "closed" in str(e).lower():
                await self._handle_connection_loss()
            raise ConnectionError(f"Failed to send job data request: {str(e)}") from e

    async def request_bulk_job_data(
        self,
        job_ids: List[int],
        additional_fields: Optional[List[str]] = None,
        timeout: float = None
    ) -> List[Dict]:
        """
        Request bulk job data and wait for response.
        
        Args:
            job_ids: List of job IDs to request
            additional_fields: Optional list of additional fields to include
            timeout: Response timeout in seconds
            
        Returns:
            List[Dict]: List of job data responses
            
        Raises:
            TimeoutError: If request times out
            ValueError: If job_ids is invalid
            ConnectionError: If RabbitMQ connection fails
        """
        if not self.is_connected:
            await self.connect()
        
        # Create standardized event message
        event_message = JobDataRequestBuilder.create_bulk_job_request(
            job_ids=job_ids,
            ml_service_id=self.service_id,
            additional_fields=additional_fields
        )
        
        # Setup response future
        response_future = asyncio.Future()
        request_id = event_message.data["request_id"]
        self.response_handlers[request_id] = response_future
        
        try:
            # Send message
            message = Message(
                json.dumps(event_message.to_dict()).encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
                message_id=event_message.event_id,
                correlation_id=event_message.correlation_id,
                headers={
                    "source_service": self.service_id,
                    "event_type": event_message.event_type.value
                }
            )
            
            await self.exchange.publish(
                message,
                routing_key=RoutingKeyGenerator.for_bulk_job_data_request()
            )
            
            logger.debug(f"Bulk job data request sent: {len(job_ids)} jobs, request_id={request_id}")
            
            # Wait for response
            response_timeout = timeout or self.RESPONSE_TIMEOUT * 2  # Longer timeout for bulk
            response = await asyncio.wait_for(response_future, timeout=response_timeout)
            
            return response
            
        except asyncio.TimeoutError:
            self.response_handlers.pop(request_id, None)
            logger.error(f"Bulk request timeout for {len(job_ids)} jobs")
            raise TimeoutError(f"Bulk request timeout for {len(job_ids)} jobs")
        except Exception as e:
            self.response_handlers.pop(request_id, None)
            logger.error(f"Failed to send bulk job data request: {str(e)}")
            
            if "closed" in str(e).lower():
                await self._handle_connection_loss()
            raise ConnectionError(f"Failed to send bulk job data request: {str(e)}") from e

    async def _handle_response(self, message: aio_pika.Message) -> None:
        """
        Handle incoming response messages with proper error handling and validation.
        
        Args:
            message: Incoming RabbitMQ message
        """
        try:
            # Parse message
            message_data = json.loads(message.body.decode())
            event_message = EventMessage.from_dict(message_data)
            
            request_id = event_message.data.get("request_id")
            
            if not request_id:
                logger.warning("Received response without request_id")
                await message.ack()
                return
            
            # Find and execute handler
            if request_id in self.response_handlers:
                response_future = self.response_handlers.pop(request_id)
                
                # Extract response data based on event type
                if event_message.event_type == EventType.JOB_DATA_RESPONSE:
                    response_data = self._extract_job_data_response(event_message)
                elif event_message.event_type == EventType.BULK_JOB_DATA_RESPONSE:
                    response_data = self._extract_bulk_job_data_response(event_message)
                else:
                    logger.warning(f"Unexpected response event type: {event_message.event_type}")
                    await message.ack()
                    return
                
                # Set result or exception
                if event_message.data.get("status") == "error":
                    error_msg = event_message.data.get("error", "Unknown error")
                    response_future.set_exception(Exception(error_msg))
                else:
                    response_future.set_result(response_data)
                
                logger.debug(f"Response processed for request_id: {request_id}")
            else:
                logger.warning(f"No handler found for request_id: {request_id}")
            
            await message.ack()
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in response message: {str(e)}")
            await message.nack(requeue=False)
        except Exception as e:
            logger.error(f"Error handling response: {str(e)}")
            await message.nack(requeue=False)
    
    def _extract_job_data_response(self, event_message: EventMessage) -> Dict:
        """Extract job data from single job response."""
        job_data = event_message.data.get("job_data")
        if not job_data:
            raise ValueError("No job data in response")
        return job_data
    
    def _extract_bulk_job_data_response(self, event_message: EventMessage) -> List[Dict]:
        """Extract job data from bulk job response."""
        bulk_data = event_message.data.get("bulk_data", {})
        jobs = bulk_data.get("jobs", [])
        return jobs
    
    async def _handle_connection_loss(self) -> None:
        """Handle connection loss and reset state."""
        logger.warning("RabbitMQ connection lost, resetting state")
        self.is_connected = False
        self.is_consuming = False
        
        # Fail all pending requests
        for request_id, future in self.response_handlers.items():
            if not future.done():
                future.set_exception(ConnectionError("RabbitMQ connection lost"))
        
        self.response_handlers.clear()

    async def close(self) -> None:
        """
        Gracefully close RabbitMQ connection and clean up resources.
        """
        async with self._connection_lock:
            try:
                # Cancel all pending requests
                for request_id, future in self.response_handlers.items():
                    if not future.done():
                        future.set_exception(ConnectionError("Client shutting down"))
                self.response_handlers.clear()
                
                # Close channel and connection
                if self.channel and not self.channel.is_closed:
                    await self.channel.close()
                
                if self.connection and not self.connection.is_closed:
                    await self.connection.close()
                
                self.is_connected = False
                self.is_consuming = False
                logger.info("RabbitMQ connection closed gracefully")
                
            except Exception as e:
                logger.error(f"Error during RabbitMQ cleanup: {str(e)}")
    
    async def health_check(self) -> bool:
        """
        Check if RabbitMQ connection is healthy.
        
        Returns:
            bool: True if connection is healthy
        """
        try:
            if not self.is_connected or not self.connection or self.connection.is_closed:
                return False
            
            # Try to declare a temporary queue to test connection
            temp_queue = await self.channel.declare_queue(
                exclusive=True,
                auto_delete=True
            )
            await temp_queue.delete()
            return True
            
        except Exception as e:
            logger.warning(f"RabbitMQ health check failed: {str(e)}")
            return False


# Global client instance
_rabbitmq_client = None


async def get_rabbitmq_client() -> JobDataClient:
    """
    Get or create the global RabbitMQ client instance.
    
    Returns:
        JobDataClient: Connected RabbitMQ client
        
    Raises:
        ConnectionError: If unable to establish connection
    """
    global _rabbitmq_client
    
    if _rabbitmq_client is None:
        _rabbitmq_client = JobDataClient()
    
    if not _rabbitmq_client.is_connected:
        await _rabbitmq_client.connect()
    
    return _rabbitmq_client


@asynccontextmanager
async def rabbitmq_client_context():
    """
    Context manager for RabbitMQ client with automatic cleanup.
    
    Usage:
        async with rabbitmq_client_context() as client:
            result = await client.request_job_data(123)
    """
    client = await get_rabbitmq_client()
    try:
        yield client
    except Exception:
        # On error, check if we need to reconnect
        if not await client.health_check():
            await client.close()
        raise


async def cleanup_rabbitmq_client() -> None:
    """
    Clean up the global RabbitMQ client instance.
    Should be called during application shutdown.
    """
    global _rabbitmq_client
    
    if _rabbitmq_client:
        await _rabbitmq_client.close()
        _rabbitmq_client = None
        logger.info("RabbitMQ client cleanup completed")