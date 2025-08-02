"""
Event system for RabbitMQ-based inter-service communication.
Provides standardized event types and message structures for microservice integration.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional
import uuid


class EventType(Enum):
    """Standard event types for inter-service communication."""
    
    # Job events
    JOB_SCRAPED = "job.scraped"
    JOB_UPDATED = "job.updated"
    JOB_DELETED = "job.deleted"
    JOBS_BULK_IMPORTED = "jobs.bulk_imported"
    
    # User events
    USER_REGISTERED = "user.registered"
    USER_LOGIN = "user.login"
    USER_PROFILE_UPDATED = "user.profile_updated"
    
    # ML service events - Data requests
    JOB_DATA_REQUESTED = "ml.job_data_requested"
    JOB_DATA_RESPONSE = "ml.job_data_response"
    BULK_JOB_DATA_REQUESTED = "ml.bulk_job_data_requested"
    BULK_JOB_DATA_RESPONSE = "ml.bulk_job_data_response"
    
    # ML service events - Analysis
    JOB_ANALYSIS_REQUESTED = "ml.job_analysis_requested"
    JOB_ANALYSIS_COMPLETED = "ml.job_analysis_completed"
    RECOMMENDATION_GENERATED = "ml.recommendation_generated"
    
    # LLM events
    LLM_PROCESSING_REQUESTED = "llm.processing_requested"
    LLM_PROCESSING_COMPLETED = "llm.processing_completed"
    
    # System events
    CLEANUP_REQUESTED = "system.cleanup_requested"
    HEALTH_CHECK_FAILED = "system.health_check_failed"
    
    # Notification events
    EMAIL_REQUEST = "notification.email_request"
    ALERT_TRIGGERED = "notification.alert_triggered"


@dataclass
class EventMessage:
    """
    Standard event message structure for RabbitMQ communication.
    
    Ensures consistent message format across all microservices and provides
    serialization/deserialization capabilities for reliable message handling.
    """
    
    event_id: str
    event_type: EventType
    source_service: str
    timestamp: str
    data: Dict[str, Any]
    correlation_id: Optional[str] = None
    retry_count: int = 0
    
    def __post_init__(self):
        """Validate and normalize message fields after initialization."""
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        
        if not self.correlation_id:
            self.correlation_id = self.event_id
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert EventMessage to dictionary for JSON serialization.
        
        Returns:
            Dict[str, Any]: Serializable dictionary representation
        """
        message_dict = asdict(self)
        message_dict["event_type"] = self.event_type.value
        return message_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventMessage":
        """
        Create EventMessage from dictionary (deserialization).
        
        Args:
            data: Dictionary containing message data
            
        Returns:
            EventMessage: Reconstructed message object
            
        Raises:
            ValueError: If event_type is not valid
        """
        try:
            data["event_type"] = EventType(data["event_type"])
            return cls(**data)
        except ValueError as e:
            raise ValueError(f"Invalid event_type: {data.get('event_type')}") from e
    
    @classmethod
    def create_request(
        cls,
        event_type: EventType,
        source_service: str,
        data: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> "EventMessage":
        """
        Create a new request event message.
        
        Args:
            event_type: Type of event
            source_service: Service originating the message
            data: Event-specific data payload
            correlation_id: Optional correlation ID for request tracking
            
        Returns:
            EventMessage: New event message instance
        """
        return cls(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source_service=source_service,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data,
            correlation_id=correlation_id
        )
    
    @classmethod
    def create_response(
        cls,
        event_type: EventType,
        source_service: str,
        data: Dict[str, Any],
        correlation_id: str
    ) -> "EventMessage":
        """
        Create a response event message with matching correlation ID.
        
        Args:
            event_type: Response event type
            source_service: Service sending the response
            data: Response data payload
            correlation_id: Correlation ID from original request
            
        Returns:
            EventMessage: New response event message
        """
        return cls(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source_service=source_service,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data,
            correlation_id=correlation_id
        )
    
    def is_response_to(self, request_message: "EventMessage") -> bool:
        """
        Check if this message is a response to the given request.
        
        Args:
            request_message: Original request message
            
        Returns:
            bool: True if this is a response to the request
        """
        return (
            self.correlation_id == request_message.correlation_id and
            self.correlation_id is not None
        )
    
    def increment_retry(self) -> "EventMessage":
        """
        Create a copy of this message with incremented retry count.
        
        Returns:
            EventMessage: New message with incremented retry count
        """
        return EventMessage(
            event_id=str(uuid.uuid4()),
            event_type=self.event_type,
            source_service=self.source_service,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=self.data.copy(),
            correlation_id=self.correlation_id,
            retry_count=self.retry_count + 1
        )


class JobDataRequestBuilder:
    """Builder for job data request messages with validation."""
    
    @staticmethod
    def create_single_job_request(
        job_id: int,
        ml_service_id: str,
        additional_fields: Optional[list] = None,
        correlation_id: Optional[str] = None
    ) -> EventMessage:
        """
        Create a single job data request message.
        
        Args:
            job_id: ID of the job to request
            ml_service_id: Identifier of the requesting ML service
            additional_fields: Optional list of additional fields to include
            correlation_id: Optional correlation ID for tracking
            
        Returns:
            EventMessage: Formatted job data request
            
        Raises:
            ValueError: If job_id is invalid
        """
        if not isinstance(job_id, int) or job_id <= 0:
            raise ValueError(f"Invalid job_id: {job_id}")
        
        request_id = str(uuid.uuid4())
        
        data = {
            "job_id": job_id,
            "request_id": request_id,
            "ml_service_id": ml_service_id,
            "additional_fields": additional_fields or []
        }
        
        return EventMessage.create_request(
            event_type=EventType.JOB_DATA_REQUESTED,
            source_service=ml_service_id,
            data=data,
            correlation_id=correlation_id
        )
    
    @staticmethod
    def create_bulk_job_request(
        job_ids: list[int],
        ml_service_id: str,
        filters: Optional[Dict[str, Any]] = None,
        additional_fields: Optional[list] = None,
        correlation_id: Optional[str] = None
    ) -> EventMessage:
        """
        Create a bulk job data request message.
        
        Args:
            job_ids: List of job IDs to request
            ml_service_id: Identifier of the requesting ML service
            filters: Optional filtering criteria
            additional_fields: Optional list of additional fields to include
            correlation_id: Optional correlation ID for tracking
            
        Returns:
            EventMessage: Formatted bulk job data request
            
        Raises:
            ValueError: If job_ids is invalid
        """
        if not job_ids or not all(isinstance(jid, int) and jid > 0 for jid in job_ids):
            raise ValueError(f"Invalid job_ids: {job_ids}")
        
        request_id = str(uuid.uuid4())
        
        request_filters = filters or {}
        request_filters["job_ids"] = job_ids
        
        data = {
            "request_id": request_id,
            "ml_service_id": ml_service_id,
            "filters": request_filters,
            "additional_fields": additional_fields or []
        }
        
        return EventMessage.create_request(
            event_type=EventType.BULK_JOB_DATA_REQUESTED,
            source_service=ml_service_id,
            data=data,
            correlation_id=correlation_id
        )


class RoutingKeyGenerator:
    """Generates consistent routing keys for different event types."""
    
    @staticmethod
    def for_job_data_request() -> str:
        """Routing key for job data requests."""
        return "ml.job_data_requested"
    
    @staticmethod
    def for_bulk_job_data_request() -> str:
        """Routing key for bulk job data requests."""
        return "ml.bulk_job_data_requested"
    
    @staticmethod
    def for_job_data_response(ml_service_id: str) -> str:
        """Routing key for job data responses."""
        return f"ml.{ml_service_id}.job_data_response"
    
    @staticmethod
    def for_bulk_job_data_response(ml_service_id: str) -> str:
        """Routing key for bulk job data responses."""
        return f"ml.{ml_service_id}.bulk_job_data_response"