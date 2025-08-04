"""
Comprehensive audit logging system for security events.
Provides detailed logging, monitoring, and analysis capabilities.
"""

import json
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

import aiofiles
from sqlalchemy import Column, String, DateTime, Text, Integer, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.utils.logger import logger


class SecurityEventType(Enum):
    """Security event types for categorization."""
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_FAILURE = "authorization_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    THREAT_DETECTED = "threat_detected"
    INPUT_VALIDATION_FAILURE = "input_validation_failure"
    ACCESS_DENIED = "access_denied"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_ERROR = "system_error"
    CONFIGURATION_CHANGE = "config_change"
    USER_CREATION = "user_creation"
    USER_DELETION = "user_deletion"
    PASSWORD_CHANGE = "password_change"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    FILE_ACCESS = "file_access"
    API_ABUSE = "api_abuse"
    MALICIOUS_REQUEST = "malicious_request"
    VULNERABILITY_EXPLOIT = "vulnerability_exploit"
    REQUEST_PROCESSED = "request_processed"
    IP_NOT_WHITELISTED = "ip_not_whitelisted"
    REQUEST_VALIDATION_FAILED = "request_validation_failed"
    THREAT_THRESHOLD_EXCEEDED = "threat_threshold_exceeded"
    MIDDLEWARE_ERROR = "middleware_error"


class SeverityLevel(Enum):
    """Security event severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_type: SecurityEventType
    severity: SeverityLevel
    timestamp: datetime
    client_ip: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    user_agent: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    threat_score: float = 0.0
    geolocation: Optional[Dict[str, str]] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    def get_event_hash(self) -> str:
        """Generate unique hash for the event."""
        hash_data = f"{self.event_type.value}:{self.client_ip}:{self.timestamp.isoformat()}"
        return hashlib.sha256(hash_data.encode()).hexdigest()[:16]


class AuditStorage:
    """Abstract base class for audit storage backends."""
    
    async def store_event(self, event: SecurityEvent) -> bool:
        """Store security event."""
        raise NotImplementedError
    
    async def get_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[SecurityEventType]] = None,
        client_ip: Optional[str] = None,
        limit: int = 100
    ) -> List[SecurityEvent]:
        """Retrieve security events."""
        raise NotImplementedError
    
    async def get_threat_summary(
        self,
        time_window: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """Get threat summary for time window."""
        raise NotImplementedError


class FileAuditStorage(AuditStorage):
    """File-based audit storage."""
    
    def __init__(self, log_directory: str = "logs/security"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.current_date = None
        self.current_file = None
    
    async def store_event(self, event: SecurityEvent) -> bool:
        """Store event to daily log file."""
        try:
            date_str = event.timestamp.strftime("%Y-%m-%d")
            log_file = self.log_directory / f"security_{date_str}.jsonl"
            
            event_data = event.to_dict()
            event_line = json.dumps(event_data) + "\n"
            
            async with aiofiles.open(log_file, "a", encoding="utf-8") as f:
                await f.write(event_line)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store security event to file: {e}")
            return False
    
    async def get_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[SecurityEventType]] = None,
        client_ip: Optional[str] = None,
        limit: int = 100
    ) -> List[SecurityEvent]:
        """Retrieve events from log files."""
        events = []
        
        try:
            # Determine date range
            if not start_time:
                start_time = datetime.utcnow() - timedelta(days=7)
            if not end_time:
                end_time = datetime.utcnow()
            
            # Read files for date range
            current_date = start_time.date()
            while current_date <= end_time.date() and len(events) < limit:
                log_file = self.log_directory / f"security_{current_date.isoformat()}.jsonl"
                
                if log_file.exists():
                    async with aiofiles.open(log_file, "r", encoding="utf-8") as f:
                        async for line in f:
                            if len(events) >= limit:
                                break
                            
                            try:
                                event_data = json.loads(line.strip())
                                event = self._dict_to_event(event_data)
                                
                                # Apply filters
                                if self._matches_filters(
                                    event, start_time, end_time,
                                    event_types, client_ip
                                ):
                                    events.append(event)
                                    
                            except (json.JSONDecodeError, KeyError) as e:
                                logger.warning(f"Invalid event data in log: {e}")
                
                current_date += timedelta(days=1)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to retrieve events from files: {e}")
            return []
    
    def _dict_to_event(self, data: Dict[str, Any]) -> SecurityEvent:
        """Convert dictionary to SecurityEvent."""
        return SecurityEvent(
            event_type=SecurityEventType(data['event_type']),
            severity=SeverityLevel(data['severity']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            client_ip=data['client_ip'],
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            endpoint=data.get('endpoint'),
            method=data.get('method'),
            user_agent=data.get('user_agent'),
            details=data.get('details'),
            threat_score=data.get('threat_score', 0.0),
            geolocation=data.get('geolocation'),
            success=data.get('success', True),
            error_message=data.get('error_message')
        )
    
    def _matches_filters(
        self,
        event: SecurityEvent,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        event_types: Optional[List[SecurityEventType]],
        client_ip: Optional[str]
    ) -> bool:
        """Check if event matches filters."""
        if start_time and event.timestamp < start_time:
            return False
        
        if end_time and event.timestamp > end_time:
            return False
        
        if event_types and event.event_type not in event_types:
            return False
        
        if client_ip and event.client_ip != client_ip:
            return False
        
        return True
    
    async def get_threat_summary(
        self,
        time_window: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """Get threat summary from log files."""
        end_time = datetime.utcnow()
        start_time = end_time - time_window
        
        events = await self.get_events(
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        return self._calculate_threat_summary(events)
    
    def _calculate_threat_summary(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Calculate threat summary from events."""
        summary = {
            'total_events': len(events),
            'by_severity': {level.name: 0 for level in SeverityLevel},
            'by_type': {event_type.value: 0 for event_type in SecurityEventType},
            'top_clients': {},
            'threat_score_stats': {
                'min': 0.0,
                'max': 0.0,
                'avg': 0.0,
                'total': 0.0
            },
            'failed_attempts': 0,
            'unique_clients': set()
        }
        
        threat_scores = []
        
        for event in events:
            # Count by severity
            summary['by_severity'][event.severity.name] += 1
            
            # Count by type
            summary['by_type'][event.event_type.value] += 1
            
            # Track clients
            summary['unique_clients'].add(event.client_ip)
            summary['top_clients'][event.client_ip] = (
                summary['top_clients'].get(event.client_ip, 0) + 1
            )
            
            # Track threat scores
            if event.threat_score > 0:
                threat_scores.append(event.threat_score)
            
            # Count failures
            if not event.success:
                summary['failed_attempts'] += 1
        
        # Calculate threat score statistics
        if threat_scores:
            summary['threat_score_stats'] = {
                'min': min(threat_scores),
                'max': max(threat_scores),
                'avg': sum(threat_scores) / len(threat_scores),
                'total': sum(threat_scores)
            }
        
        # Convert unique clients set to count
        summary['unique_clients'] = len(summary['unique_clients'])
        
        # Sort top clients
        summary['top_clients'] = dict(
            sorted(summary['top_clients'].items(), 
                   key=lambda x: x[1], reverse=True)[:10]
        )
        
        return summary


class DatabaseAuditStorage(AuditStorage):
    """Database-based audit storage using SQLAlchemy."""
    
    def __init__(self, database_url: str, table_name: str = "security_events"):
        self.database_url = database_url
        self.table_name = table_name
        self.Base = declarative_base()
        self._create_table_model()
    
    def _create_table_model(self):
        """Create SQLAlchemy table model."""
        class SecurityEventModel(self.Base):
            __tablename__ = self.table_name
            
            id = Column(String(16), primary_key=True)  # Event hash
            event_type = Column(String(50), nullable=False, index=True)
            severity = Column(Integer, nullable=False, index=True)
            timestamp = Column(DateTime, nullable=False, index=True)
            client_ip = Column(String(45), nullable=False, index=True)
            user_id = Column(String(100), nullable=True, index=True)
            session_id = Column(String(100), nullable=True)
            endpoint = Column(String(500), nullable=True)
            method = Column(String(10), nullable=True)
            user_agent = Column(Text, nullable=True)
            details = Column(JSON, nullable=True)
            threat_score = Column(Integer, default=0)  # Store as int (score * 100)
            geolocation = Column(JSON, nullable=True)
            success = Column(Boolean, default=True, index=True)
            error_message = Column(Text, nullable=True)
        
        self.SecurityEventModel = SecurityEventModel
    
    async def store_event(self, event: SecurityEvent) -> bool:
        """Store event to database."""
        try:
            # Convert to database model
            db_event = self.SecurityEventModel(
                id=event.get_event_hash(),
                event_type=event.event_type.value,
                severity=event.severity.value,
                timestamp=event.timestamp,
                client_ip=event.client_ip,
                user_id=event.user_id,
                session_id=event.session_id,
                endpoint=event.endpoint,
                method=event.method,
                user_agent=event.user_agent,
                details=event.details,
                threat_score=int(event.threat_score * 100),
                geolocation=event.geolocation,
                success=event.success,
                error_message=event.error_message
            )
            
            # Store to database (implement with your DB session)
            # This is a placeholder - implement with your database session
            logger.info(f"Would store event to database: {event.event_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store security event to database: {e}")
            return False


class ThreatAnalyzer:
    """Advanced threat analysis and pattern detection."""
    
    def __init__(self):
        self.suspicious_patterns = {
            'brute_force': {
                'threshold': 10,  # Failed attempts
                'time_window': 300,  # 5 minutes
                'severity': SeverityLevel.HIGH
            },
            'rate_limit_abuse': {
                'threshold': 5,  # Rate limit hits
                'time_window': 600,  # 10 minutes
                'severity': SeverityLevel.MEDIUM
            },
            'vulnerability_scan': {
                'threshold': 20,  # Different endpoints
                'time_window': 3600,  # 1 hour
                'severity': SeverityLevel.HIGH
            }
        }
    
    async def analyze_events(
        self,
        events: List[SecurityEvent],
        time_window: timedelta = timedelta(hours=1)
    ) -> Dict[str, Any]:
        """Analyze events for threat patterns."""
        analysis = {
            'threat_patterns': [],
            'high_risk_clients': [],
            'anomalies': [],
            'recommendations': []
        }
        
        # Group events by client IP
        client_events = {}
        for event in events:
            if event.client_ip not in client_events:
                client_events[event.client_ip] = []
            client_events[event.client_ip].append(event)
        
        # Analyze each client
        for client_ip, client_event_list in client_events.items():
            client_analysis = self._analyze_client_behavior(client_event_list)
            
            if client_analysis['risk_score'] > 5.0:
                analysis['high_risk_clients'].append({
                    'client_ip': client_ip,
                    'risk_score': client_analysis['risk_score'],
                    'patterns': client_analysis['patterns'],
                    'event_count': len(client_event_list)
                })
        
        # Detect system-wide patterns
        system_patterns = self._detect_system_patterns(events)
        analysis['threat_patterns'].extend(system_patterns)
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_client_behavior(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Analyze behavior patterns for a single client."""
        analysis = {
            'risk_score': 0.0,
            'patterns': [],
            'event_types': set(),
            'failed_attempts': 0,
            'unique_endpoints': set()
        }
        
        for event in events:
            analysis['event_types'].add(event.event_type)
            
            if not event.success:
                analysis['failed_attempts'] += 1
                analysis['risk_score'] += 1.0
            
            if event.endpoint:
                analysis['unique_endpoints'].add(event.endpoint)
            
            # Add threat score
            analysis['risk_score'] += event.threat_score
        
        # Detect patterns
        if analysis['failed_attempts'] > 5:
            analysis['patterns'].append('brute_force_attempt')
            analysis['risk_score'] += 3.0
        
        if len(analysis['unique_endpoints']) > 20:
            analysis['patterns'].append('endpoint_scanning')
            analysis['risk_score'] += 2.0
        
        if SecurityEventType.THREAT_DETECTED in analysis['event_types']:
            analysis['patterns'].append('threat_activity')
            analysis['risk_score'] += 5.0
        
        return analysis
    
    def _detect_system_patterns(self, events: List[SecurityEvent]) -> List[Dict[str, Any]]:
        """Detect system-wide threat patterns."""
        patterns = []
        
        # Count events by type
        event_type_counts = {}
        for event in events:
            event_type_counts[event.event_type] = (
                event_type_counts.get(event.event_type, 0) + 1
            )
        
        # Check for suspicious spikes
        for event_type, count in event_type_counts.items():
            if event_type in [
                SecurityEventType.AUTHENTICATION_FAILURE,
                SecurityEventType.RATE_LIMIT_EXCEEDED,
                SecurityEventType.THREAT_DETECTED
            ] and count > 50:
                patterns.append({
                    'type': 'suspicious_spike',
                    'event_type': event_type.value,
                    'count': count,
                    'severity': SeverityLevel.MEDIUM.value
                })
        
        return patterns
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on analysis."""
        recommendations = []
        
        if analysis['high_risk_clients']:
            recommendations.append(
                f"Consider blocking or rate-limiting {len(analysis['high_risk_clients'])} "
                "high-risk IP addresses"
            )
        
        if analysis['threat_patterns']:
            recommendations.append(
                "Review and strengthen security measures for detected threat patterns"
            )
        
        # Add more specific recommendations based on patterns
        for pattern in analysis['threat_patterns']:
            if pattern.get('type') == 'suspicious_spike':
                recommendations.append(
                    f"Investigate spike in {pattern['event_type']} events"
                )
        
        return recommendations


class AuditLogger:
    """
    Comprehensive audit logging system for security events.
    
    Features:
    - Multiple storage backends (file, database)
    - Event categorization and severity levels
    - Threat analysis and pattern detection
    - Real-time alerting capabilities
    - Compliance reporting
    """
    
    def __init__(
        self,
        storage_backends: Optional[List[AuditStorage]] = None,
        enable_analysis: bool = True,
        alert_threshold: SeverityLevel = SeverityLevel.HIGH
    ):
        self.storage_backends = storage_backends or [FileAuditStorage()]
        self.enable_analysis = enable_analysis
        self.alert_threshold = alert_threshold
        self.threat_analyzer = ThreatAnalyzer() if enable_analysis else None
        
        # Event queue for batch processing
        self.event_queue = asyncio.Queue()
        self.processing_task = None
        
        logger.info("Audit logger initialized")
    
    async def start(self):
        """Start the audit logger background processing."""
        if not self.processing_task:
            self.processing_task = asyncio.create_task(self._process_events())
    
    async def stop(self):
        """Stop the audit logger background processing."""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            self.processing_task = None
    
    async def log_security_event(
        self,
        event_data: Union[Dict[str, Any], SecurityEvent]
    ) -> bool:
        """Log a security event."""
        try:
            # Convert dict to SecurityEvent if needed
            if isinstance(event_data, dict):
                event = self._dict_to_security_event(event_data)
            else:
                event = event_data
            
            # Add to processing queue
            await self.event_queue.put(event)
            
            # Immediate alert for critical events
            if event.severity == SeverityLevel.CRITICAL:
                await self._send_immediate_alert(event)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
            return False
    
    async def _process_events(self):
        """Background task to process events from queue."""
        while True:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(
                    self.event_queue.get(), timeout=1.0
                )
                
                # Store event in all backends
                for backend in self.storage_backends:
                    try:
                        await backend.store_event(event)
                    except Exception as e:
                        logger.error(f"Storage backend error: {e}")
                
                # Mark task as done
                self.event_queue.task_done()
                
            except asyncio.TimeoutError:
                # No events in queue, continue
                continue
            except Exception as e:
                logger.error(f"Event processing error: {e}")
    
    def _dict_to_security_event(self, data: Dict[str, Any]) -> SecurityEvent:
        """Convert dictionary to SecurityEvent."""
        return SecurityEvent(
            event_type=SecurityEventType(data.get('event_type', 'system_error')),
            severity=self._determine_severity(data),
            timestamp=datetime.utcnow(),
            client_ip=data.get('client_ip', 'unknown'),
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            endpoint=data.get('endpoint'),
            method=data.get('method'),
            user_agent=data.get('user_agent'),
            details=data.get('details'),
            threat_score=data.get('threat_score', 0.0),
            success=data.get('success', True),
            error_message=data.get('error_message')
        )
    
    def _determine_severity(self, data: Dict[str, Any]) -> SeverityLevel:
        """Determine event severity based on data."""
        event_type = data.get('event_type')
        
        # High severity events
        high_severity_events = [
            'threat_detected',
            'vulnerability_exploit',
            'privilege_escalation',
            'malicious_request'
        ]
        
        # Critical severity events
        critical_events = [
            'system_compromise',
            'data_breach',
            'unauthorized_access'
        ]
        
        if event_type in critical_events:
            return SeverityLevel.CRITICAL
        elif event_type in high_severity_events:
            return SeverityLevel.HIGH
        elif data.get('threat_score', 0) > 5.0:
            return SeverityLevel.HIGH
        elif not data.get('success', True):
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    async def _send_immediate_alert(self, event: SecurityEvent):
        """Send immediate alert for critical events."""
        # Implement alerting mechanism (email, Slack, etc.)
        logger.critical(
            f"SECURITY ALERT: {event.event_type.value} from {event.client_ip}",
            extra={
                "event_type": event.event_type.value,
                "client_ip": event.client_ip,
                "severity": event.severity.name,
                "details": event.details
            }
        )
    
    async def get_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[SecurityEventType]] = None,
        client_ip: Optional[str] = None,
        limit: int = 100
    ) -> List[SecurityEvent]:
        """Retrieve security events."""
        # Use first storage backend for queries
        if self.storage_backends:
            return await self.storage_backends[0].get_events(
                start_time, end_time, event_types, client_ip, limit
            )
        return []
    
    async def get_threat_analysis(
        self,
        time_window: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """Get comprehensive threat analysis."""
        if not self.threat_analyzer:
            return {"error": "Threat analysis not enabled"}
        
        # Get recent events
        end_time = datetime.utcnow()
        start_time = end_time - time_window
        
        events = await self.get_events(start_time=start_time, end_time=end_time)
        
        # Perform analysis
        analysis = await self.threat_analyzer.analyze_events(events, time_window)
        
        # Add summary statistics
        if self.storage_backends:
            summary = await self.storage_backends[0].get_threat_summary(time_window)
            analysis['summary'] = summary
        
        return analysis
    
    async def export_compliance_report(
        self,
        start_time: datetime,
        end_time: datetime,
        format: str = "json"
    ) -> str:
        """Export compliance report for specified time period."""
        events = await self.get_events(start_time=start_time, end_time=end_time)
        
        report = {
            "report_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "total_events": len(events),
            "events_by_type": {},
            "events_by_severity": {},
            "compliance_metrics": {
                "authentication_events": 0,
                "authorization_failures": 0,
                "data_access_events": 0,
                "system_changes": 0
            }
        }
        
        # Process events
        for event in events:
            # Count by type
            event_type = event.event_type.value
            report["events_by_type"][event_type] = (
                report["events_by_type"].get(event_type, 0) + 1
            )
            
            # Count by severity
            severity = event.severity.name
            report["events_by_severity"][severity] = (
                report["events_by_severity"].get(severity, 0) + 1
            )
            
            # Compliance metrics
            if event.event_type in [
                SecurityEventType.AUTHENTICATION_SUCCESS,
                SecurityEventType.AUTHENTICATION_FAILURE
            ]:
                report["compliance_metrics"]["authentication_events"] += 1
            
            if event.event_type == SecurityEventType.AUTHORIZATION_FAILURE:
                report["compliance_metrics"]["authorization_failures"] += 1
            
            if event.event_type == SecurityEventType.DATA_ACCESS:
                report["compliance_metrics"]["data_access_events"] += 1
            
            if event.event_type == SecurityEventType.CONFIGURATION_CHANGE:
                report["compliance_metrics"]["system_changes"] += 1
        
        if format == "json":
            return json.dumps(report, indent=2)
        else:
            # Implement other formats (CSV, XML, etc.)
            return json.dumps(report, indent=2)


# Global audit logger instance
_global_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(
    storage_backends: Optional[List[AuditStorage]] = None,
    enable_analysis: bool = True
) -> AuditLogger:
    """Get the global audit logger instance."""
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = AuditLogger(
            storage_backends=storage_backends,
            enable_analysis=enable_analysis
        )
    return _global_audit_logger


# Convenience functions
async def log_security_event(event_data: Union[Dict[str, Any], SecurityEvent]) -> bool:
    """Log security event (convenience function)."""
    audit_logger = get_audit_logger()
    return await audit_logger.log_security_event(event_data)


async def get_threat_analysis(time_window: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
    """Get threat analysis (convenience function)."""
    audit_logger = get_audit_logger()
    return await audit_logger.get_threat_analysis(time_window)