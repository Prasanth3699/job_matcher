"""
Unified JSON serialization utilities for the resume job matcher service.

This module provides centralized JSON serialization functionality to eliminate
code duplication across the application. It handles various data types including
Pydantic models, datetime objects, Decimal values, and URL objects.
"""

import json
import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Union, Optional
from uuid import UUID

# Local imports
from app.core.constants import ErrorCodes, ErrorMessages

logger = logging.getLogger(__name__)


class SerializationError(Exception):
    """Custom exception for serialization errors."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)


class JSONSerializer:
    """
    Centralized JSON serialization utility class.
    
    Provides methods to convert various Python objects to JSON-serializable format
    with comprehensive error handling and logging.
    """
    
    @staticmethod
    def make_serializable(obj: Any) -> Any:
        """
        Convert objects to JSON-serializable format.
        
        Handles HttpUrl, Decimal, datetime, Pydantic models, and other problematic types.
        
        Args:
            obj: The object to make serializable
            
        Returns:
            JSON-serializable representation of the object
            
        Raises:
            SerializationError: If serialization fails critically
        """
        if obj is None:
            return None

        try:
            # Handle Pydantic models first (most common case)
            if hasattr(obj, "model_dump"):
                try:
                    return JSONSerializer.make_serializable(obj.model_dump())
                except Exception as e:
                    logger.warning(f"Failed to serialize Pydantic model using model_dump: {e}")
                    # Fallback to dict conversion
                    try:
                        return JSONSerializer.make_serializable(dict(obj))
                    except Exception:
                        logger.error(f"Failed to serialize Pydantic model: {e}")
                        return str(obj)

            # Handle URL types (HttpUrl, AnyUrl, etc.)
            if hasattr(obj, "__class__"):
                class_name = str(obj.__class__)
                if any(url_type in class_name for url_type in ["HttpUrl", "AnyUrl", "Url"]):
                    try:
                        return str(obj)
                    except Exception as e:
                        logger.warning(f"Failed to convert URL object to string: {e}")
                        return str(type(obj))

            # Handle basic types (fast path)
            if isinstance(obj, (str, int, float, bool)):
                return obj
            
            # Handle None explicitly
            if obj is None:
                return None

            # Handle datetime objects
            if isinstance(obj, (datetime, date)):
                try:
                    return obj.isoformat()
                except Exception as e:
                    logger.warning(f"Failed to convert datetime to ISO format: {e}")
                    return str(obj)

            # Handle UUID objects
            if isinstance(obj, UUID):
                return str(obj)

            # Handle Decimal objects
            if isinstance(obj, Decimal):
                try:
                    return float(obj)
                except (ValueError, OverflowError) as e:
                    logger.warning(f"Failed to convert Decimal to float: {e}")
                    return str(obj)

            # Handle dictionaries
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    try:
                        serialized_key = str(key) if not isinstance(key, str) else key
                        result[serialized_key] = JSONSerializer.make_serializable(value)
                    except Exception as e:
                        logger.warning(f"Failed to serialize dict item {key}: {e}")
                        result[str(key)] = str(value)
                return result

            # Handle lists and tuples
            if isinstance(obj, (list, tuple)):
                result = []
                for item in obj:
                    try:
                        result.append(JSONSerializer.make_serializable(item))
                    except Exception as e:
                        logger.warning(f"Failed to serialize list/tuple item: {e}")
                        result.append(str(item))
                return result

            # Handle sets
            if isinstance(obj, set):
                try:
                    return [JSONSerializer.make_serializable(item) for item in obj]
                except Exception as e:
                    logger.warning(f"Failed to serialize set: {e}")
                    return [str(item) for item in obj]

            # Handle bytes
            if isinstance(obj, bytes):
                try:
                    # Try to decode as UTF-8
                    return obj.decode('utf-8')
                except UnicodeDecodeError:
                    # Fallback to base64 encoding
                    import base64
                    return base64.b64encode(obj).decode('ascii')

            # Test if object is already JSON serializable
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                pass

            # Handle objects with custom serialization methods
            if hasattr(obj, '__dict__'):
                try:
                    return JSONSerializer.make_serializable(obj.__dict__)
                except Exception as e:
                    logger.warning(f"Failed to serialize object __dict__: {e}")

            # Final fallback: convert to string
            try:
                return str(obj)
            except Exception as e:
                logger.error(f"Failed to convert object to string: {e}")
                return f"<{type(obj).__name__} object>"

        except Exception as e:
            logger.error(f"Critical serialization error for {type(obj)}: {e}")
            raise SerializationError(
                f"Failed to serialize object of type {type(obj).__name__}",
                original_error=e
            )

    @staticmethod
    def serialize_to_json(obj: Any, **json_kwargs) -> str:
        """
        Serialize object to JSON string.
        
        Args:
            obj: Object to serialize
            **json_kwargs: Additional arguments for json.dumps()
            
        Returns:
            JSON string representation
            
        Raises:
            SerializationError: If JSON serialization fails
        """
        try:
            serializable_obj = JSONSerializer.make_serializable(obj)
            
            # Default JSON serialization options
            default_kwargs = {
                'ensure_ascii': False,
                'separators': (',', ':'),
                'sort_keys': False
            }
            default_kwargs.update(json_kwargs)
            
            return json.dumps(serializable_obj, **default_kwargs)
            
        except Exception as e:
            logger.error(f"Failed to serialize to JSON: {e}")
            raise SerializationError(
                "Failed to convert object to JSON string",
                original_error=e
            )

    @staticmethod
    def deserialize_from_json(json_str: str) -> Any:
        """
        Deserialize JSON string to Python object.
        
        Args:
            json_str: JSON string to deserialize
            
        Returns:
            Deserialized Python object
            
        Raises:
            SerializationError: If JSON deserialization fails
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to deserialize JSON: {e}")
            raise SerializationError(
                f"Invalid JSON format: {e.msg}",
                original_error=e
            )
        except Exception as e:
            logger.error(f"Unexpected error during JSON deserialization: {e}")
            raise SerializationError(
                "Failed to deserialize JSON string",
                original_error=e
            )

    @staticmethod
    def serialize_for_cache(obj: Any) -> str:
        """
        Serialize object for caching with compression-friendly format.
        
        Args:
            obj: Object to serialize for caching
            
        Returns:
            Compact JSON string suitable for caching
        """
        return JSONSerializer.serialize_to_json(
            obj,
            separators=(',', ':'),
            sort_keys=True,
            ensure_ascii=True
        )

    @staticmethod
    def serialize_for_logging(obj: Any, max_length: int = 1000) -> str:
        """
        Serialize object for logging with length limitation.
        
        Args:
            obj: Object to serialize for logging
            max_length: Maximum length of serialized string
            
        Returns:
            JSON string truncated if necessary
        """
        try:
            json_str = JSONSerializer.serialize_to_json(obj, indent=2)
            if len(json_str) > max_length:
                truncated = json_str[:max_length - 10]
                return f"{truncated}...[truncated]"
            return json_str
        except Exception as e:
            logger.warning(f"Failed to serialize for logging: {e}")
            return f"<Serialization failed: {str(obj)[:100]}...>"


class BatchSerializer:
    """
    Utility for batch serialization operations.
    
    Useful for processing multiple objects with error isolation.
    """
    
    @staticmethod
    def serialize_batch(objects: List[Any]) -> List[Dict[str, Any]]:
        """
        Serialize a batch of objects with error isolation.
        
        Args:
            objects: List of objects to serialize
            
        Returns:
            List of serialization results with error information
        """
        results = []
        
        for i, obj in enumerate(objects):
            try:
                serialized = JSONSerializer.make_serializable(obj)
                results.append({
                    'index': i,
                    'success': True,
                    'data': serialized,
                    'error': None
                })
            except Exception as e:
                logger.warning(f"Failed to serialize object at index {i}: {e}")
                results.append({
                    'index': i,
                    'success': False,
                    'data': None,
                    'error': str(e)
                })
        
        return results

    @staticmethod
    def get_successful_results(batch_results: List[Dict[str, Any]]) -> List[Any]:
        """
        Extract successfully serialized objects from batch results.
        
        Args:
            batch_results: Results from serialize_batch()
            
        Returns:
            List of successfully serialized objects
        """
        return [
            result['data']
            for result in batch_results
            if result['success'] and result['data'] is not None
        ]


class ValidationSerializer:
    """
    Serializer with built-in validation.
    
    Ensures serialized data meets specific criteria.
    """
    
    @staticmethod
    def serialize_with_validation(
        obj: Any,
        max_size_bytes: Optional[int] = None,
        required_fields: Optional[List[str]] = None
    ) -> str:
        """
        Serialize object with validation checks.
        
        Args:
            obj: Object to serialize
            max_size_bytes: Maximum size of serialized JSON
            required_fields: Required fields in serialized object
            
        Returns:
            Validated JSON string
            
        Raises:
            SerializationError: If validation fails
        """
        # Serialize the object
        json_str = JSONSerializer.serialize_to_json(obj)
        
        # Validate size if specified
        if max_size_bytes and len(json_str.encode('utf-8')) > max_size_bytes:
            raise SerializationError(
                f"Serialized object exceeds maximum size of {max_size_bytes} bytes"
            )
        
        # Validate required fields if specified
        if required_fields:
            try:
                data = json.loads(json_str)
                if isinstance(data, dict):
                    missing_fields = [
                        field for field in required_fields
                        if field not in data
                    ]
                    if missing_fields:
                        raise SerializationError(
                            f"Missing required fields: {missing_fields}"
                        )
            except json.JSONDecodeError:
                raise SerializationError("Invalid JSON format for field validation")
        
        return json_str


# Convenience functions for backward compatibility and ease of use
def make_json_serializable(obj: Any) -> Any:
    """
    Convenience function for JSON serialization.
    
    This function provides backward compatibility with existing code
    while using the new centralized serialization logic.
    """
    return JSONSerializer.make_serializable(obj)


def serialize_to_json(obj: Any, **kwargs) -> str:
    """Convenience function for JSON string serialization."""
    return JSONSerializer.serialize_to_json(obj, **kwargs)


def deserialize_from_json(json_str: str) -> Any:
    """Convenience function for JSON deserialization."""
    return JSONSerializer.deserialize_from_json(json_str)