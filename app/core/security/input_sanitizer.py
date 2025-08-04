"""
Input sanitization and validation security module.
Prevents XSS, SQL injection, and other input-based attacks.
"""

import re
import html
import json
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote, unquote
import bleach

from app.utils.logger import logger


class InputSanitizer:
    """
    Comprehensive input sanitization system.
    Provides protection against various input-based security vulnerabilities.
    """
    
    # XSS prevention patterns
    XSS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'vbscript:',
        r'onload\s*=',
        r'onerror\s*=',
        r'onclick\s*=',
        r'onmouseover\s*=',
        r'<iframe[^>]*>.*?</iframe>',
        r'<object[^>]*>.*?</object>',
        r'<embed[^>]*>.*?</embed>',
    ]
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)',
        r'(\bunion\s+select\b)',
        r'(\bor\s+1\s*=\s*1\b)',
        r'(\band\s+1\s*=\s*1\b)',
        r'(\bor\s+\'1\'\s*=\s*\'1\'\b)',
        r'(\band\s+\'1\'\s*=\s*\'1\'\b)',
        r'(\bselect\s+\*\s+from\b)',
        r'(\bdrop\s+table\b)',
        r'(\btruncate\s+table\b)',
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r'(\b(system|exec|eval|shell_exec|passthru|popen|proc_open)\b)',
        r'(\$\(.*\))',
        r'(`.*`)',
        r'(\|\|)',
        r'(&&)',
        r'(;\s*rm\b)',
        r'(;\s*cat\b)',
        r'(;\s*ls\b)',
        r'(;\s*pwd\b)',
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r'\.\./',
        r'\.\.\/',
        r'\.\.\\',
        r'%2e%2e%2f',
        r'%2e%2e/',
        r'..%2f',
        r'%2e%2e%5c',
    ]
    
    def __init__(self):
        self._compile_patterns()
        logger.info("Input sanitizer initialized")
    
    def _compile_patterns(self):
        """Compile regex patterns for performance."""
        self.xss_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.XSS_PATTERNS]
        self.sql_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.SQL_INJECTION_PATTERNS]
        self.cmd_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.COMMAND_INJECTION_PATTERNS]
        self.path_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.PATH_TRAVERSAL_PATTERNS]
    
    def sanitize_string(self, input_string: str, strict: bool = False) -> str:
        """
        Sanitize a string input for XSS and other attacks.
        
        Args:
            input_string: The string to sanitize
            strict: If True, uses more aggressive sanitization
            
        Returns:
            Sanitized string
        """
        if not isinstance(input_string, str):
            return str(input_string)
        
        # HTML escape
        sanitized = html.escape(input_string)
        
        if strict:
            # More aggressive sanitization using bleach
            sanitized = bleach.clean(
                sanitized,
                tags=[],  # No HTML tags allowed
                attributes={},  # No attributes allowed
                strip=True
            )
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Remove control characters (except common ones like \n, \r, \t)
        sanitized = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        return sanitized
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename to prevent path traversal and other attacks.
        
        Args:
            filename: The filename to sanitize
            
        Returns:
            Sanitized filename
        """
        if not isinstance(filename, str):
            filename = str(filename)
        
        # Remove path separators and traversal attempts
        sanitized = re.sub(r'[/\\:]', '_', filename)
        sanitized = re.sub(r'\.\.+', '.', sanitized)
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1F\x7F]', '', sanitized)
        
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
            sanitized = name[:250] + ('.' + ext if ext else '')
        
        # Ensure it's not empty after sanitization
        if not sanitized:
            sanitized = 'sanitized_file'
        
        return sanitized
    
    def sanitize_json(self, json_data: Union[str, Dict, List]) -> Union[Dict, List, None]:
        """
        Sanitize JSON data recursively.
        
        Args:
            json_data: JSON data to sanitize
            
        Returns:
            Sanitized JSON data or None if invalid
        """
        try:
            if isinstance(json_data, str):
                # Parse JSON string
                data = json.loads(json_data)
            else:
                data = json_data
            
            return self._sanitize_json_recursive(data)
            
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Invalid JSON data: {e}")
            return None
    
    def _sanitize_json_recursive(self, data: Any) -> Any:
        """Recursively sanitize JSON data."""
        if isinstance(data, dict):
            return {
                self.sanitize_string(key): self._sanitize_json_recursive(value)
                for key, value in data.items()
                if isinstance(key, (str, int, float, bool))
            }
        elif isinstance(data, list):
            return [self._sanitize_json_recursive(item) for item in data]
        elif isinstance(data, str):
            return self.sanitize_string(data)
        elif isinstance(data, (int, float, bool, type(None))):
            return data
        else:
            # Convert unknown types to string and sanitize
            return self.sanitize_string(str(data))
    
    def detect_xss(self, input_string: str) -> bool:
        """
        Detect potential XSS attacks in input.
        
        Args:
            input_string: String to check
            
        Returns:
            True if XSS detected, False otherwise
        """
        if not isinstance(input_string, str):
            return False
        
        # Convert to lowercase for case-insensitive detection
        lower_input = input_string.lower()
        
        # Check against XSS patterns
        for pattern in self.xss_regex:
            if pattern.search(lower_input):
                return True
        
        return False
    
    def detect_sql_injection(self, input_string: str) -> bool:
        """
        Detect potential SQL injection attacks.
        
        Args:
            input_string: String to check
            
        Returns:
            True if SQL injection detected, False otherwise
        """
        if not isinstance(input_string, str):
            return False
        
        # Convert to lowercase for case-insensitive detection
        lower_input = input_string.lower()
        
        # Check against SQL injection patterns
        for pattern in self.sql_regex:
            if pattern.search(lower_input):
                return True
        
        return False
    
    def detect_command_injection(self, input_string: str) -> bool:
        """
        Detect potential command injection attacks.
        
        Args:
            input_string: String to check
            
        Returns:
            True if command injection detected, False otherwise
        """
        if not isinstance(input_string, str):
            return False
        
        # Check against command injection patterns
        for pattern in self.cmd_regex:
            if pattern.search(input_string):
                return True
        
        return False
    
    def detect_path_traversal(self, input_string: str) -> bool:
        """
        Detect potential path traversal attacks.
        
        Args:
            input_string: String to check
            
        Returns:
            True if path traversal detected, False otherwise
        """
        if not isinstance(input_string, str):
            return False
        
        # Check against path traversal patterns
        for pattern in self.path_regex:
            if pattern.search(input_string):
                return True
        
        return False
    
    def comprehensive_check(self, input_string: str) -> Dict[str, bool]:
        """
        Perform comprehensive security check on input.
        
        Args:
            input_string: String to check
            
        Returns:
            Dictionary with results of all security checks
        """
        return {
            'xss': self.detect_xss(input_string),
            'sql_injection': self.detect_sql_injection(input_string),
            'command_injection': self.detect_command_injection(input_string),
            'path_traversal': self.detect_path_traversal(input_string),
        }
    
    def is_safe_input(self, input_string: str) -> bool:
        """
        Check if input is safe (no detected vulnerabilities).
        
        Args:
            input_string: String to check
            
        Returns:
            True if input is safe, False if vulnerabilities detected
        """
        results = self.comprehensive_check(input_string)
        return not any(results.values())
    
    def sanitize_dict(self, data: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
        """
        Sanitize all string values in a dictionary.
        
        Args:
            data: Dictionary to sanitize
            strict: If True, uses strict sanitization
            
        Returns:
            Sanitized dictionary
        """
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            safe_key = self.sanitize_string(str(key), strict=strict)
            
            # Sanitize value based on type
            if isinstance(value, str):
                sanitized[safe_key] = self.sanitize_string(value, strict=strict)
            elif isinstance(value, dict):
                sanitized[safe_key] = self.sanitize_dict(value, strict=strict)
            elif isinstance(value, list):
                sanitized[safe_key] = self.sanitize_list(value, strict=strict)
            else:
                sanitized[safe_key] = value
        
        return sanitized
    
    def sanitize_list(self, data: List[Any], strict: bool = False) -> List[Any]:
        """
        Sanitize all string values in a list.
        
        Args:
            data: List to sanitize
            strict: If True, uses strict sanitization
            
        Returns:
            Sanitized list
        """
        sanitized = []
        
        for item in data:
            if isinstance(item, str):
                sanitized.append(self.sanitize_string(item, strict=strict))
            elif isinstance(item, dict):
                sanitized.append(self.sanitize_dict(item, strict=strict))
            elif isinstance(item, list):
                sanitized.append(self.sanitize_list(item, strict=strict))
            else:
                sanitized.append(item)
        
        return sanitized
    
    def validate_and_sanitize(self, input_string: str, strict: bool = False) -> tuple[str, bool]:
        """
        Validate input for security issues and return sanitized version.
        
        Args:
            input_string: String to validate and sanitize
            strict: If True, uses strict sanitization
            
        Returns:
            Tuple of (sanitized_string, is_safe)
        """
        is_safe = self.is_safe_input(input_string)
        sanitized = self.sanitize_string(input_string, strict=strict)
        
        if not is_safe:
            logger.warning(
                "Potentially malicious input detected and sanitized",
                extra={
                    "original_input": input_string[:100],  # Log first 100 chars
                    "security_checks": self.comprehensive_check(input_string)
                }
            )
        
        return sanitized, is_safe


# Global sanitizer instance
_global_sanitizer: Optional[InputSanitizer] = None


def get_input_sanitizer() -> InputSanitizer:
    """Get the global input sanitizer instance."""
    global _global_sanitizer
    if _global_sanitizer is None:
        _global_sanitizer = InputSanitizer()
    return _global_sanitizer


# Convenience functions
def sanitize_string(input_string: str, strict: bool = False) -> str:
    """Sanitize a string (convenience function)."""
    return get_input_sanitizer().sanitize_string(input_string, strict=strict)


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename (convenience function)."""
    return get_input_sanitizer().sanitize_filename(filename)


def is_safe_input(input_string: str) -> bool:
    """Check if input is safe (convenience function).""" 
    return get_input_sanitizer().is_safe_input(input_string)