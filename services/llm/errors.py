"""
LLM Service Error Handling

This module provides typed exceptions and error taxonomy for LLM operations.
"""

from typing import Optional, Dict, Any


class LLMError(Exception):
    """Base exception for all LLM-related errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class LLMConfigError(LLMError):
    """Configuration-related errors"""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        details = {"config_key": config_key} if config_key else {}
        super().__init__(f"LLM Configuration Error: {message}", details)


class LLMResponseError(LLMError):
    """Response parsing and validation errors"""
    
    def __init__(self, message: str, response: Optional[str] = None, expected_format: Optional[str] = None):
        details = {
            "response": response,
            "expected_format": expected_format
        }
        super().__init__(f"LLM Response Error: {message}", details)


class LLMTimeoutError(LLMError):
    """Timeout errors during LLM calls"""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None):
        details = {"timeout_seconds": timeout_seconds} if timeout_seconds else {}
        super().__init__(f"LLM Timeout Error: {message}", details)


class LLMRetryError(LLMError):
    """Errors after all retry attempts exhausted"""
    
    def __init__(self, message: str, attempts: int, last_error: Optional[str] = None):
        details = {
            "attempts": attempts,
            "last_error": last_error
        }
        super().__init__(f"LLM Retry Error: {message}", details)


class LLMJSONParseError(LLMResponseError):
    """JSON parsing errors"""
    
    def __init__(self, message: str, raw_response: str, json_error: Optional[str] = None):
        details = {
            "raw_response": raw_response,
            "json_error": json_error
        }
        super().__init__(f"LLM JSON Parse Error: {message}", details, "JSON")


# Error taxonomy for categorization
class ErrorCategory:
    """Error categories for LLM operations"""
    
    CONFIGURATION = "configuration"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RESPONSE_PARSE = "response_parse"
    VALIDATION = "validation"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    UNKNOWN = "unknown"


def categorize_error(error: Exception) -> str:
    """Categorize an error based on its type and content"""
    if isinstance(error, LLMConfigError):
        return ErrorCategory.CONFIGURATION
    elif isinstance(error, LLMTimeoutError):
        return ErrorCategory.TIMEOUT
    elif isinstance(error, LLMResponseError):
        return ErrorCategory.RESPONSE_PARSE
    elif isinstance(error, LLMRetryError):
        return ErrorCategory.NETWORK
    else:
        return ErrorCategory.UNKNOWN
