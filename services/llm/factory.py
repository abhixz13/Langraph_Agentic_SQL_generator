"""
LLM Client Factory

This module provides factory functions to create LLM clients from configuration.
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from .errors import LLMConfigError, LLMError

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM operations"""
    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0


def load_llm_config() -> LLMConfig:
    """Load LLM configuration from environment variables"""
    try:
        config = LLMConfig()
        
        # Override with environment variables if present
        if os.getenv("LLM_MODEL"):
            config.model = os.getenv("LLM_MODEL")
        if os.getenv("LLM_TEMPERATURE"):
            config.temperature = float(os.getenv("LLM_TEMPERATURE"))
        if os.getenv("LLM_MAX_TOKENS"):
            config.max_tokens = int(os.getenv("LLM_MAX_TOKENS"))
        if os.getenv("LLM_TIMEOUT"):
            config.timeout = int(os.getenv("LLM_TIMEOUT"))
        if os.getenv("LLM_MAX_RETRIES"):
            config.max_retries = int(os.getenv("LLM_MAX_RETRIES"))
        if os.getenv("LLM_RETRY_DELAY"):
            config.retry_delay = float(os.getenv("LLM_RETRY_DELAY"))
        
        return config
        
    except (ValueError, TypeError) as e:
        raise LLMConfigError(f"Invalid LLM configuration: {str(e)}")


def create_llm_client(
    api_key: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    **kwargs
):
    """
    Create a new LLM client instance
    
    Args:
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        config: LLM configuration (defaults to environment-based config)
        **kwargs: Additional configuration overrides
        
    Returns:
        Configured LLMClient instance
        
    Raises:
        LLMConfigError: If configuration is invalid
    """
    try:
        # Get API key
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise LLMConfigError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Get configuration
        if not config:
            config = load_llm_config()
        
        # Apply any overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Import here to avoid circular import
        from .client import LLMClient
        
        # Create client
        client = LLMClient(api_key=api_key, config=config)
        logger.info(f"LLM client created with model: {config.model}")
        
        return client
        
    except Exception as e:
        if isinstance(e, LLMConfigError):
            raise
        raise LLMConfigError(f"Failed to create LLM client: {str(e)}")


# Global client instance for convenience
_global_client = None


def get_llm_client():
    """
    Get or create the global LLM client instance
    
    Returns:
        Global LLMClient instance
        
    Raises:
        LLMConfigError: If client creation fails
    """
    global _global_client
    
    if _global_client is None:
        _global_client = create_llm_client()
    
    return _global_client


def reset_llm_client():
    """Reset the global LLM client (useful for testing)"""
    global _global_client
    _global_client = None
