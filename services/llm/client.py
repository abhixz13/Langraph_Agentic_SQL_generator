"""
LLM Client

This module provides the main LLM client with three core methods:
- text(): Generate text responses
- json(): Generate structured JSON responses
- embed(): Generate embeddings
"""

import json
import logging
import time
import re
from typing import Dict, Any, List, Optional, Union

from openai import OpenAI
from openai.types.chat import ChatCompletion

from .errors import (
    LLMError, LLMTimeoutError, LLMRetryError, 
    LLMJSONParseError, LLMResponseError
)
from .factory import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Main LLM client for SQL Assistant operations"""
    
    def __init__(self, api_key: str, config: LLMConfig):
        """Initialize LLM client with configuration"""
        self.api_key = api_key
        self.config = config
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            timeout=self.config.timeout
        )
        
        logger.info(f"LLM Client initialized with model: {self.config.model}")
    
    def text(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text response from LLM
        
        Args:
            prompt: The user prompt
            system_message: Optional system message
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional parameters
            
        Returns:
            LLM response as string
            
        Raises:
            LLMTimeoutError: If request times out
            LLMRetryError: If all retry attempts fail
            LLMError: For other errors
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            **kwargs
        }
        
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"LLM text call attempt {attempt + 1}/{self.config.max_retries}")
                
                response = self.client.chat.completions.create(**params)
                
                if response.choices and response.choices[0].message:
                    result = response.choices[0].message.content
                    logger.debug(f"LLM text call successful, response length: {len(result)}")
                    return result
                else:
                    raise LLMResponseError("Empty response from LLM")
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM text call attempt {attempt + 1} failed: {last_error}")
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All LLM text call attempts failed: {last_error}")
                    raise LLMRetryError(
                        f"All retry attempts failed: {last_error}",
                        attempts=self.config.max_retries,
                        last_error=last_error
                    )
    
    def json(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        expected_schema: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON response from LLM
        
        Args:
            prompt: The user prompt
            system_message: Optional system message
            expected_schema: Expected JSON schema for validation
            temperature: Override default temperature
            **kwargs: Additional parameters
            
        Returns:
            Parsed JSON response as dictionary
            
        Raises:
            LLMJSONParseError: If JSON parsing fails
            LLMError: For other errors
        """
        # Add JSON formatting instruction to prompt
        json_prompt = f"{prompt}\n\nPlease respond with valid JSON only."
        
        if expected_schema:
            schema_str = json.dumps(expected_schema, indent=2)
            json_prompt += f"\n\nExpected schema:\n{schema_str}"
        
        try:
            response = self.text(json_prompt, system_message, temperature, **kwargs)
            
            # Try to extract JSON from response
            response = response.strip()
            
            # Handle cases where response might have markdown formatting
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            result = json.loads(response)
            logger.debug(f"JSON response parsed successfully: {type(result)}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"Raw response: {response}")
            raise LLMJSONParseError(
                f"Invalid JSON response from LLM: {str(e)}",
                raw_response=response,
                json_error=str(e)
            )
        except Exception as e:
            if isinstance(e, (LLMJSONParseError, LLMRetryError, LLMTimeoutError)):
                raise
            raise LLMError(f"JSON generation failed: {str(e)}")
    
    def embed(
        self,
        text: str,
        model: str = "text-embedding-3-small"
    ) -> List[float]:
        """
        Generate embeddings for text
        
        Args:
            text: Text to embed
            model: Embedding model to use
            
        Returns:
            List of embedding values
            
        Raises:
            LLMError: For embedding errors
        """
        try:
            logger.debug(f"Generating embeddings with model: {model}")
            
            response = self.client.embeddings.create(
                model=model,
                input=text
            )
            
            if response.data and response.data[0].embedding:
                embedding = response.data[0].embedding
                logger.debug(f"Embedding generated successfully, length: {len(embedding)}")
                return embedding
            else:
                raise LLMResponseError("Empty embedding response")
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise LLMError(f"Embedding generation failed: {str(e)}")
    
    def batch_embed(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small"
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
            
        Returns:
            List of embedding lists
            
        Raises:
            LLMError: For embedding errors
        """
        try:
            logger.debug(f"Generating batch embeddings for {len(texts)} texts")
            
            response = self.client.embeddings.create(
                model=model,
                input=texts
            )
            
            if response.data:
                embeddings = [data.embedding for data in response.data]
                logger.debug(f"Batch embeddings generated successfully, count: {len(embeddings)}")
                return embeddings
            else:
                raise LLMResponseError("Empty batch embedding response")
                
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {str(e)}")
            raise LLMError(f"Batch embedding generation failed: {str(e)}")
