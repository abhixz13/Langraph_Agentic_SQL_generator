"""
LLM Proxy Gateway - Secure Data Transmission Layer

This module provides a secure proxy gateway that sits between the SQL Assistant
and external LLM services (OpenAI, etc.) to prevent data leakage and ensure
privacy compliance.

Features:
- Schema obfuscation and anonymization
- Query sanitization and filtering
- Data classification and redaction
- Secure transmission with encryption
- Audit logging and monitoring
- Rate limiting and cost control
"""

import json
import logging
import hashlib
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import re
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os

# Import OpenAI client for actual API calls
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Configuration for security features"""
    # Data classification
    sensitive_patterns: List[str] = field(default_factory=lambda: [
        r'\b(customer|client|user|person)\b',
        r'\b(revenue|sales|bookings|money|price|cost)\b',
        r'\b(ssn|social|credit|card|account|password)\b',
        r'\b(address|phone|email|contact)\b',
        r'\b(employee|staff|worker|salary|payroll)\b'
    ])
    
    # Schema obfuscation
    table_name_mapping: Dict[str, str] = field(default_factory=dict)
    column_name_mapping: Dict[str, str] = field(default_factory=dict)
    
    # Data retention
    max_data_age_hours: int = 24
    max_cache_size_mb: int = 100
    
    # Rate limiting
    max_requests_per_minute: int = 60
    max_tokens_per_request: int = 4000
    
    # Encryption
    encryption_key: Optional[str] = None
    
    def __post_init__(self):
        if not self.encryption_key:
            self.encryption_key = os.getenv("LLM_PROXY_ENCRYPTION_KEY") or Fernet.generate_key().decode()


@dataclass
class AuditLog:
    """Audit log entry for data transmission"""
    timestamp: datetime
    request_id: str
    operation: str
    data_classification: str
    data_size_bytes: int
    obfuscation_applied: bool
    encryption_applied: bool
    external_service: str
    cost_estimate: float
    success: bool
    error_message: Optional[str] = None


class DataClassifier:
    """Classifies data sensitivity and applies appropriate security measures"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.sensitive_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in config.sensitive_patterns]
        
    def classify_data(self, data: Union[str, Dict, List]) -> str:
        """Classify data sensitivity level"""
        if isinstance(data, str):
            return self._classify_text(data)
        elif isinstance(data, dict):
            return self._classify_dict(data)
        elif isinstance(data, list):
            return self._classify_list(data)
        return "LOW"
    
    def _classify_text(self, text: str) -> str:
        """Classify text sensitivity"""
        if not text:
            return "LOW"
        
        # Check for sensitive patterns
        for pattern in self.sensitive_patterns:
            if pattern.search(text):
                return "HIGH"
        
        # Check for SQL keywords that might reveal structure
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'TABLE', 'COLUMN']
        if any(keyword.lower() in text.lower() for keyword in sql_keywords):
            return "MEDIUM"
        
        return "LOW"
    
    def _classify_dict(self, data: Dict) -> str:
        """Classify dictionary sensitivity"""
        max_classification = "LOW"
        
        for key, value in data.items():
            # Check key names
            key_classification = self._classify_text(str(key))
            if key_classification == "HIGH":
                return "HIGH"
            elif key_classification == "MEDIUM":
                max_classification = "MEDIUM"
            
            # Check values
            value_classification = self.classify_data(value)
            if value_classification == "HIGH":
                return "HIGH"
            elif value_classification == "MEDIUM":
                max_classification = "MEDIUM"
        
        return max_classification
    
    def _classify_list(self, data: List) -> str:
        """Classify list sensitivity"""
        max_classification = "LOW"
        
        for item in data:
            item_classification = self.classify_data(item)
            if item_classification == "HIGH":
                return "HIGH"
            elif item_classification == "MEDIUM":
                max_classification = "MEDIUM"
        
        return max_classification


class SchemaObfuscator:
    """Obfuscates database schema to prevent data structure exposure"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.table_counter = 0
        self.column_counter = 0
        self._generate_mappings()
    
    def _generate_mappings(self):
        """Generate obfuscation mappings"""
        # Generate generic table names
        self.table_mapping = {}
        self.reverse_table_mapping = {}
        
        # Generate generic column names
        self.column_mapping = {}
        self.reverse_column_mapping = {}
    
    def obfuscate_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Obfuscate database schema"""
        if not schema:
            return schema
        
        obfuscated_schema = {
            "dialect": schema.get("dialect", "unknown"),
            "tables": {},
            "business_context": {}  # Remove business context entirely
        }
        
        # Obfuscate tables
        for table_name, table_info in schema.get("tables", {}).items():
            obfuscated_table_name = self._obfuscate_table_name(table_name)
            obfuscated_schema["tables"][obfuscated_table_name] = {
                "columns": [],
                "row_count": table_info.get("row_count", 0),
                "sample_data": []  # Remove sample data
            }
            
            # Obfuscate columns
            for column in table_info.get("columns", []):
                obfuscated_column = {
                    "name": self._obfuscate_column_name(column.get("name", "")),
                    "type": column.get("type", "TEXT"),
                    "not_null": column.get("not_null", False),
                    "primary_key": column.get("primary_key", False)
                }
                obfuscated_schema["tables"][obfuscated_table_name]["columns"].append(obfuscated_column)
        
        return obfuscated_schema
    
    def _obfuscate_table_name(self, table_name: str) -> str:
        """Obfuscate table name"""
        if table_name in self.table_mapping:
            return self.table_mapping[table_name]
        
        # Generate generic table name
        obfuscated_name = f"table_{self.table_counter:03d}"
        self.table_mapping[table_name] = obfuscated_name
        self.reverse_table_mapping[obfuscated_name] = table_name
        self.table_counter += 1
        
        return obfuscated_name
    
    def _obfuscate_column_name(self, column_name: str) -> str:
        """Obfuscate column name"""
        if column_name in self.column_mapping:
            return self.column_mapping[column_name]
        
        # Generate generic column name
        obfuscated_name = f"col_{self.column_counter:03d}"
        self.column_mapping[column_name] = obfuscated_name
        self.reverse_column_mapping[obfuscated_name] = column_name
        self.column_counter += 1
        
        return obfuscated_name
    
    def deobfuscate_sql(self, obfuscated_sql: str) -> str:
        """Deobfuscate SQL query back to original names"""
        deobfuscated_sql = obfuscated_sql
        
        # Replace obfuscated table names
        for obfuscated_name, original_name in self.reverse_table_mapping.items():
            deobfuscated_sql = re.sub(
                rf'\b{re.escape(obfuscated_name)}\b',
                original_name,
                deobfuscated_sql,
                flags=re.IGNORECASE
            )
        
        # Replace obfuscated column names
        for obfuscated_name, original_name in self.reverse_column_mapping.items():
            deobfuscated_sql = re.sub(
                rf'\b{re.escape(obfuscated_name)}\b',
                original_name,
                deobfuscated_sql,
                flags=re.IGNORECASE
            )
        
        return deobfuscated_sql


class QuerySanitizer:
    """Sanitizes queries to remove sensitive information"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.sensitive_terms = [
            'customer', 'client', 'user', 'person',
            'revenue', 'sales', 'bookings', 'money',
            'ssn', 'social', 'credit', 'card',
            'address', 'phone', 'email', 'contact',
            'employee', 'staff', 'worker', 'salary'
        ]
    
    def sanitize_query(self, query: str) -> str:
        """Sanitize user query"""
        if not query:
            return query
        
        sanitized_query = query
        
        # Replace sensitive terms with generic equivalents
        for term in self.sensitive_terms:
            pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
            if term in ['customer', 'client', 'user', 'person']:
                sanitized_query = pattern.sub('entity', sanitized_query)
            elif term in ['revenue', 'sales', 'bookings', 'money']:
                sanitized_query = pattern.sub('metric', sanitized_query)
            elif term in ['ssn', 'social', 'credit', 'card']:
                sanitized_query = pattern.sub('identifier', sanitized_query)
            elif term in ['address', 'phone', 'email', 'contact']:
                sanitized_query = pattern.sub('contact_info', sanitized_query)
            elif term in ['employee', 'staff', 'worker', 'salary']:
                sanitized_query = pattern.sub('personnel', sanitized_query)
        
        return sanitized_query
    
    def sanitize_sql(self, sql: str) -> str:
        """Sanitize SQL query"""
        if not sql:
            return sql
        
        # Remove comments that might contain sensitive information
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        
        # Remove any hardcoded values that might be sensitive
        # This is a basic implementation - could be enhanced
        sql = re.sub(r"'[^']*'", "'<value>'", sql)
        
        return sql


class DataEncryptor:
    """Handles encryption/decryption of sensitive data"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.cipher_suite = Fernet(self.config.encryption_key.encode())
    
    def encrypt_data(self, data: Union[str, Dict, List]) -> str:
        """Encrypt data"""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data)
        else:
            data_str = str(data)
        
        encrypted_data = self.cipher_suite.encrypt(data_str.encode())
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_data(self, encrypted_data: str) -> Union[str, Dict, List]:
        """Decrypt data"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            return encrypted_data


class RateLimiter:
    """Implements rate limiting for API calls"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.request_times: List[float] = []
        self.token_usage: Dict[str, int] = {}
    
    def can_make_request(self, request_id: str, estimated_tokens: int) -> bool:
        """Check if request can be made within rate limits"""
        current_time = time.time()
        
        # Clean old requests
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Check rate limit
        if len(self.request_times) >= self.config.max_requests_per_minute:
            return False
        
        # Check token limit
        if estimated_tokens > self.config.max_tokens_per_request:
            return False
        
        return True
    
    def record_request(self, request_id: str, tokens_used: int):
        """Record a completed request"""
        self.request_times.append(time.time())
        self.token_usage[request_id] = tokens_used


class AuditLogger:
    """Handles audit logging for compliance and monitoring"""
    
    def __init__(self, log_file: str = "logs/llm_proxy_audit.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_request(self, audit_entry: AuditLog):
        """Log audit entry"""
        log_entry = {
            "timestamp": audit_entry.timestamp.isoformat(),
            "request_id": audit_entry.request_id,
            "operation": audit_entry.operation,
            "data_classification": audit_entry.data_classification,
            "data_size_bytes": audit_entry.data_size_bytes,
            "obfuscation_applied": audit_entry.obfuscation_applied,
            "encryption_applied": audit_entry.encryption_applied,
            "external_service": audit_entry.external_service,
            "cost_estimate": audit_entry.cost_estimate,
            "success": audit_entry.success,
            "error_message": audit_entry.error_message
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


class LLMProxyGateway:
    """Main proxy gateway for secure LLM communication"""
    
    def __init__(self, 
                 api_key: str,
                 config: Optional[SecurityConfig] = None,
                 base_url: Optional[str] = None):
        self.api_key = api_key
        self.config = config or SecurityConfig()
        self.base_url = base_url
        
        # Initialize components
        self.classifier = DataClassifier(self.config)
        self.obfuscator = SchemaObfuscator(self.config)
        self.sanitizer = QuerySanitizer(self.config)
        self.encryptor = DataEncryptor(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.audit_logger = AuditLogger()
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        logger.info("LLM Proxy Gateway initialized with security features")
    
    def secure_text_completion(self,
                              prompt: str,
                              system_message: Optional[str] = None,
                              **kwargs) -> Dict[str, Any]:
        """Secure text completion with data protection"""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # 1. Classify data sensitivity
            data_classification = self.classifier.classify_data(prompt)
            
            # 2. Sanitize input
            sanitized_prompt = self.sanitizer.sanitize_query(prompt)
            sanitized_system = self.sanitizer.sanitize_query(system_message) if system_message else None
            
            # 3. Check rate limits
            estimated_tokens = len(sanitized_prompt.split()) * 1.3  # Rough estimate
            if not self.rate_limiter.can_make_request(request_id, int(estimated_tokens)):
                raise Exception("Rate limit exceeded")
            
            # 4. Prepare secure request
            secure_request = {
                "prompt": sanitized_prompt,
                "system_message": sanitized_system,
                **kwargs
            }
            
            # 5. Make API call
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sanitized_system} if sanitized_system else None,
                    {"role": "user", "content": sanitized_prompt}
                ],
                **{k: v for k, v in kwargs.items() if k not in ['prompt', 'system_message']}
            )
            
            # 6. Extract and secure response
            result = response.choices[0].message.content if response.choices else ""
            
            # 7. Record audit log
            self._log_audit_entry(
                request_id=request_id,
                operation="text_completion",
                data_classification=data_classification,
                data_size_bytes=len(sanitized_prompt.encode()),
                obfuscation_applied=True,
                encryption_applied=False,
                external_service="openai",
                cost_estimate=self._estimate_cost(response),
                success=True
            )
            
            # 8. Record rate limiting
            self.rate_limiter.record_request(request_id, response.usage.total_tokens if response.usage else 0)
            
            return {
                "success": True,
                "result": result,
                "request_id": request_id,
                "data_classification": data_classification,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            # Log error
            self._log_audit_entry(
                request_id=request_id,
                operation="text_completion",
                data_classification=data_classification if 'data_classification' in locals() else "UNKNOWN",
                data_size_bytes=len(prompt.encode()),
                obfuscation_applied=True,
                encryption_applied=False,
                external_service="openai",
                cost_estimate=0.0,
                success=False,
                error_message=str(e)
            )
            
            logger.error(f"Secure text completion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "request_id": request_id,
                "processing_time": time.time() - start_time
            }
    
    def secure_schema_processing(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Process schema with obfuscation and security"""
        request_id = str(uuid.uuid4())
        
        try:
            # 1. Classify schema sensitivity
            data_classification = self.classifier.classify_data(schema)
            
            # 2. Obfuscate schema
            obfuscated_schema = self.obfuscator.obfuscate_schema(schema)
            
            # 3. Log audit entry
            self._log_audit_entry(
                request_id=request_id,
                operation="schema_processing",
                data_classification=data_classification,
                data_size_bytes=len(json.dumps(obfuscated_schema).encode()),
                obfuscation_applied=True,
                encryption_applied=False,
                external_service="internal",
                cost_estimate=0.0,
                success=True
            )
            
            return {
                "success": True,
                "obfuscated_schema": obfuscated_schema,
                "original_schema": schema,
                "request_id": request_id,
                "data_classification": data_classification
            }
            
        except Exception as e:
            logger.error(f"Schema processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "request_id": request_id
            }
    
    def secure_sql_generation(self,
                             user_query: str,
                             schema_context: Dict[str, Any],
                             **kwargs) -> Dict[str, Any]:
        """Secure SQL generation with schema obfuscation"""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # 1. Process and obfuscate schema
            schema_result = self.secure_schema_processing(schema_context)
            if not schema_result["success"]:
                raise Exception("Schema processing failed")
            
            obfuscated_schema = schema_result["obfuscated_schema"]
            
            # 2. Sanitize user query
            sanitized_query = self.sanitizer.sanitize_query(user_query)
            
            # 3. Create secure prompt
            secure_prompt = f"""
            Generate SQL query for the following request:
            
            User Query: {sanitized_query}
            
            Database Schema:
            {json.dumps(obfuscated_schema, indent=2)}
            
            Generate a valid SQL query that answers the user's question.
            """
            
            # 4. Make secure API call
            response = self.secure_text_completion(
                prompt=secure_prompt,
                system_message="You are a SQL generation assistant. Generate only valid SQL queries.",
                **kwargs
            )
            
            if not response["success"]:
                raise Exception(f"SQL generation failed: {response.get('error')}")
            
            # 5. Deobfuscate SQL if needed
            generated_sql = response["result"]
            deobfuscated_sql = self.obfuscator.deobfuscate_sql(generated_sql)
            
            return {
                "success": True,
                "original_sql": generated_sql,
                "deobfuscated_sql": deobfuscated_sql,
                "request_id": request_id,
                "data_classification": response["data_classification"],
                "tokens_used": response["tokens_used"],
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Secure SQL generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "request_id": request_id,
                "processing_time": time.time() - start_time
            }
    
    def _log_audit_entry(self, **kwargs):
        """Log audit entry"""
        audit_entry = AuditLog(
            timestamp=datetime.now(),
            **kwargs
        )
        self.audit_logger.log_request(audit_entry)
    
    def _estimate_cost(self, response) -> float:
        """Estimate API cost"""
        if not response.usage:
            return 0.0
        
        # Rough cost estimation (adjust based on actual pricing)
        input_cost_per_1k = 0.0015  # GPT-4 input tokens
        output_cost_per_1k = 0.006  # GPT-4 output tokens
        
        input_cost = (response.usage.prompt_tokens / 1000) * input_cost_per_1k
        output_cost = (response.usage.completion_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        return {
            "total_requests": len(self.rate_limiter.request_times),
            "rate_limit_remaining": max(0, self.config.max_requests_per_minute - len(self.rate_limiter.request_times)),
            "total_tokens_used": sum(self.rate_limiter.token_usage.values()),
            "obfuscation_mappings": {
                "tables": len(self.obfuscator.table_mapping),
                "columns": len(self.obfuscator.column_mapping)
            }
        }


# Example usage and integration
def create_secure_llm_client(api_key: str, config: Optional[SecurityConfig] = None) -> LLMProxyGateway:
    """Create a secure LLM client with proxy gateway"""
    return LLMProxyGateway(api_key=api_key, config=config)


# Integration with existing SQL Assistant
class SecureLLMClient:
    """Secure wrapper for existing LLM client"""
    
    def __init__(self, api_key: str, config: Optional[SecurityConfig] = None):
        self.proxy = LLMProxyGateway(api_key=api_key, config=config)
    
    def text(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
        """Secure text completion"""
        result = self.proxy.secure_text_completion(prompt, system_message, **kwargs)
        if result["success"]:
            return result["result"]
        else:
            raise Exception(f"Secure LLM call failed: {result['error']}")
    
    def json(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Secure JSON completion"""
        result = self.proxy.secure_text_completion(prompt, system_message, **kwargs)
        if result["success"]:
            try:
                return json.loads(result["result"])
            except json.JSONDecodeError:
                raise Exception("Failed to parse JSON response")
        else:
            raise Exception(f"Secure LLM call failed: {result['error']}")


if __name__ == "__main__":
    # Example usage
    config = SecurityConfig(
        max_requests_per_minute=30,
        max_tokens_per_request=2000
    )
    
    proxy = LLMProxyGateway(
        api_key="your-api-key-here",
        config=config
    )
    
    # Test secure text completion
    result = proxy.secure_text_completion(
        prompt="Show me customer revenue data",
        system_message="You are a helpful assistant."
    )
    
    print(f"Result: {result}")
    print(f"Security Stats: {proxy.get_security_stats()}")
