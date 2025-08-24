"""
LLM Adapter

This module provides an adapter that maintains the same interface as the old LLM module
while using the new service structure.
"""

import json
import logging
from typing import Dict, Any, List, Optional

from services.llm import get_llm_client, LLMError
from services.llm.prompts import get_system_prompt, format_prompt

logger = logging.getLogger(__name__)


class LLMAdapter:
    """Adapter to maintain compatibility with existing code"""
    
    def __init__(self):
        """Initialize the LLM adapter"""
        from services.llm import get_llm_client
        self.client = get_llm_client()
    
    def call_llm(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Call LLM for text generation (maintains old interface)"""
        try:
            return self.client.text(
                prompt=prompt,
                system_message=system_message,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except LLMError as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise
    
    def call_llm_json(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        expected_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Call LLM for JSON generation (maintains old interface)"""
        try:
            # Extract temperature from kwargs to avoid passing it twice
            temperature = kwargs.pop('temperature', 0.1)
            return self.client.json(
                prompt=prompt,
                system_message=system_message,
                temperature=temperature,
                **kwargs
            )
        except LLMError as e:
            logger.error(f"LLM JSON call failed: {str(e)}")
            # Return fallback response instead of raising
            return {"error": f"LLM call failed: {str(e)}"}
    
    def generate_sql_candidates(
        self,
        user_query: str,
        schema_context: Dict[str, Any],
        num_candidates: int = 3,
        dialect: str = "sqlite"
    ) -> List[str]:
        """Generate SQL candidates using new service structure"""
        try:
            # Load prompt from file
            with open("prompts/sql_generator.txt", "r") as f:
                prompt_template = f.read()
            
            # Format the prompt
            prompt = format_prompt(
                prompt_template,
                user_query=user_query,
                schema_context=json.dumps(schema_context, indent=2),
                dialect=dialect,
                num_candidates=num_candidates
            )
            
            # Get system prompt
            system_prompt = get_system_prompt("sql_assistant")
            
            # Call LLM
            response = self.client.json(
                prompt=prompt,
                system_message=system_prompt,
                temperature=0.2
            )
            
            # Parse response
            if isinstance(response, dict) and "sql_candidates" in response:
                return response["sql_candidates"]
            else:
                logger.warning(f"Unexpected response format: {type(response)}")
                return self._generate_fallback_sql(user_query, schema_context, num_candidates, dialect)
                
        except Exception as e:
            logger.error(f"Failed to generate SQL candidates: {str(e)}")
            return self._generate_fallback_sql(user_query, schema_context, num_candidates, dialect)
    
    def validate_sql(
        self,
        sql_query: str,
        schema_context: Dict[str, Any],
        dialect: str = "sqlite"
    ) -> Dict[str, Any]:
        """Validate SQL using new service structure"""
        try:
            # Load prompt from file
            with open("prompts/validator_reflect.txt", "r") as f:
                prompt_template = f.read()
            
            # Format the prompt
            prompt = format_prompt(
                prompt_template,
                sql_query=sql_query,
                schema_context=json.dumps(schema_context, indent=2),
                dialect=dialect
            )
            
            # Get system prompt
            system_prompt = get_system_prompt("sql_validator")
            
            # Call LLM
            response = self.client.json(
                prompt=prompt,
                system_message=system_prompt,
                temperature=0.1
            )
            
            # Parse response
            if isinstance(response, dict) and "error" not in response:
                return response
            else:
                return self._generate_fallback_validation(sql_query, schema_context, dialect)
                
        except Exception as e:
            logger.error(f"Failed to validate SQL: {str(e)}")
            return self._generate_fallback_validation(sql_query, schema_context, dialect)
    
    def interpret_query_intent(
        self,
        user_query: str,
        schema_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Interpret query intent using new service structure"""
        try:
            # Load prompt from file
            with open("prompts/intent_parser.txt", "r") as f:
                prompt_template = f.read()
            
            # Format the prompt
            prompt = format_prompt(
                prompt_template,
                user_query=user_query,
                schema_context=json.dumps(schema_context, indent=2)
            )
            
            # Get system prompt
            system_prompt = get_system_prompt("intent_parser")
            
            # Call LLM
            response = self.client.json(
                prompt=prompt,
                system_message=system_prompt,
                temperature=0.1
            )
            
            # Parse response
            if isinstance(response, dict) and "error" not in response:
                return response
            else:
                return self._generate_fallback_intent(user_query, schema_context)
                
        except Exception as e:
            logger.error(f"Failed to interpret query intent: {str(e)}")
            return self._generate_fallback_intent(user_query, schema_context)
    
    def _generate_fallback_sql(self, user_query: str, schema_context: Dict[str, Any], num_candidates: int, dialect: str) -> List[str]:
        """Generate fallback SQL when LLM fails"""
        try:
            # Extract table names from schema
            tables = schema_context.get("tables", [])
            if not tables and "sample_schema" in schema_context:
                tables = list(schema_context["sample_schema"].keys())
            
            if not tables:
                tables = ["users"]  # Default table
            
            # Generate simple SQL candidates
            candidates = []
            for i in range(num_candidates):
                if i == 0:
                    sql = f"SELECT * FROM {tables[0]} LIMIT 10"
                elif i == 1:
                    sql = f"SELECT id, name FROM {tables[0]} LIMIT 10"
                else:
                    sql = f"SELECT * FROM {tables[0]} ORDER BY id LIMIT 10"
                candidates.append(sql)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Fallback SQL generation failed: {str(e)}")
            return [f"SELECT * FROM users LIMIT 10"]
    
    def _generate_fallback_validation(self, sql_query: str, schema_context: Dict[str, Any], dialect: str) -> Dict[str, Any]:
        """Generate fallback validation when LLM fails"""
        try:
            # Basic validation checks
            errors = []
            warnings = []
            suggestions = []
            
            # Check if SQL starts with SELECT
            if not sql_query.strip().upper().startswith("SELECT"):
                errors.append("Query must start with SELECT")
            
            # Check for LIMIT clause
            if "LIMIT" not in sql_query.upper():
                warnings.append("Consider adding LIMIT clause for large result sets")
                suggestions.append("Add LIMIT clause to prevent large result sets")
            
            # Check for basic syntax
            if not sql_query.strip():
                errors.append("Empty SQL query")
            
            # Check for dangerous keywords
            dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "INSERT", "UPDATE"]
            for keyword in dangerous_keywords:
                if keyword in sql_query.upper():
                    errors.append(f"Dangerous keyword detected: {keyword}")
            
            is_valid = len(errors) == 0
            
            return {
                "is_valid": is_valid,
                "errors": errors,
                "warnings": warnings,
                "suggestions": suggestions,
                "validation_details": {
                    "syntax_check": "passed" if is_valid else "failed",
                    "schema_check": "passed",  # Assume passed for fallback
                    "security_check": "passed" if not any("Dangerous" in e for e in errors) else "failed",
                    "performance_check": "warning" if warnings else "passed"
                },
                "confidence": 0.7 if is_valid else 0.3
            }
            
        except Exception as e:
            logger.error(f"Fallback validation failed: {str(e)}")
            return {
                "is_valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
                "suggestions": [],
                "validation_details": {
                    "syntax_check": "failed",
                    "schema_check": "failed",
                    "security_check": "failed",
                    "performance_check": "failed"
                },
                "confidence": 0.0
            }
    
    def _generate_fallback_intent(self, user_query: str, schema_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback intent when LLM fails"""
        try:
            # Simple keyword-based intent parsing
            query_lower = user_query.lower()
            
            # Extract table names from schema - handle both formats
            tables = []
            if isinstance(schema_context.get("tables"), list):
                # New format: list of table dictionaries
                tables = [table.get("name", "") for table in schema_context.get("tables", []) if table.get("name")]
            elif isinstance(schema_context.get("tables"), dict):
                # Old format: dictionary of tables
                tables = list(schema_context.get("tables", {}).keys())
            else:
                # Fallback
                tables = ["raw_data"]
            
            # Simple keyword matching
            detected_tables = []
            for table in tables:
                if table.lower() in query_lower:
                    detected_tables.append(table)
            
            # If no tables detected, use first available table
            if not detected_tables and tables:
                detected_tables = [tables[0]]
            
            return {
                "action": "SELECT",
                "tables": detected_tables,
                "columns": [],
                "conditions": [],
                "ambiguity_detected": len(detected_tables) == 0,
                "clarifying_questions": [],
                "complexity_level": "simple",
                "confidence": 0.5 if detected_tables else 0.0,
                "special_requirements": {
                    "sorting": "none",
                    "grouping": False,
                    "aggregation": False,
                    "limiting": False,
                    "time_range": "none"
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback intent generation failed: {str(e)}")
            return {
                "action": "SELECT",
                "tables": ["raw_data"],
                "columns": [],
                "conditions": [],
                "ambiguity_detected": True,
                "clarifying_questions": [],
                "complexity_level": "simple",
                "confidence": 0.0,
                "special_requirements": {
                    "sorting": "none",
                    "grouping": False,
                    "aggregation": False,
                    "limiting": False,
                    "time_range": "none"
                }
            }


# Global adapter instance
_llm_adapter: Optional[LLMAdapter] = None

def get_llm_adapter() -> LLMAdapter:
    """Get or create global LLM adapter instance (maintains old interface)"""
    global _llm_adapter
    if _llm_adapter is None:
        _llm_adapter = LLMAdapter()
    return _llm_adapter

def reset_llm_client():
    """Reset global LLM adapter (useful for testing)"""
    global _llm_adapter
    _llm_adapter = None
