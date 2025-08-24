"""
Few-Shot Example Generator for SQL queries using LLM

This module generates complex SQL examples with thinking processes that can be
injected as few-shot examples in the SQL generator agent's prompts.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from core.llm_adapter import get_llm_adapter
from core.config import load_settings

logger = logging.getLogger(__name__)

class FewShotExampleGenerator:
    """
    Generates complex SQL examples with thinking processes using LLM.
    
    Features:
    - Schema-aware example generation
    - Complex query patterns (JOINs, subqueries, window functions)
    - Step-by-step thinking process documentation
    - JSON output for easy integration
    """
    
    def __init__(self, examples_dir: str = "data/few_shot_examples"):
        """
        Initialize the few-shot example generator.
        
        Args:
            examples_dir: Directory to store generated examples
        """
        self.llm_client = get_llm_adapter()
        self.examples_dir = Path(examples_dir)
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("FewShotExampleGenerator initialized")
    
    def generate_examples(self, schema_context: Dict[str, Any], plan_dsl: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate few-shot examples based on database schema and plan DSL.
        
        Args:
            schema_context: Database schema context
            plan_dsl: Optional current plan DSL for context-aware examples
            
        Returns:
            Dict containing generated examples with thinking processes
        """
        try:
            # Create context from plan DSL
            plan_context = self._create_plan_context(plan_dsl) if plan_dsl else ""
            
            # Load the prompt template
            with open("prompts/few_shot_generator.txt", "r") as f:
                prompt_template = f.read()
            
            # Build the user prompt
            user_prompt = self._build_generation_prompt(
                plan_dsl=plan_dsl or {},
                schema_context=schema_context,
                dialect="sqlite",  # Default dialect
                num_examples=4
            )
            
            # Get system prompt
            from core.llm_adapter import get_system_prompt
            system_prompt = get_system_prompt("sql_assistant")
            
            # Call LLM
            response = self.llm_client.call_llm_json(
                prompt=user_prompt,
                system_message=system_prompt
            )
            
            # Parse and validate response
            if isinstance(response, dict) and "examples" in response:
                examples_data = response
            else:
                # Fallback parsing
                examples_data = self._parse_fallback_response(response)
            
            # Add metadata
            schema_hash = self._generate_schema_hash(schema_context)
            examples_data["schema_hash"] = schema_hash
            examples_data["generated_at"] = self._get_timestamp()
            
            # Save to file
            filename = f"examples_{schema_hash[:8]}.json"
            filepath = self.examples_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(examples_data, f, indent=2)
            
            logger.info(f"Generated {len(examples_data.get('examples', []))} examples, saved to {filepath}")
            return examples_data
            
        except Exception as e:
            logger.error(f"Error generating examples: {e}")
            return self._fallback_examples()
    
    def get_examples_for_schema(self, schema_context: Dict[str, Any], plan_dsl: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get examples for a schema, generating if not cached.
        
        Args:
            schema_context: Database schema context
            plan_dsl: Optional current plan DSL for context-aware examples
            
        Returns:
            Dict containing examples or empty dict if failed
        """
        # Check if ANY examples already exist in the directory
        existing_files = list(self.examples_dir.glob("examples_*.json"))
        
        if existing_files:
            # Use the first available cached example file
            existing_file = existing_files[0]
            try:
                with open(existing_file, 'r') as f:
                    examples_data = json.load(f)
                logger.info(f"✅ Using cached examples from {existing_file.name}")
                return examples_data
            except Exception as e:
                logger.warning(f"Error loading cached examples from {existing_file}: {e}")
        
        # Only generate new examples if no cached examples exist
        logger.info("📝 No cached examples found, generating new few-shot examples...")
        return self.generate_examples(schema_context, plan_dsl)
    
    def format_examples_for_prompt(self, examples_data: Dict[str, Any]) -> str:
        """
        Format examples for injection into SQL generator prompt.
        
        Args:
            examples_data: Generated examples data
            
        Returns:
            Formatted string for prompt injection
        """
        if not examples_data or "examples" not in examples_data:
            return self._get_default_examples()
        
        formatted_examples = []
        
        for example in examples_data.get("examples", []):
            thinking = "\n".join([f"  {step}" for step in example.get("thinking_process", [])])
            breakdown = "\n".join([f"  {step}" for step in example.get("query_breakdown", [])])
            
            formatted = f"""
Example: {example.get('query', 'Unknown question')}

Thinking Process:
{thinking}

Query Breakdown:
{breakdown}

SQL:
{example.get('sql', 'SELECT * FROM table;')}

Explanation: {example.get('explanation', 'No explanation provided')}
"""
            formatted_examples.append(formatted)
        
        return "\n" + "="*50 + "\n".join(formatted_examples) + "\n" + "="*50
    
    def _build_generation_prompt(
        self, 
        plan_dsl: Dict[str, Any], 
        schema_context: Dict[str, Any], 
        dialect: str, 
        num_examples: int
    ) -> str:
        """Build the user prompt for few-shot generation."""
        try:
            # Load the prompt template
            with open("prompts/few_shot_generator.txt", "r") as f:
                prompt_template = f.read()
            
            # Manual string replacement to avoid conflicts with JSON braces
            prompt = prompt_template.replace("{plan_dsl}", json.dumps(plan_dsl, indent=2))
            prompt = prompt.replace("{schema_context}", json.dumps(schema_context, indent=2))
            prompt = prompt.replace("{dialect}", dialect)
            prompt = prompt.replace("{num_examples}", str(num_examples))
            
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to load prompt template, using fallback: {str(e)}")
            # Fallback prompt
            return f"""
Generate {num_examples} complex SQL examples based on the following plan and schema:

Plan DSL:
{json.dumps(plan_dsl, indent=2)}

Database Schema:
{json.dumps(schema_context, indent=2)}

SQL Dialect: {dialect}

Generate {num_examples} examples that demonstrate:
1. Complex analytical queries (JOINs, subqueries, window functions)
2. Step-by-step thinking process
3. Query breakdown and planning
4. Schema relationship understanding

Return in this JSON format:
{{
    "examples": [
        {{
            "query": "natural language question",
            "thinking_process": ["Step 1: thought", "Step 2: thought"],
            "query_breakdown": ["1. step", "2. step"],
            "sql": "SELECT ... FROM ...",
            "explanation": "explanation of approach"
        }}
    ]
}}
"""
    
    def _create_plan_context(self, plan_dsl: Dict[str, Any]) -> str:
        """Create context from current plan DSL to guide example generation."""
        if not plan_dsl:
            return ""
        
        action = plan_dsl.get("action", "")
        entity = plan_dsl.get("entity", "")
        complexity = plan_dsl.get("complexity_level", "")
        
        context_parts = [
            f"CURRENT QUERY CONTEXT:",
            f"The user is asking for a '{action}' operation on '{entity}'.",
            f"Query complexity: {complexity}"
        ]
        
        # Add aggregation context
        if plan_dsl.get("aggregation"):
            context_parts.append(f"Query involves aggregation: {plan_dsl['aggregation']}")
        
        # Add conditions context
        if plan_dsl.get("conditions"):
            context_parts.append(f"Query has filtering conditions")
        
        context_parts.extend([
            "",
            "Generate examples that include similar query patterns and demonstrate:",
            f"- How to handle '{action}' operations effectively",
            f"- Best practices for querying '{entity}'",
            "- Related analytical patterns users might need",
            "- Progressive complexity from the current query type",
            ""
        ])
        
        return "\n".join(context_parts)
    
    def _generate_schema_hash(self, schema_context: Dict[str, Any]) -> str:
        """Generate hash for schema to use as cache key."""
        schema_str = json.dumps(schema_context, sort_keys=True)
        return hashlib.md5(schema_str.encode()).hexdigest()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.now().isoformat()
    
    def _parse_fallback_response(self, response: Any) -> Dict[str, Any]:
        """Parse response with fallback handling."""
        if isinstance(response, str):
            try:
                # Try to extract JSON from text
                start = response.find("{")
                end = response.rfind("}")
                if start != -1 and end != -1:
                    json_str = response[start:end+1]
                    return json.loads(json_str)
            except:
                pass
        
        # Return fallback if parsing fails
        return self._fallback_examples()
    
    def _fallback_examples(self) -> Dict[str, Any]:
        """Return fallback examples if generation fails."""
        return {
            "schema_hash": "fallback",
            "generated_at": self._get_timestamp(),
            "examples": [
                {
                    "query": "Show me customer names",
                    "thinking_process": [
                        "Step 1: Identify the customer name column",
                        "Step 2: Select from the main table",
                        "Step 3: Add limit for safety"
                    ],
                    "query_breakdown": [
                        "1. SELECT CUSTOMER_NAME",
                        "2. FROM raw_data table",
                        "3. LIMIT 10 for safety"
                    ],
                    "sql": "SELECT CUSTOMER_NAME FROM raw_data LIMIT 10",
                    "explanation": "Simple selection of customer names with safety limit"
                },
                {
                    "query": "What is the total revenue?",
                    "thinking_process": [
                        "Step 1: Identify revenue column (ACTUAL_BOOKINGS)",
                        "Step 2: Use SUM aggregation",
                        "Step 3. Select from main table"
                    ],
                    "query_breakdown": [
                        "1. SELECT SUM(ACTUAL_BOOKINGS)",
                        "2. FROM raw_data table",
                        "3. Alias the result for clarity"
                    ],
                    "sql": "SELECT SUM(ACTUAL_BOOKINGS) as total_revenue FROM raw_data",
                    "explanation": "Simple aggregation to get total revenue"
                }
            ]
        }
    
    def _get_default_examples(self) -> str:
        """Get default examples if none are available."""
        return """
Example: Show customer names
Thinking: Need to select customer names from the main table
SQL: SELECT CUSTOMER_NAME FROM raw_data LIMIT 10;

Example: Calculate total revenue
Thinking: Aggregate the revenue column using SUM function
SQL: SELECT SUM(ACTUAL_BOOKINGS) as total_revenue FROM raw_data;
"""
