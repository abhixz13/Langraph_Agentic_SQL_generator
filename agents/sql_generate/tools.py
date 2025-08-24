"""
SQL Generation Tools

Thin wrappers around SQL generation functionality for the sql_generate subgraph.
"""

import json
import logging
from typing import Dict, Any, List, Optional

from core.llm_adapter import get_llm_adapter

logger = logging.getLogger(__name__)


def build_plan_from_intent(intent_json: Dict[str, Any], schema_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a plan DSL from structured intent.
    
    Args:
        intent_json: Structured intent from interpret_plan
        schema_context: Database schema context
        
    Returns:
        Plan DSL dictionary
    """
    try:
        # Extract key information from intent
        action = intent_json.get("action", "SELECT")
        tables = intent_json.get("tables", [])
        columns = intent_json.get("columns", [])
        conditions = intent_json.get("conditions", [])
        complexity_level = intent_json.get("complexity_level", "simple")
        
        # Build plan DSL
        plan_dsl = {
            "action": action,
            "tables": tables,
            "columns": columns,
            "conditions": conditions,
            "complexity_level": complexity_level,
            "joins": [],
            "group_by": None,
            "order_by": None,
            "limit": None,
            "aggregation": None,
            "window_functions": None
        }
        
        # Add aggregation if present
        if "aggregation_function" in intent_json:
            plan_dsl["aggregation"] = {
                "function": intent_json["aggregation_function"],
                "column": intent_json.get("aggregation_column")
            }
        
        # Add GROUP BY if present
        if "group_by" in intent_json and intent_json["group_by"]:
            plan_dsl["group_by"] = intent_json["group_by"]
        
        # Add ordering if present
        if "order_by" in intent_json and intent_json["order_by"]:
            plan_dsl["order_by"] = {
                "column": intent_json["order_by"],
                "direction": "DESC" if intent_json.get("desc", False) else "ASC"
            }
        
        # Add limit if present
        if "limit" in intent_json and intent_json["limit"]:
            plan_dsl["limit"] = intent_json["limit"]
        
        # Detect window function patterns (e.g., "TOP N per group")
        if _detect_window_function_pattern(intent_json):
            plan_dsl["window_functions"] = _build_window_function_config(intent_json)
        
        logger.debug(f"Built plan DSL: {plan_dsl}")
        return plan_dsl
        
    except Exception as e:
        logger.error(f"Failed to build plan from intent: {str(e)}")
        raise

def _detect_window_function_pattern(intent_json: Dict[str, Any]) -> bool:
    """
    Detect if the intent requires window functions (e.g., TOP N per group).
    
    Args:
        intent_json: Structured intent
        
    Returns:
        True if window functions are needed
    """
    # Check for "TOP N per group" patterns
    has_top_aggregation = (
        intent_json.get("aggregation_function") == "TOP" or
        "top" in intent_json.get("aggregation_function", "").lower()
    )
    has_group_by = bool(intent_json.get("group_by"))
    has_limit = bool(intent_json.get("limit"))
    
    # Window function needed if: TOP aggregation + GROUP BY + LIMIT
    return has_top_aggregation and has_group_by and has_limit

def _build_window_function_config(intent_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build window function configuration for TOP N per group patterns.
    
    Args:
        intent_json: Structured intent
        
    Returns:
        Window function configuration
    """
    return {
        "type": "row_number",
        "partition_by": intent_json.get("group_by", []),
        "order_by": {
            "column": intent_json.get("aggregation_column") or intent_json.get("order_by"),
            "direction": "DESC" if intent_json.get("desc", False) else "ASC"
        },
        "limit": intent_json.get("limit")
    }


def validate_fix_plan(plan_dsl: Dict[str, Any], schema_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fix the plan DSL against schema context.
    
    Args:
        plan_dsl: Plan DSL to validate
        schema_context: Database schema context
        
    Returns:
        Fixed plan DSL
    """
    try:
        # Get available tables and columns from schema
        available_tables = list(schema_context.get("tables", {}).keys())
        available_columns = {}
        
        for table_name, table_info in schema_context.get("tables", {}).items():
            if isinstance(table_info.get("columns"), dict):
                available_columns[table_name] = list(table_info["columns"].keys())
            elif isinstance(table_info.get("columns"), list):
                available_columns[table_name] = [col.get("name", "") for col in table_info["columns"]]
        
        # Validate tables
        plan_tables = plan_dsl.get("tables", [])
        valid_tables = [table for table in plan_tables if table in available_tables]
        
        if not valid_tables and plan_tables:
            # If no valid tables found, use the first available table
            if available_tables:
                valid_tables = [available_tables[0]]
                logger.warning(f"No valid tables found in plan, using first available: {valid_tables[0]}")
        
        # Validate columns
        valid_columns = []
        for col in plan_dsl.get("columns", []):
            for table in valid_tables:
                if col in available_columns.get(table, []):
                    valid_columns.append(col)
                    break
        
        # Build fixed plan
        fixed_plan = plan_dsl.copy()
        fixed_plan["tables"] = valid_tables
        fixed_plan["columns"] = valid_columns
        
        # Validate conditions - preserve dictionary conditions
        valid_conditions = []
        for condition in plan_dsl.get("conditions", []):
            if isinstance(condition, dict):
                # Validate dictionary conditions
                column = condition.get("column")
                operator = condition.get("operator")
                value = condition.get("value")
                
                # Check if column exists in any valid table
                column_valid = False
                for table in valid_tables:
                    if column in available_columns.get(table, []):
                        column_valid = True
                        break
                
                if column_valid and operator and value is not None:
                    valid_conditions.append(condition)
                    logger.debug(f"Validated condition: {condition}")
                else:
                    logger.warning(f"Invalid condition removed: {condition}")
            elif isinstance(condition, str):
                # Preserve string conditions
                valid_conditions.append(condition)
        
        fixed_plan["conditions"] = valid_conditions
        
        # Validate GROUP BY - preserve if columns exist
        group_by = plan_dsl.get("group_by")
        if group_by:
            if isinstance(group_by, list):
                valid_group_by = []
                for col in group_by:
                    for table in valid_tables:
                        if col in available_columns.get(table, []):
                            valid_group_by.append(col)
                            break
                fixed_plan["group_by"] = valid_group_by if valid_group_by else None
            else:
                # Single column case
                for table in valid_tables:
                    if group_by in available_columns.get(table, []):
                        fixed_plan["group_by"] = group_by
                        break
                else:
                    fixed_plan["group_by"] = None
        
        # Validate window functions
        window_functions = plan_dsl.get("window_functions")
        if window_functions:
            # Validate partition_by columns exist
            partition_by = window_functions.get("partition_by", [])
            if isinstance(partition_by, list):
                valid_partition_by = []
                for col in partition_by:
                    for table in valid_tables:
                        if col in available_columns.get(table, []):
                            valid_partition_by.append(col)
                            break
                window_functions["partition_by"] = valid_partition_by
                
                # Only keep window functions if partition columns are valid
                if not valid_partition_by:
                    fixed_plan["window_functions"] = None
                else:
                    fixed_plan["window_functions"] = window_functions
        
        logger.debug(f"Validated and fixed plan: {fixed_plan}")
        return fixed_plan
        
    except Exception as e:
        logger.error(f"Failed to validate/fix plan: {str(e)}")
        return plan_dsl  # Return original plan if validation fails


def get_fewshots(plan_dsl: Dict[str, Any], schema_context: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Get few-shot examples based on plan complexity using LLM-based generator.
    
    Args:
        plan_dsl: Plan DSL
        schema_context: Database schema context
        top_k: Number of examples to return
        
    Returns:
        List of few-shot examples
    """
    try:
        complexity_level = plan_dsl.get("complexity_level", "simple")
        
        # Only use few-shots for moderate/complex queries
        if complexity_level in ["simple"]:
            logger.debug("Skipping few-shots for simple query")
            return []
        
        # Use the new LLM-based few-shot generator
        from .few_shot_generator import FewShotExampleGenerator
        
        generator = FewShotExampleGenerator()
        examples_data = generator.get_examples_for_schema(schema_context, plan_dsl)
        
        # Extract and format examples for the SQL generator
        examples = []
        for example in examples_data.get("examples", [])[:top_k]:
            examples.append({
                "query": example.get("query", ""),
                "sql": example.get("sql", ""),
                "explanation": example.get("explanation", "")
            })
        
        logger.debug(f"Generated {len(examples)} few-shot examples using LLM")
        return examples
        
    except Exception as e:
        logger.error(f"Failed to get few-shots: {str(e)}")
        # Fallback to basic examples
        return _get_fallback_fewshots(plan_dsl, top_k)

def _get_fallback_fewshots(plan_dsl: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    """Fallback few-shot examples if LLM generation fails."""
    examples = []
    
    # Example 1: Basic SELECT
    if plan_dsl.get("action") == "SELECT":
        examples.append({
            "query": "Show me customer names",
            "sql": "SELECT CUSTOMER_NAME FROM raw_data LIMIT 10"
        })
    
    # Example 2: Aggregation
    if plan_dsl.get("aggregation"):
        examples.append({
            "query": "What is the total revenue?",
            "sql": "SELECT SUM(ACTUAL_BOOKINGS) as total_revenue FROM raw_data"
        })
    
    # Example 3: Filtering
    if plan_dsl.get("conditions"):
        examples.append({
            "query": "Show SaaS customers",
            "sql": "SELECT CUSTOMER_NAME FROM raw_data WHERE IntersightConsumption = 'SaaS'"
        })
    
    return examples[:top_k]


def generate_sql_candidates(
    plan_dsl: Dict[str, Any], 
    schema_context: Dict[str, Any], 
    dialect: str, 
    k: int, 
    fewshots: Optional[List[Dict[str, Any]]] = None
) -> List[str]:
    """
    Generate SQL candidates from plan DSL.
    
    Args:
        plan_dsl: Plan DSL
        schema_context: Database schema context
        dialect: SQL dialect
        k: Number of candidates to generate
        fewshots: Optional few-shot examples
        
    Returns:
        List of SQL candidate strings
    """
    try:
        llm_client = get_llm_adapter()
        
        # Build prompt from plan DSL
        prompt = _build_sql_generation_prompt(plan_dsl, schema_context, dialect, k, fewshots)
        
        # Get system prompt from the standard location
        from core.llm_adapter import get_system_prompt
        system_prompt = get_system_prompt("sql_assistant")
        
        # Call LLM
        response = llm_client.call_llm_json(
            prompt=prompt,
            system_message=system_prompt,
            temperature=0.2
        )
        
        # Parse response
        if isinstance(response, dict) and "sql_candidates" in response:
            candidates = response["sql_candidates"]
            if isinstance(candidates, list):
                return candidates[:k]
        
        # Fallback: generate simple SQL from plan
        return _generate_fallback_sql(plan_dsl, schema_context, dialect, k)
        
    except Exception as e:
        logger.error(f"Failed to generate SQL candidates: {str(e)}")
        return _generate_fallback_sql(plan_dsl, schema_context, dialect, k)


def _build_sql_generation_prompt(
    plan_dsl: Dict[str, Any], 
    schema_context: Dict[str, Any], 
    dialect: str, 
    k: int, 
    fewshots: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Build prompt for SQL generation using the existing template but adapted for plan DSL."""
    
    try:
        # Load the existing prompt template
        with open("prompts/sql_generator.txt", "r") as f:
            prompt_template = f.read()
        
        # Convert plan DSL to a structured query description
        plan_description = _plan_dsl_to_query_description(plan_dsl)
        
        # Add few-shot examples if available
        if fewshots:
            fewshot_section = f"\n\nFew-shot Examples:\n{json.dumps(fewshots, indent=2)}"
            # Insert few-shot examples before the final instruction
            prompt_template = prompt_template.replace(
                "Generate the SQL candidates now:",
                f"{fewshot_section}\n\nGenerate the SQL candidates now:"
            )
        
        # Manual string replacement to avoid conflicts with JSON braces
        prompt = prompt_template.replace("{user_query}", plan_description)
        prompt = prompt.replace("{schema_context}", json.dumps(schema_context, indent=2))
        prompt = prompt.replace("{dialect}", dialect)
        prompt = prompt.replace("{num_candidates}", str(k))
        
        return prompt
        
    except Exception as e:
        logger.error(f"Failed to load prompt template, using fallback: {str(e)}")
        # Fallback to the original implementation
        fewshot_text = f"Few-shot examples:\n{json.dumps(fewshots, indent=2)}\n" if fewshots else ""
        
        prompt = f"""Generate {k} SQL query candidates based on the following plan:

Plan DSL:
{json.dumps(plan_dsl, indent=2)}

Schema Context:
{json.dumps(schema_context, indent=2)}

Dialect: {dialect}

Requirements:
1. Generate exactly {k} different SQL candidates
2. Use only tables and columns from the schema
3. Follow {dialect} syntax
4. Make queries safe (no DDL/DML)
5. Consider different approaches for each candidate

{fewshot_text}

Please respond with a JSON object:
{{
    "sql_candidates": [
        "SELECT ... FROM ... WHERE ...",
        "SELECT ... FROM ... JOIN ... WHERE ...",
        ...
    ]
}}

Generate the SQL candidates now:"""
        
        return prompt


def _plan_dsl_to_query_description(plan_dsl: Dict[str, Any]) -> str:
    """
    Convert plan DSL to a comprehensive natural language query description for the prompt template.
    
    This function creates a detailed, context-rich description that helps the LLM understand:
    - The exact query intent and business logic
    - Required SQL patterns and techniques
    - Performance considerations
    - Expected output format
    """
    
    action = plan_dsl.get("action", "SELECT")
    tables = plan_dsl.get("tables", [])
    columns = plan_dsl.get("columns", [])
    conditions = plan_dsl.get("conditions", [])
    aggregation = plan_dsl.get("aggregation")
    group_by = plan_dsl.get("group_by")
    order_by = plan_dsl.get("order_by")
    limit = plan_dsl.get("limit")
    window_functions = plan_dsl.get("window_functions")
    complexity_level = plan_dsl.get("complexity_level", "simple")
    
    # Build comprehensive description
    description_sections = []
    
    # 1. Query Intent and Business Context
    intent_section = _build_query_intent_section(plan_dsl)
    if intent_section:
        description_sections.append(intent_section)
    
    # 2. Data Source and Scope
    data_section = _build_data_source_section(tables, columns)
    if data_section:
        description_sections.append(data_section)
    
    # 3. Filtering and Conditions
    filter_section = _build_filter_section(conditions)
    if filter_section:
        description_sections.append(filter_section)
    
    # 4. Aggregation and Grouping
    agg_section = _build_aggregation_section(aggregation, group_by)
    if agg_section:
        description_sections.append(agg_section)
    
    # 5. Window Functions and Ranking
    window_section = _build_window_function_section(window_functions)
    if window_section:
        description_sections.append(window_section)
    
    # 6. Sorting and Limiting
    sort_section = _build_sorting_section(order_by, limit, window_functions)
    if sort_section:
        description_sections.append(sort_section)
    
    # 7. Performance and Technical Requirements
    tech_section = _build_technical_requirements_section(complexity_level, window_functions)
    if tech_section:
        description_sections.append(tech_section)
    
    # Combine all sections
    if description_sections:
        return "\n\n".join(description_sections)
    else:
        return f"Generate a {action} query"

def _build_query_intent_section(plan_dsl: Dict[str, Any]) -> str:
    """Build the query intent and business context section."""
    action = plan_dsl.get("action", "SELECT")
    window_functions = plan_dsl.get("window_functions")
    aggregation = plan_dsl.get("aggregation")
    
    intent_parts = []
    
    # Determine the main query type
    if window_functions:
        wf_type = window_functions.get("type", "")
        wf_limit = window_functions.get("limit")
        partition_by = window_functions.get("partition_by", [])
        
        if wf_type == "row_number":
            partition_list = ", ".join(partition_by) if isinstance(partition_by, list) else partition_by
            intent_parts.append(f"Find the top {wf_limit} records for each {partition_list}")
    elif aggregation:
        func = aggregation.get("function", "")
        if func.upper() == "TOP":
            intent_parts.append("Find the top records based on specified criteria")
        else:
            intent_parts.append(f"Calculate {func.lower()} aggregation")
    else:
        intent_parts.append(f"Retrieve data using {action}")
    
    return "**Query Intent:** " + " ".join(intent_parts)

def _build_data_source_section(tables: List[str], columns: List[str]) -> str:
    """Build the data source and scope section."""
    sections = []
    
    # Data source
    if tables:
        table_list = ", ".join(tables)
        sections.append(f"**Data Source:** {table_list}")
    
    # Columns to retrieve
    if columns and columns != ["*"]:
        column_list = ", ".join(columns)
        sections.append(f"**Required Columns:** {column_list}")
    elif not columns:
        sections.append("**Required Columns:** All available columns")
    
    return "\n".join(sections)

def _build_filter_section(conditions: List[Any]) -> str:
    """Build the filtering and conditions section."""
    if not conditions:
        return ""
    
    filter_descriptions = []
    for condition in conditions:
        if isinstance(condition, dict):
            col = condition.get("column", "")
            op = condition.get("operator", "")
            val = condition.get("value", "")
            
            # Create business-friendly descriptions
            if op == "=":
                filter_descriptions.append(f"only include records where {col} equals '{val}'")
            elif op == "!=":
                filter_descriptions.append(f"exclude records where {col} equals '{val}'")
            elif op == ">":
                filter_descriptions.append(f"only include records where {col} is greater than '{val}'")
            elif op == "<":
                filter_descriptions.append(f"only include records where {col} is less than '{val}'")
            elif op == ">=":
                filter_descriptions.append(f"only include records where {col} is greater than or equal to '{val}'")
            elif op == "<=":
                filter_descriptions.append(f"only include records where {col} is less than or equal to '{val}'")
            elif op == "LIKE":
                filter_descriptions.append(f"only include records where {col} matches pattern '{val}'")
            elif op == "IN":
                filter_descriptions.append(f"only include records where {col} is in the list {val}")
            else:
                filter_descriptions.append(f"filter by {col} {op} '{val}'")
        else:
            filter_descriptions.append(str(condition))
    
    return "**Filtering:** " + " and ".join(filter_descriptions)

def _build_aggregation_section(aggregation: Dict[str, Any], group_by: Any) -> str:
    """Build the aggregation and grouping section."""
    sections = []
    
    # Aggregation
    if aggregation:
        func = aggregation.get("function", "")
        col = aggregation.get("column", "")
        
        if func.upper() == "TOP":
            sections.append(f"**Aggregation:** Find top records based on {col}")
        elif func.upper() in ["SUM", "COUNT", "AVG", "MAX", "MIN"]:
            sections.append(f"**Aggregation:** Calculate {func.lower()}({col})")
        else:
            sections.append(f"**Aggregation:** Apply {func} function to {col}")
    
    # Grouping
    if group_by:
        if isinstance(group_by, list):
            group_list = ", ".join(group_by)
            sections.append(f"**Grouping:** Group results by {group_list}")
        else:
            sections.append(f"**Grouping:** Group results by {group_by}")
    
    return "\n".join(sections) if sections else ""

def _build_window_function_section(window_functions: Dict[str, Any]) -> str:
    """Build the window functions and ranking section."""
    if not window_functions:
        return ""
    
    wf_type = window_functions.get("type", "")
    partition_by = window_functions.get("partition_by", [])
    wf_order_by = window_functions.get("order_by", {})
    wf_limit = window_functions.get("limit")
    
    if wf_type == "row_number" and partition_by and wf_limit:
        partition_list = ", ".join(partition_by) if isinstance(partition_by, list) else partition_by
        order_col = wf_order_by.get("column", "")
        order_dir = wf_order_by.get("direction", "DESC")
        
        sections = [
            f"**Window Function Required:** Use ROW_NUMBER() window function",
            f"**Partitioning:** Partition by {partition_list}",
            f"**Ranking:** Order within each partition by {order_col} {order_dir}",
            f"**Result:** Return only the top {wf_limit} records from each partition"
        ]
        
        return "\n".join(sections)
    
    return ""

def _build_sorting_section(order_by: Dict[str, Any], limit: int, window_functions: Dict[str, Any]) -> str:
    """Build the sorting and limiting section."""
    sections = []
    
    # Sorting (only if not handled by window function)
    if order_by and not window_functions:
        col = order_by.get("column", "")
        direction = order_by.get("direction", "ASC")
        sections.append(f"**Sorting:** Order results by {col} {direction}")
    
    # Limiting (only if not handled by window function)
    if limit and not window_functions:
        sections.append(f"**Limiting:** Return only the first {limit} records")
    
    return "\n".join(sections) if sections else ""

def _build_technical_requirements_section(complexity_level: str, window_functions: Dict[str, Any]) -> str:
    """Build the technical requirements and performance considerations section."""
    sections = []
    
    # Complexity-based requirements
    if complexity_level == "hard":
        sections.append("**Complexity:** This is a complex query requiring advanced SQL techniques")
    elif complexity_level == "moderate":
        sections.append("**Complexity:** This is a moderately complex query")
    
    # Window function requirements
    if window_functions:
        sections.extend([
            "**SQL Technique:** Use window functions (ROW_NUMBER, RANK, or DENSE_RANK)",
            "**Performance:** Consider indexing on partition and order columns for optimal performance",
            "**Alternative:** If window functions are not supported, use subqueries or self-joins"
        ])
    
    # General requirements
    sections.extend([
        "**Safety:** Ensure the query is read-only (no DDL/DML operations)",
        "**Efficiency:** Use appropriate indexes and avoid unnecessary table scans",
        "**Clarity:** Make the SQL readable and well-formatted"
    ])
    
    return "\n".join(sections)


def _generate_fallback_sql(
    plan_dsl: Dict[str, Any], 
    schema_context: Dict[str, Any], 
    dialect: str, 
    k: int
) -> List[str]:
    """Generate fallback SQL when LLM fails."""
    
    try:
        tables = plan_dsl.get("tables", [])
        columns = plan_dsl.get("columns", [])
        conditions = plan_dsl.get("conditions", [])
        
        if not tables:
            # Use first available table
            available_tables = list(schema_context.get("tables", {}).keys())
            if available_tables:
                tables = [available_tables[0]]
        
        if not columns:
            # Use common columns
            columns = ["*"]
        
        # Build basic SQL
        sql = f"SELECT {', '.join(columns)} FROM {', '.join(tables)}"
        
        if conditions:
            sql += f" WHERE {' AND '.join(conditions)}"
        
        # Add limit for safety
        sql += " LIMIT 100"
        
        return [sql] * min(k, 1)  # Return same SQL k times or just once
        
    except Exception as e:
        logger.error(f"Fallback SQL generation failed: {str(e)}")
        return []
