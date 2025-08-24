"""
SQL Generation Agent Subgraph

This module implements the SQL generation agent that converts structured intent
into SQL statements using the intent from interpret_plan for efficiency.
"""

import logging
from typing import Dict, Any, List
from langgraph.graph import StateGraph, START, END

from core.state import AppState, create_error_response, create_success_response
from .tools import (
    build_plan_from_intent,
    validate_fix_plan,
    get_fewshots,
    generate_sql_candidates
)

logger = logging.getLogger(__name__)

# Policy constants
MAX_K = 5  # Maximum number of candidates

# Safety: Block dangerous DDL/DML operations
BLOCKED_KEYWORDS = [
    "DROP", "DELETE", "TRUNCATE", "ALTER", "GRANT", "REVOKE", 
    "CREATE", "INSERT", "UPDATE", "EXEC", "EXECUTE", "CALL"
]

# Smart LIMIT policy: Only require LIMIT for potentially large result sets
REQUIRE_LIMIT_FOR_LARGE_QUERIES = True  # Whether to require LIMIT for large queries
LARGE_QUERY_THRESHOLD = 1000  # Threshold for considering a query potentially large

# Allow certain keywords in specific contexts (e.g., CREATE in CREATE VIEW)
ALLOWED_CONTEXTS = {
    "CREATE": ["CREATE VIEW", "CREATE TEMP", "CREATE TEMPORARY"],
    "ALTER": ["ALTER VIEW"]
}


def sql_generate_node(state: AppState) -> Dict[str, Any]:
    """
    Generate SQL candidates using intent from interpret_plan
    
    This node takes the structured intent and schema context to generate
    multiple SQL query candidates using the LLM.
    
    Runtime Contract:
    Inputs: intent_json, schema_context, gen_k, dialect
    Outputs: sql_candidates, sql, gen_attempts
    """
    try:
        logger.info("Starting SQL generation from intent")
        
        # Preconditions & Guards
        intent_json = getattr(state, "intent_json", None)
        if not intent_json:
            return create_error_response(
                error_message="Cannot generate SQL: missing intent_json",
                sql_candidates=[],
                sql=None,
                gen_attempts=(getattr(state, "gen_attempts", 0) or 0) + 1,
                status="ERROR"
            )
        
        # Check for ambiguity
        if intent_json.get("ambiguity_detected", False):
            return create_error_response(
                error_message="Cannot generate SQL: ambiguous intent detected",
                sql_candidates=[],
                sql=None,
                gen_attempts=(getattr(state, "gen_attempts", 0) or 0) + 1,
                status="ERROR"
            )
        
        schema_context = getattr(state, "schema_context", None)
        if not schema_context or not schema_context.get("tables"):
            return create_error_response(
                error_message="Cannot generate SQL: schema_context empty",
                sql_candidates=[],
                sql=None,
                gen_attempts=(getattr(state, "gen_attempts", 0) or 0) + 1,
                status="ERROR"
            )
        
        # Normalize gen_k
        gen_k = getattr(state, "gen_k", 3) or 3
        if gen_k < 1:
            gen_k = 1
        elif gen_k > MAX_K:
            gen_k = MAX_K
            logger.info(f"Clamped gen_k to policy maximum: {MAX_K}")
        
        # Validate dialect
        dialect = getattr(state, "dialect", "sqlite") or "sqlite"
        valid_dialects = ["postgres", "mysql", "sqlite", "bigquery", "snowflake"]
        if dialect not in valid_dialects:
            dialect = "sqlite"  # Default to sqlite
            logger.warning(f"Unknown dialect, defaulting to sqlite: {dialect}")
        
        logger.info(f"Generating {gen_k} SQL candidates for dialect: {dialect}")
        
        # Tool-Calling (internal only)
        # 1. Use the rich plan from interpret_plan instead of rebuilding
        plan_dsl = getattr(state, "plan", {})
        if not plan_dsl:
            # Fallback: Build plan from intent if plan is missing
            plan_dsl = build_plan_from_intent(intent_json, schema_context)
            logger.debug(f"Built fallback plan DSL: {plan_dsl}")
        else:
            logger.debug(f"Using rich plan from interpret_plan: {plan_dsl}")
        
        # 2. Validate and fix plan
        fixed_plan = validate_fix_plan(plan_dsl, schema_context)
        logger.debug(f"Validated plan: {fixed_plan}")
        
        # 3. Get few-shot examples (conditional)
        fewshots = None
        complexity_level = intent_json.get("complexity_level", "simple")
        if complexity_level in ["moderate", "hard"]:
            fewshots = get_fewshots(fixed_plan, schema_context, top_k=3)
            logger.debug(f"Using {len(fewshots)} few-shot examples")
        
        # 4. Generate SQL candidates
        raw_candidates = generate_sql_candidates(
            plan_dsl=fixed_plan,
            schema_context=schema_context,
            dialect=dialect,
            k=gen_k,
            fewshots=fewshots
        )
        
        if not raw_candidates:
            return create_error_response(
                error_message="Cannot generate SQL: generator returned no candidates",
                sql_candidates=[],
                sql=None,
                gen_attempts=(getattr(state, "gen_attempts", 0) or 0) + 1,
                status="ERROR"
            )
        
        # 5. Filtering & Dedup
        filtered_candidates = _filter_and_dedup_candidates(raw_candidates, gen_k, fixed_plan)
        
        if not filtered_candidates:
            return create_error_response(
                error_message="Cannot generate SQL: all candidates violated safety/policy",
                sql_candidates=[],
                sql=None,
                gen_attempts=(getattr(state, "gen_attempts", 0) or 0) + 1,
                status="ERROR"
            )
        
        # 6. Selection (Primary SQL)
        primary_sql = _select_primary_sql(filtered_candidates, fixed_plan)
        
        if not primary_sql:
            return create_error_response(
                error_message="Cannot generate SQL: failed to select primary SQL",
                sql_candidates=filtered_candidates,
                sql=None,
                gen_attempts=(getattr(state, "gen_attempts", 0) or 0) + 1,
                status="ERROR"
            )
        
        # Return patch with only required outputs
        patch = create_success_response(
            sql_candidates=filtered_candidates,
            sql=primary_sql,
            gen_attempts=(getattr(state, "gen_attempts", 0) or 0) + 1,
            status="GENERATING"
        )
        
        logger.info(f"SQL generation completed. Generated {len(filtered_candidates)} candidates")
        logger.debug(f"Primary SQL: {primary_sql[:200]}...")
        
        return patch
        
    except Exception as e:
        logger.error(f"SQL generation failed: {str(e)}")
        return create_error_response(
            error_message=f"SQL generation error: {str(e)}",
            sql_candidates=[],
            sql=None,
            gen_attempts=(getattr(state, "gen_attempts", 0) or 0) + 1,
            status="ERROR"
        )


def _filter_and_dedup_candidates(candidates: List[str], max_k: int, plan_dsl: Dict[str, Any] = None) -> List[str]:
    """
    Intelligently filter and deduplicate SQL candidates.
    
    Args:
        candidates: Raw SQL candidates
        max_k: Maximum number of candidates to return
        plan_dsl: Plan DSL for context-aware filtering (optional)
        
    Returns:
        Filtered and deduplicated candidates
    """
    # Normalize and deduplicate
    normalized_candidates = []
    seen = set()
    
    for candidate in candidates:
        if not candidate or not isinstance(candidate, str):
            continue
            
        # Normalize whitespace and case
        normalized = " ".join(candidate.strip().split()).upper()
        
        if normalized in seen:
            continue
            
        # 1. Safety Check: Block dangerous operations
        if _is_dangerous_sql(normalized):
            logger.debug(f"Filtered dangerous candidate: {candidate[:100]}...")
            continue
            
        # 2. Context-Aware LIMIT Check: Only require LIMIT for potentially large queries
        if _requires_limit_check(normalized, plan_dsl) and "LIMIT" not in normalized:
            logger.debug(f"Filtered candidate without LIMIT (large query): {candidate[:100]}...")
            continue
            
        # 3. Basic SQL Validation: Check for minimal SQL structure
        if not _is_valid_sql_structure(normalized):
            logger.debug(f"Filtered invalid SQL structure: {candidate[:100]}...")
            continue
            
        # 4. Plan DSL Compliance: Check if query matches plan requirements
        if plan_dsl and not _matches_plan_requirements(normalized, plan_dsl):
            logger.debug(f"Filtered candidate not matching plan: {candidate[:100]}...")
            continue
            
        seen.add(normalized)
        normalized_candidates.append(candidate)
        
        if len(normalized_candidates) >= max_k:
            break
    
    return normalized_candidates

def _is_dangerous_sql(normalized_sql: str) -> bool:
    """
    Check if SQL contains dangerous operations.
    
    Args:
        normalized_sql: Normalized SQL string (uppercase)
        
    Returns:
        True if SQL is dangerous
    """
    # Check for blocked keywords
    for keyword in BLOCKED_KEYWORDS:
        if keyword in normalized_sql:
            # Check if it's in an allowed context
            if keyword in ALLOWED_CONTEXTS:
                allowed_contexts = ALLOWED_CONTEXTS[keyword]
                if not any(context in normalized_sql for context in allowed_contexts):
                    return True
            else:
                return True
    
    return False

def _requires_limit_check(normalized_sql: str, plan_dsl: Dict[str, Any] = None) -> bool:
    """
    Determine if a query requires LIMIT check based on context.
    
    Args:
        normalized_sql: Normalized SQL string (uppercase)
        plan_dsl: Plan DSL for context
        
    Returns:
        True if LIMIT should be required
    """
    if not REQUIRE_LIMIT_FOR_LARGE_QUERIES:
        return False
    
    # Don't require LIMIT for aggregation queries (they return limited results)
    if any(func in normalized_sql for func in ["SUM(", "COUNT(", "AVG(", "MAX(", "MIN("]):
        return False
    
    # Don't require LIMIT for window function queries (they're already limited)
    if "ROW_NUMBER(" in normalized_sql or "RANK(" in normalized_sql or "DENSE_RANK(" in normalized_sql:
        return False
    
    # Don't require LIMIT for queries with WHERE conditions that limit results
    if "WHERE" in normalized_sql and any(condition in normalized_sql for condition in ["=", ">", "<", "LIKE", "IN"]):
        return False
    
    # Don't require LIMIT for GROUP BY queries (they aggregate results)
    if "GROUP BY" in normalized_sql:
        return False
    
    # Don't require LIMIT for queries with explicit LIMIT
    if "LIMIT" in normalized_sql:
        return False
    
    # Check plan DSL context
    if plan_dsl:
        # Don't require LIMIT for window function patterns
        if plan_dsl.get("window_functions"):
            return False
        
        # Don't require LIMIT for aggregation patterns
        if plan_dsl.get("aggregation"):
            return False
    
    # Require LIMIT for simple SELECT queries that could return large datasets
    return "SELECT" in normalized_sql and "FROM" in normalized_sql

def _is_valid_sql_structure(normalized_sql: str) -> bool:
    """
    Check if SQL has valid basic structure.
    
    Args:
        normalized_sql: Normalized SQL string (uppercase)
        
    Returns:
        True if SQL has valid structure
    """
    # Must start with SELECT
    if not normalized_sql.startswith("SELECT"):
        return False
    
    # Must have FROM clause
    if "FROM" not in normalized_sql:
        return False
    
    # Basic syntax checks
    if normalized_sql.count("(") != normalized_sql.count(")"):
        return False
    
    # Check for balanced quotes (simple check)
    single_quotes = normalized_sql.count("'")
    double_quotes = normalized_sql.count('"')
    if single_quotes % 2 != 0 or double_quotes % 2 != 0:
        return False
    
    return True

def _matches_plan_requirements(normalized_sql: str, plan_dsl: Dict[str, Any]) -> bool:
    """
    Check if SQL matches plan DSL requirements.
    
    Args:
        normalized_sql: Normalized SQL string (uppercase)
        plan_dsl: Plan DSL
        
    Returns:
        True if SQL matches plan requirements
    """
    # Check for required tables
    required_tables = plan_dsl.get("tables", [])
    for table in required_tables:
        if table.upper() not in normalized_sql:
            return False
    
    # Check for required columns (if specified)
    required_columns = plan_dsl.get("columns", [])
    if required_columns and required_columns != ["*"]:
        for column in required_columns:
            if column.upper() not in normalized_sql:
                return False
    
    # Check for required conditions
    required_conditions = plan_dsl.get("conditions", [])
    for condition in required_conditions:
        if isinstance(condition, dict):
            column = condition.get("column", "").upper()
            value = condition.get("value", "").upper()
            if column not in normalized_sql or value not in normalized_sql:
                return False
    
    # Check for window functions if required
    if plan_dsl.get("window_functions"):
        if not any(func in normalized_sql for func in ["ROW_NUMBER(", "RANK(", "DENSE_RANK("]):
            return False
    
    # Check for aggregation if required
    aggregation = plan_dsl.get("aggregation")
    if aggregation:
        func = aggregation.get("function", "").upper()
        if func in ["SUM", "COUNT", "AVG", "MAX", "MIN"]:
            if f"{func}(" not in normalized_sql:
                return False
    
    return True


def _select_primary_sql(candidates: List[str], plan_dsl: Dict[str, Any]) -> str:
    """
    Select the primary SQL from candidates using simple heuristics.
    
    Args:
        candidates: Filtered SQL candidates
        plan_dsl: Plan DSL for comparison
        
    Returns:
        Selected primary SQL
    """
    if not candidates:
        return None
    
    # Simple heuristic: prefer shorter, simpler queries
    # In a real implementation, you might use more sophisticated ranking
    
    # Sort by length (shorter is better)
    sorted_candidates = sorted(candidates, key=len)
    
    # Prefer queries that match plan assertions
    for candidate in sorted_candidates:
        candidate_upper = candidate.upper()
        
        # Check if it has required elements from plan
        has_aggregation = plan_dsl.get("aggregation") and any(
            func in candidate_upper for func in ["SUM(", "COUNT(", "AVG(", "MAX(", "MIN("]
        )
        
        # If plan requires aggregation, prefer queries with aggregation
        if plan_dsl.get("aggregation") and has_aggregation:
            return candidate
        
        # If plan doesn't require aggregation, prefer simpler queries
        if not plan_dsl.get("aggregation") and not has_aggregation:
            return candidate
    
    # Fallback to shortest candidate
    return sorted_candidates[0]


def sql_generate_router(state: AppState) -> str:
    """
    Route based on SQL generation results
    
    Routes to:
    - 'validate_diagnose': If SQL was generated successfully
    - 'error': If generation failed
    """
    # Check for errors first
    status = getattr(state, "status", None)
    if status == "ERROR" or getattr(state, "error_message", None):
        logger.warning(f"SQL generation failed, routing to error: {getattr(state, 'error_message', None)}")
        return "error"
    
    # Check for successful generation
    sql = getattr(state, "sql", None)
    sql_candidates = getattr(state, "sql_candidates", None)
    
    if sql and sql_candidates and len(sql_candidates) > 0:
        logger.info("SQL generation successful, routing to validate_diagnose")
        return "validate_diagnose"
    
    logger.warning("No SQL candidates generated, routing to error")
    return "error"


def build_sql_generate_subgraph() -> StateGraph:
    """
    Build the SQL generation subgraph
    
    This subgraph handles:
    1. SQL candidate generation from intent
    2. Plan building and validation
    3. Few-shot example selection
    4. Candidate filtering and selection
    5. Primary SQL output
    
    Returns a subgraph that routes to END with route labels for parent workflow.
    """
    workflow = StateGraph(AppState)
    
    # Add nodes
    workflow.add_node("sql_generate", sql_generate_node)
    
    # Add edges
    workflow.add_edge(START, "sql_generate")
    
    # Add conditional routing - route to END with labels for parent workflow
    workflow.add_conditional_edges(
        "sql_generate",
        sql_generate_router,
        {
            "validate_diagnose": END,  # Route to END, parent will handle
            "error": END  # Route to END, parent will handle
        }
    )
    
    return workflow.compile()


# Export the compiled subgraph
sql_generate_subgraph = build_sql_generate_subgraph()
