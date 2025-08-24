"""
Validate Diagnose Agent Subgraph

This module implements the validate diagnose agent that uses LLM to validate
SQL queries and provide detailed error diagnosis.
"""

from typing import Dict, Any
import logging
from langgraph.graph import StateGraph, START, END

from core.state import AppState, create_error_response, create_success_response
from core.llm_adapter import get_llm_adapter

logger = logging.getLogger(__name__)

def validate_diagnose_node(state: AppState) -> Dict[str, Any]:
    """
    Validate SQL query and provide diagnosis using LLM
    
    This node uses LLM to:
    1. Validate SQL syntax and semantics
    2. Check against database schema
    3. Identify potential issues
    4. Provide improvement suggestions
    """
    try:
        logger.info(f"Validating SQL query: {state.sql[:100] if state.sql else 'None'}...")
        
        # Get LLM client
        llm_client = get_llm_adapter()
        
        # Extract schema context
        schema_context = state.schema_context or {}
        
        # Validate SQL using LLM
        validation_result = llm_client.validate_sql(
            sql_query=state.sql,
            schema_context=schema_context,
            dialect=state.dialect or "sqlite"
        )
        
        logger.info(f"SQL validation completed. Valid: {validation_result.get('is_valid', False)}")
        
        # Extract validation details
        is_valid = validation_result.get("is_valid", False)
        errors = validation_result.get("errors", [])
        warnings = validation_result.get("warnings", [])
        suggestions = validation_result.get("suggestions", [])
        validation_details = validation_result.get("validation_details", {})
        confidence = validation_result.get("confidence", 0.5)
        
        # Determine validation status
        validation_ok = is_valid and len(errors) == 0
        
        # Create validation reasons
        val_reasons = []
        if errors:
            val_reasons.extend([f"ERROR: {error}" for error in errors])
        if warnings:
            val_reasons.extend([f"WARNING: {warning}" for warning in warnings])
        if suggestions:
            val_reasons.extend([f"SUGGESTION: {suggestion}" for suggestion in suggestions])
        
        # Create validation signals
        val_signals = {
            "syntax_ok": validation_details.get("syntax_check") == "passed",
            "schema_ok": validation_details.get("schema_check") == "passed",
            "security_ok": validation_details.get("security_check") == "passed",
            "performance_ok": validation_details.get("performance_check") == "passed"
        }
        
        # Update state with results
        patch = create_success_response(
            status="VALIDATING",
            validation_ok=validation_ok,
            val_reasons=val_reasons,
            val_signals=val_signals,
            val_attempts=(state.val_attempts or 0) + 1
        )
        
        logger.info(f"Validation completed. OK: {validation_ok}, Reasons: {len(val_reasons)}")
        return patch
        
    except Exception as e:
        logger.error(f"SQL validation failed: {str(e)}")
        return create_error_response(
            error_message=f"SQL validation error: {str(e)}",
            status="ERROR",
            validation_ok=False,
            val_reasons=[f"Validation failed: {str(e)}"],
            val_signals={
                "syntax_ok": False,
                "schema_ok": False,
                "security_ok": False,
                "performance_ok": False
            },
            val_attempts=(state.val_attempts or 0) + 1
        )

def validate_diagnose_router(state: AppState) -> str:
    """
    Route based on validation results
    
    Routes to:
    - 'execute': If validation passed
    - 'retry': If validation failed but can be retried
    - 'error_handler': If validation failed completely
    """
    # Check for errors first
    if getattr(state, "status", None) == "ERROR" or getattr(state, "error_message", None):
        logger.warning(f"Validation failed, routing to error handler: {getattr(state, 'error_message', None)}")
        return "error"
    
    if not getattr(state, "validation_ok", False):
        # Check if we should retry
        max_attempts = 2  # Allow 2 validation attempts
        current_attempts = getattr(state, "val_attempts", 0)
        
        if current_attempts < max_attempts:
            logger.info(f"Validation failed, retrying (attempt {current_attempts + 1}/{max_attempts})")
            return "retry"
        else:
            logger.warning(f"Validation failed after {max_attempts} attempts, routing to error handler")
            return "error"
    
    logger.info("Validation successful, routing to execute")
    return "execute"

def build_validate_diagnose_subgraph() -> StateGraph:
    """
    Build the validate diagnose subgraph
    
    This subgraph handles:
    1. SQL validation and safety checks
    2. Error diagnosis and suggestions
    3. Plan patching if needed
    4. Routing based on validation results
    
    Returns a subgraph that routes to END with route labels for parent workflow.
    """
    workflow = StateGraph(AppState)
    
    # Add nodes
    workflow.add_node("validate_diagnose", validate_diagnose_node)
    
    # Add edges
    workflow.add_edge(START, "validate_diagnose")
    
    # Add conditional routing - route to END with labels for parent workflow
    workflow.add_conditional_edges(
        "validate_diagnose",
        validate_diagnose_router,
        {
            "execute": END,  # Route to END, parent will handle
            "retry": END,  # Route to END, parent will handle
            "error": END  # Route to END, parent will handle
        }
    )
    
    return workflow.compile()

# Export the compiled subgraph
validate_diagnose_subgraph = build_validate_diagnose_subgraph()
