from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any, Callable
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import our components
from core.state import AppState, create_error_response, create_success_response, apply_patch
from core.config import Settings, load_settings

# Import subgraphs
from agents.schema_context.subgraph import schema_context_subgraph
from agents.interpret_plan.subgraph import interpret_plan_subgraph
from agents.sql_generate.subgraph import sql_generate_subgraph
from agents.validate_diagnose.subgraph import validate_diagnose_subgraph

def wrap_node(node_func: Callable) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Wrapper to convert dict state to AppState for node execution.
    
    This allows nodes to work with AppState objects while LangGraph
    continues to use dict state internally.
    """
    def wrapper(state_input) -> Dict[str, Any]:
        try:
            # Handle both dict and AppState inputs
            if isinstance(state_input, AppState):
                app_state = state_input
            else:
                # Convert dict to AppState
                app_state = AppState(**state_input)
            
            # Call the original node function with AppState
            patch = node_func(app_state)
            
            # Apply the patch to get new AppState
            new_app_state = apply_patch(app_state, patch)
            
            # Convert back to dict for LangGraph
            return new_app_state.model_dump()
            
        except Exception as e:
            # Return error in dict format
            if isinstance(state_input, AppState):
                state_dict = state_input.model_dump()
            else:
                state_dict = state_input
            return {
                **state_dict,
                "error": f"Node execution failed: {str(e)}"
            }
    
    return wrapper

class SQLAssistantWorkflow:
    """Main SQL Assistant workflow using LangGraph"""
    
    def __init__(self):
        self.settings = load_settings()
        self.workflow = self.build_workflow()
    
    def build_workflow(self) -> StateGraph:
        """Build the main workflow graph"""
        workflow = StateGraph(AppState)
        
        # Add subgraphs
        workflow.add_node("schema_context", schema_context_subgraph)
        workflow.add_node("interpret_plan", interpret_plan_subgraph)
        workflow.add_node("sql_generate", sql_generate_subgraph)
        workflow.add_node("validate_diagnose", validate_diagnose_subgraph)
        
        # Add individual nodes
        workflow.add_node("execute", wrap_node(self.execute_node))
        workflow.add_node("present", wrap_node(self.present_node))
        workflow.add_node("human_approval", wrap_node(self.human_approval_node))
        workflow.add_node("error_handler", wrap_node(self.error_handler_node))
        
        # Add entry point
        workflow.add_edge(START, "schema_context")
        
        # Add unconditional edges
        workflow.add_edge("schema_context", "interpret_plan")
        workflow.add_edge("execute", "present")
        workflow.add_edge("human_approval", "interpret_plan")
        workflow.add_edge("error_handler", END)
        workflow.add_edge("present", END)
        
        # Add conditional edges for subgraph routing
        workflow.add_conditional_edges(
            "interpret_plan",
            self.interpret_plan_router,
            {
                "sql_generate": "sql_generate",
                "human_approval": "human_approval",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "sql_generate",
            self.sql_generate_router,
            {
                "validate_diagnose": "validate_diagnose",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_diagnose",
            self.validate_diagnose_router,
            {
                "execute": "execute",
                "retry": "sql_generate",
                "error": "error_handler"
            }
        )
        
        return workflow.compile()
    
    # Node implementations - now work with AppState objects and return patches
    
    def execute_node(self, state: AppState) -> Dict[str, Any]:
        """Execute SQL safely"""
        try:
            validation_ok = state.validation_ok
            if not validation_ok:
                return create_error_response(
                    error_message="Query failed validation",
                    exec_preview=None,
                    result_sample=None,
                    exec_rows=0
                )
            
            # Execute SQL (placeholder - returns sample data)
            sql = state.sql or ""
            result_sample = [
                {"customer_name": "Acme Corp", "total_revenue": 150000.00},
                {"customer_name": "TechStart Inc", "total_revenue": 125000.00}
            ]
            
            exec_preview = {
                "execution_time": 0.1,
                "row_count": len(result_sample),
                "column_names": ["customer_name", "total_revenue"],
                "sql": sql
            }
            
            return create_success_response(
                status="EXECUTED",
                exec_preview=exec_preview,
                result_sample=result_sample,
                exec_rows=len(result_sample)
            )
            
        except Exception as e:
            return create_error_response(
                error_message=f"Execution error: {str(e)}",
                exec_preview=None,
                result_sample=None,
                exec_rows=0
            )
    
    def present_node(self, state: AppState) -> Dict[str, Any]:
        """Format and present results"""
        try:
            # Check if execution was successful
            error = state.error
            if error:
                return {"error": error}
            
            # Results are already in result_sample
            return {}  # No additional state changes needed
            
        except Exception as e:
            return {"error": f"Presentation error: {str(e)}"}
    
    def human_approval_node(self, state: AppState) -> Dict[str, Any]:
        """Handle human approval for ambiguous queries"""
        try:
            # In real implementation, this would prompt the user
            return create_success_response(
                status="APPROVED",
                clarifying_answer="Query approved",
                ambiguity=False
            )
            
        except Exception as e:
            return create_error_response(
                error_message=f"Human approval error: {str(e)}",
                status="ERROR"
            )
    
    def error_handler_node(self, state: AppState) -> Dict[str, Any]:
        """Handle errors gracefully"""
        try:
            # Generate helpful error suggestions
            val_reasons = state.val_reasons or []
            error_suggestions = []
            
            if val_reasons:
                error_suggestions.extend(val_reasons)
            
            # Add general suggestions if no specific reasons
            if not error_suggestions:
                error_suggestions.append("Try rephrasing your query or check table names")
            
            return create_error_response(
                error_message="Query processing failed",
                status="ERROR",
                val_reasons=error_suggestions
            )
            
        except Exception as e:
            return create_error_response(
                error_message=f"Error handling failed: {str(e)}",
                status="ERROR"
            )
    
    # Router functions - must be pure (read-only) and work with AppState
    def interpret_plan_router(self, state: AppState) -> str:
        """Route after planning"""
        status = getattr(state, "status", None)
        error_message = getattr(state, "error_message", None)
        ambiguity = getattr(state, "ambiguity", False)
        plan_ok = getattr(state, "plan_ok", False)
        plan_confidence = getattr(state, "plan_confidence", 0.0)
        
        # Check for errors first
        if status == "ERROR" or error_message:
            logger.warning(f"Plan interpretation failed, routing to error handler: {error_message}")
            return "error"
        
        # Check for ambiguity or low confidence
        if ambiguity or plan_confidence < 0.3:
            logger.info(f"Ambiguity detected or low confidence, routing to human approval")
            return "human_approval"
        
        # Check if plan is acceptable
        if plan_ok:
            logger.info(f"Plan is acceptable, routing to SQL generation")
            return "sql_generate"
        
        # Default to human approval for unclear cases
        logger.info(f"Plan unclear, routing to human approval")
        return "human_approval"
    
    def sql_generate_router(self, state: AppState) -> str:
        """Route after SQL generation"""
        status = getattr(state, "status", None)
        error_message = getattr(state, "error_message", None)
        sql_candidates = getattr(state, "sql_candidates", [])
        sql = getattr(state, "sql", None)
        
        # Check for errors first
        if status == "ERROR" or error_message:
            logger.warning(f"SQL generation failed, routing to error handler: {error_message}")
            return "error"
        
        if not sql_candidates or not sql:
            logger.warning(f"No SQL candidates or primary SQL found, routing to error handler")
            return "error"
        
        logger.info("SQL generation successful, routing to validation")
        return "validate_diagnose"
    
    def validate_diagnose_router(self, state: AppState) -> str:
        """Route after validation"""
        status = getattr(state, "status", None)
        error_message = getattr(state, "error_message", None)
        validation_ok = getattr(state, "validation_ok", False)
        val_attempts = getattr(state, "val_attempts", 0)
        max_attempts = 2  # Allow 2 validation attempts
        
        # Check for errors first
        if status == "ERROR" or error_message:
            logger.warning(f"Validation failed, routing to error handler: {error_message}")
            return "error"
        
        # Check validation status
        if not validation_ok:
            if val_attempts < max_attempts:
                logger.info(f"Validation failed, retrying (attempt {val_attempts + 1}/{max_attempts})")
                return "retry"
            else:
                logger.warning(f"Validation failed after {max_attempts} attempts, routing to error handler")
                return "error"
        
        logger.info("Validation successful, routing to execute")
        return "execute"
    
    def run(self, user_query: str, dialect: str = "sqlite", session_id: str = None) -> Dict[str, Any]:
        """Run the complete workflow"""
        try:
            # Check if API key is set (warn instead of hard fail for now)
            if not os.getenv("OPENAI_API_KEY"):
                print("Warning: OPENAI_API_KEY not found. LLM features will be stubbed.")
            
            # Initialize state
            initial_state = {
                "user_query": user_query,
                "dialect": dialect,
                "session_id": session_id,
                "database_url": "sqlite:///data/raw_data.db",
                "semantic_dir": "data/semantic",
                "policy": {
                    "schema_cache_enabled": False,
                    "intent": {
                        "min_confidence": 0.3,
                        "default_confidence": 0.7,
                        "fallback_confidence": 0.1,
                        "default_action": "SELECT",
                        "valid_actions": ["SELECT", "COUNT", "AGGREGATE", "SEARCH", "COMPARE"],
                        "default_complexity": "simple",
                        "valid_complexities": ["simple", "moderate", "complex"],
                        "max_tables": 5,
                        "max_columns_per_table": 25
                    }
                },
                "gen_k": self.settings.gen_k,
                "max_loops": self.settings.max_loops,
                "loop_count": 0
            }
            
            # Run workflow
            result = self.workflow.invoke(initial_state)
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error",
                "user_query": user_query
            }
