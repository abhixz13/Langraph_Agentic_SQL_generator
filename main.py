#!/usr/bin/env python3
"""
main.py — CLI entrypoint for the LangGraph SQL Assistant

This module provides a command-line interface for the SQL Assistant that:
- Loads configuration from environment variables
- Builds and runs the LangGraph workflow
- Supports both one-shot and interactive modes
- Handles human-in-the-loop when ambiguity is detected
- Provides pretty logging and JSON output options
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Import our core modules
from core.config import load_settings, policy_from_settings
from core.state import AppState, initial_state_from_settings, apply_patch, as_dict
from workflows.main_graph import SQLAssistantWorkflow


def setup_logging(log_level: str) -> None:
    """Setup logging with the specified level"""
    try:
        import rich.logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(message)s",
            handlers=[rich.logging.RichHandler(rich_tracebacks=True)]
        )
    except ImportError:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )


def build_graph() -> Any:
    """
    Build and return a compiled LangGraph app that accepts/returns dicts.
    This wraps the SQLAssistantWorkflow to match the expected interface.
    """
    workflow = SQLAssistantWorkflow()
    return workflow.workflow


def print_summary(state_dict: Dict[str, Any], interactive: bool = False) -> None:
    """Print a compact summary of the final state"""
    print("\n" + "="*60)
    print("📊 SQL ASSISTANT SUMMARY")
    print("="*60)
    
    # Basic query info
    print(f"Query: {state_dict.get('user_query', 'N/A')}")
    print(f"Dialect: {state_dict.get('dialect', 'auto-detect')}")
    
    # Route outcome
    validation_ok = state_dict.get('validation_ok', False)
    loop_count = state_dict.get('loop_count', 0)
    max_loops = state_dict.get('max_loops', 2)
    print(f"Validation: {'✅ OK' if validation_ok else '❌ Failed'}")
    print(f"Loops: {loop_count}/{max_loops}")
    
    # Ambiguity handling
    if state_dict.get('ambiguity', False):
        print(f"🤔 Ambiguity detected: {state_dict.get('clarifying_question', 'N/A')}")
        if interactive and state_dict.get('clarifying_answer'):
            print(f"💬 User response: {state_dict.get('clarifying_answer')}")
    
    # Results
    if state_dict.get('result_sample'):
        result_sample = state_dict.get('result_sample', [])
        print(f"📋 Results: {len(result_sample)} rows")
        if result_sample:
            print("Sample row:", result_sample[0])
    elif state_dict.get('exec_preview'):
        exec_preview = state_dict.get('exec_preview', [])
        print(f"👀 Preview: {len(exec_preview)} rows")
        if exec_preview:
            print("Preview row:", exec_preview[0])
    
    # Error handling
    if state_dict.get('error'):
        print(f"❌ Error: {state_dict.get('error')}")
    
    # Safety hint
    if not validation_ok and not state_dict.get('result_sample') and not state_dict.get('exec_preview'):
        print("\n💡 Tip: Try rephrasing the query or confirming table names.")
    
    print("="*60)


def run_one_shot(app: Any, state: AppState, dump_json: Optional[str] = None) -> int:
    """Run the graph once and print summary"""
    try:
        # Convert AppState to dict and invoke the graph
        state_dict = state.model_dump()
        print(f"DEBUG: State dict keys: {list(state_dict.keys())}")
        final_dict = app.invoke(state_dict)
        
        # Print summary
        print_summary(final_dict)
        
        # Dump JSON if requested
        if dump_json:
            with open(dump_json, 'w', encoding='utf-8') as f:
                json.dump(final_dict, f, indent=2, default=str)
            print(f"💾 State saved to: {dump_json}")
        
        return 0
        
    except Exception as e:
        logging.error(f"One-shot execution failed: {e}")
        return 1


def run_interactive(app: Any, state: AppState, dump_json: Optional[str] = None) -> int:
    """Run interactive mode with human-in-the-loop"""
    try:
        current_state = state
        
        while True:
            # Convert AppState to dict and run the graph
            state_dict = current_state.model_dump()
            result_dict = app.invoke(state_dict)
            
            # Check for ambiguity
            if result_dict.get('ambiguity', False):
                clarifying_question = result_dict.get('clarifying_question', 'Please clarify:')
                print(f"\n🤔 {clarifying_question}")
                
                # Get user input
                try:
                    user_input = input("> answer: ").strip()
                    if not user_input:
                        print("❌ No input provided. Exiting.")
                        return 1
                    
                    # Apply patch with user's clarifying answer
                    current_state = apply_patch(current_state, {
                        "clarifying_answer": user_input
                    })
                    
                    print("🔄 Re-running with clarification...")
                    continue
                    
                except KeyboardInterrupt:
                    print("\n❌ Interrupted by user.")
                    return 1
                except EOFError:
                    print("\n❌ End of input.")
                    return 1
            
            # No ambiguity, print final summary
            print_summary(result_dict, interactive=True)
            
            # Dump JSON if requested
            if dump_json:
                with open(dump_json, 'w', encoding='utf-8') as f:
                    json.dump(result_dict, f, indent=2, default=str)
                print(f"💾 State saved to: {dump_json}")
            
            return 0
            
    except Exception as e:
        logging.error(f"Interactive execution failed: {e}")
        return 1


def main() -> int:
    """Main CLI entrypoint"""
    parser = argparse.ArgumentParser(
        description="LangGraph SQL Assistant CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --nl "show me top 5 users by revenue"
  python main.py --interactive --nl "users with orders"
  python main.py --nl "products" --gen-k 5 --dialect postgres --dump-json result.json
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--nl", "-q",
        required=True,
        help="Natural language query (required)"
    )
    
    # Mode selection
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enable interactive human-in-the-loop mode"
    )
    
    # State overrides
    parser.add_argument(
        "--gen-k",
        type=int,
        help="Override gen_k (number of SQL candidates to generate)"
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        help="Override max_loops (maximum validation retry attempts)"
    )
    parser.add_argument(
        "--dialect",
        choices=["postgres", "mysql", "sqlite"],
        help="Override SQL dialect hint"
    )
    
    # Output options
    parser.add_argument(
        "--dump-json",
        help="Path to save final state as JSON"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (defaults to settings.log_level)"
    )
    
    args = parser.parse_args()
    
    try:
        # A) Initialization
        print("🔧 Loading configuration...")
        try:
            settings = load_settings()
            print(f"DEBUG: Settings loaded successfully: {type(settings)}")
        except Exception as e:
            print(f"DEBUG: Error loading settings: {e}")
            raise
        
        # Setup logging
        log_level = args.log_level or settings.log_level
        setup_logging(log_level)
        
        # Build initial state
        print("🏗️  Building initial state...")
        try:
            state = initial_state_from_settings(settings, user_query=args.nl)
            print(f"DEBUG: State created successfully: {type(state)}")
        except Exception as e:
            print(f"DEBUG: Error creating state: {e}")
            raise
        
        # Apply CLI overrides
        if args.dialect:
            state = apply_patch(state, {"dialect": args.dialect})
        if args.gen_k is not None:
            state = apply_patch(state, {"gen_k": args.gen_k})
        if args.max_loops is not None:
            state = apply_patch(state, {"max_loops": args.max_loops})
        
        # Build graph
        print("🔗 Building workflow graph...")
        try:
            app = build_graph()
            print(f"DEBUG: Graph built successfully: {type(app)}")
        except Exception as e:
            print(f"DEBUG: Error building graph: {e}")
            raise
        
        # B) Run mode
        if args.interactive:
            print("🔄 Starting interactive mode...")
            return run_interactive(app, state, args.dump_json)
        else:
            print("⚡ Starting one-shot mode...")
            return run_one_shot(app, state, args.dump_json)
            
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
